"""Source-bound BRiG diagnostic artifacts; validation only, no certified test path."""
from pathlib import Path
import torch

from .brig import BRiGPolicy, GroupQ, fit_brig
from .io import file_hash, fresh_dir, read_json, write_json
from .provenance import make_fit_record, validate_target_exclusion


def artifact_identity(directory):
    return {name: file_hash(Path(directory) / name) for name in
            ("brig.pt", "training.json", "receipt.json")}


def _bindings(models_dir, cache_dir, schema, manifest, parent):
    return dict(schema_hash=schema.hash, cache_sha256=manifest["responses_sha256"],
                cache_manifest_sha256=file_hash(Path(cache_dir) / "manifest.json"),
                parent_receipt_sha256=file_hash(Path(models_dir) / "receipt.json"),
                parent_config_sha256=file_hash(Path(models_dir) / "config.json"),
                parent_models_sha256=parent["models_sha256"],
                head_component_sha256=parent["head_component_sha256"],
                head_artifact_id=parent["head_artifact_id"])


def train_brig_controller(cache_dir, models_dir, out, *, config=None):
    from .pipeline import code_fingerprint, load_model_bundle
    cfg = dict(config or {})
    source = code_fingerprint()
    schema, rows, manifest, parent_cfg, parent, head, _, _ = load_model_bundle(
        models_dir, cache_dir, cfg.get("device", "cpu"))
    cfg.setdefault("seed", parent_cfg["seed"])
    groups = sorted({r["group_id"] for r in rows if r["split"] == "policy_fit"})
    validate_target_exclusion(parent["provenance"], artifact_ids=[parent["head_artifact_id"]],
                              target_group_ids=groups)
    bindings = _bindings(models_dir, cache_dir, schema, manifest, parent)
    out = fresh_dir(out)
    policy, report = fit_brig(rows, head, schema, cfg, excluded_head_group_ids=groups)
    report.update(bindings)
    report.update(source_code_hash=source, head_exclusion_provenance_checked=True,
                  response_source=manifest["response_source"], paper_evidence=False)
    torch.save({str(b): {k: v.detach().cpu() for k, v in m.state_dict().items()}
                for b, m in policy.models.items()}, out / "brig.pt")
    write_json(out / "training.json", report)
    # Re-read parent and cache to detect concurrent mutation before completion.
    s2, _, m2, _, p2, _, _, _ = load_model_bundle(models_dir, cache_dir, cfg.get("device", "cpu"))
    if source != code_fingerprint() or bindings != _bindings(models_dir, cache_dir, s2, m2, p2):
        raise ValueError("BRiG training source or parent changed")
    checkpoint = file_hash(out / "brig.pt")
    artifact_id = "brig:" + checkpoint
    provenance = parent["provenance"] + [make_fit_record(
        artifact_id, supervised_group_ids=groups, parent_ids=[parent["head_artifact_id"]],
        metadata={"role": "policy_fit", "method": "grouped_brig"})]
    receipt = dict(format="cbmjev-brig-v1", **bindings, source_code_hash=source,
                   checkpoint_sha256=checkpoint, report_sha256=file_hash(out / "training.json"),
                   artifact_id=artifact_id, provenance=provenance, validation_only=True,
                   test_evaluated=False)
    write_json(out / "receipt.json", receipt)  # Completion marker last.
    return {"out": str(out), "checkpoint_sha256": checkpoint, "validation_only": True}


def load_brig_controller(directory, models_dir, cache_dir, *, device="cpu"):
    from .pipeline import code_fingerprint, load_model_bundle
    directory = Path(directory)
    schema, rows, manifest, _, parent, _, _, _ = load_model_bundle(models_dir, cache_dir, device)
    receipt, report = read_json(directory / "receipt.json"), read_json(directory / "training.json")
    if receipt.get("format") != "cbmjev-brig-v1" or receipt.get("validation_only") is not True:
        raise ValueError("unsupported BRiG artifact")
    if receipt["checkpoint_sha256"] != file_hash(directory / "brig.pt") or receipt["report_sha256"] != file_hash(directory / "training.json"):
        raise ValueError("BRiG checkpoint/report changed")
    for key, expected in _bindings(models_dir, cache_dir, schema, manifest, parent).items():
        if receipt.get(key) != expected or report.get(key) != expected:
            raise ValueError("BRiG parent binding mismatch: " + key)
    if receipt["source_code_hash"] != report["source_code_hash"] or receipt["source_code_hash"] != code_fingerprint():
        raise ValueError("BRiG source differs from training release")
    groups = sorted({r["group_id"] for r in rows if r["split"] == "policy_fit"})
    validate_target_exclusion(parent["provenance"], artifact_ids=[parent["head_artifact_id"]], target_group_ids=groups)
    expected_id = "brig:" + receipt["checkpoint_sha256"]
    expected_provenance = parent["provenance"] + [make_fit_record(
        expected_id, supervised_group_ids=groups, parent_ids=[parent["head_artifact_id"]],
        metadata={"role": "policy_fit", "method": "grouped_brig"})]
    if receipt["artifact_id"] != expected_id or receipt["provenance"] != expected_provenance:
        raise ValueError("BRiG training ancestry mismatch")
    validation_groups = sorted({r["group_id"] for r in rows if r["split"] == "validation"})
    validate_target_exclusion(receipt["provenance"], artifact_ids=[expected_id], target_group_ids=validation_groups)
    cfg = report["config"]
    payload = torch.load(directory / "brig.pt", map_location=device, weights_only=True)
    if set(payload) != {str(b) for b in range(1, cfg["max_budget"] + 1)}:
        raise ValueError("BRiG checkpoint budget mismatch")
    models = {}
    for budget in range(1, cfg["max_budget"] + 1):
        model = GroupQ(schema, cfg["hidden"]).to(device)
        model.load_state_dict(payload[str(budget)], strict=True)
        if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
            raise ValueError("nonfinite BRiG checkpoint")
        models[budget] = model.eval().requires_grad_(False)
    return BRiGPolicy(schema, models), report

"""Authenticated local static-mask diagnostic artifacts, validation only."""
from pathlib import Path
import torch

from .brig_artifacts import _bindings
from .contracts import stable_hash
from .io import file_hash, fresh_dir, read_json, write_json
from .provenance import make_fit_record, validate_target_exclusion
from .static_mask import StaticGroupMask, fit_static_mask, _head_digest


def static_mask_identity(directory):
    return {n: file_hash(Path(directory) / n) for n in ("mask.pt", "training.json", "receipt.json")}


def _lineage(parent, artifact_id, groups, endpoint):
    return parent["provenance"] + [make_fit_record(
        artifact_id, supervised_group_ids=[] if endpoint else groups,
        parent_ids=[parent["head_artifact_id"]], fit_kind="derived" if endpoint else "supervised",
        metadata={"role": "fixed_endpoint" if endpoint else "policy_fit", "method": "learned_static_mask"})]


def train_static_mask(cache_dir, models_dir, out, *, config=None):
    from .pipeline import code_fingerprint, load_model_bundle
    cfg = dict(config or {})
    source = code_fingerprint()
    schema, rows, manifest, parent_config, parent, head, _, _ = load_model_bundle(
        models_dir, cache_dir, cfg.get("device", "cpu"))
    cfg.setdefault("seed", parent_config["seed"])
    groups = sorted({r["group_id"] for r in rows if r["split"] == "policy_fit"})
    validate_target_exclusion(parent["provenance"], artifact_ids=[parent["head_artifact_id"]], target_group_ids=groups)
    bindings = _bindings(models_dir, cache_dir, schema, manifest, parent)
    out = fresh_dir(out)
    mask, report = fit_static_mask(rows, head, schema, cfg, excluded_head_group_ids=groups)
    report.update(bindings)
    report.update(source_code_hash=source, head_exclusion_provenance_checked=True,
                  input_provenance_checked=True, response_source=manifest["response_source"])
    torch.save({k: v.detach().cpu() for k, v in mask.state_dict().items()}, out / "mask.pt")
    check = torch.load(out / "mask.pt", map_location="cpu", weights_only=True)
    if any(not torch.equal(check[k], v.detach().cpu()) for k, v in mask.state_dict().items()):
        raise ValueError("static-mask checkpoint roundtrip mismatch")
    write_json(out / "training.json", report)
    s2, _, m2, _, p2, _, _, _ = load_model_bundle(models_dir, cache_dir, cfg.get("device", "cpu"))
    if source != code_fingerprint() or bindings != _bindings(models_dir, cache_dir, s2, m2, p2):
        raise ValueError("static-mask training source or parent changed")
    checkpoint = file_hash(out / "mask.pt")
    artifact_id = "static-mask:" + checkpoint
    write_json(out / "receipt.json", dict(format="cbmjev-static-mask-v1", **bindings,
        source_code_hash=source, checkpoint_sha256=checkpoint, report_sha256=file_hash(out / "training.json"),
        artifact_id=artifact_id, provenance=_lineage(parent, artifact_id, groups, report["endpoint_no_training"]),
        validation_only=True, test_evaluated=False))
    return {"out": str(out), "checkpoint_sha256": checkpoint,
            "selected_group_ids": list(mask.selected_groups()), "validation_only": True}


def load_static_mask(directory, models_dir, cache_dir, *, device="cpu"):
    from .pipeline import code_fingerprint, load_model_bundle
    directory = Path(directory)
    schema, rows, manifest, _, parent, head, _, _ = load_model_bundle(models_dir, cache_dir, device)
    receipt, report = read_json(directory / "receipt.json"), read_json(directory / "training.json")
    if receipt.get("format") != "cbmjev-static-mask-v1" or receipt.get("validation_only") is not True:
        raise ValueError("unsupported static-mask artifact")
    if receipt["checkpoint_sha256"] != file_hash(directory / "mask.pt") or receipt["report_sha256"] != file_hash(directory / "training.json"):
        raise ValueError("static-mask checkpoint/report changed")
    for key, expected in _bindings(models_dir, cache_dir, schema, manifest, parent).items():
        if report.get(key) != expected or receipt.get(key) != expected:
            raise ValueError("static-mask parent binding mismatch: " + key)
    if receipt["source_code_hash"] != report["source_code_hash"] or receipt["source_code_hash"] != code_fingerprint():
        raise ValueError("static-mask source changed")
    policy_rows = [r for r in rows if r["split"] == "policy_fit"]
    groups = sorted({r["group_id"] for r in policy_rows})
    validate_target_exclusion(parent["provenance"], artifact_ids=[parent["head_artifact_id"]], target_group_ids=groups)
    cfg = report["config"]
    relaxation = cfg.get("relaxation", "sigmoid")
    mask = StaticGroupMask(schema, cfg["k"], seed=cfg["seed"], temperature=cfg["temperature"], device=device,
                           relaxation=relaxation)
    if (report["method"] != "global_static_ST_topK_" + relaxation + "_v1"
            or report["estimator"] != "biased_straight_through_hard_topK_forward_" + relaxation + "_backward"
            or report["backward_preserves_cardinality"] != (relaxation == "cardinality")):
        raise ValueError("static-mask relaxation/report mismatch")
    endpoint = cfg["k"] in (0, schema.num_groups)
    expected_id = "static-mask:" + receipt["checkpoint_sha256"]
    if (receipt["artifact_id"] != expected_id or receipt["provenance"] != _lineage(parent, expected_id, groups, endpoint)
            or report["endpoint_no_training"] != endpoint or report["head_tensor_sha256"] != _head_digest(head.network)
            or report["ordered_training_rows_sha256"] != stable_hash([(r["sample_id"], r["group_id"], r["z"], r["y"]) for r in policy_rows])
            or report.get("head_exclusion_provenance_checked") is not True or report.get("input_provenance_checked") is not True):
        raise ValueError("static-mask training ancestry mismatch")
    validate_target_exclusion(receipt["provenance"], artifact_ids=[expected_id],
                              target_group_ids=sorted({r["group_id"] for r in rows if r["split"] == "validation"}))
    payload = torch.load(directory / "mask.pt", map_location=device, weights_only=True)
    if not torch.equal(payload["tie_rank"], mask.tie_rank):
        raise ValueError("static-mask tie seed mismatch")
    mask.load_state_dict(payload, strict=True)
    if list(mask.selected_groups()) != report["selected_group_ids"]:
        raise ValueError("static-mask reported selection mismatch")
    return mask.eval().requires_grad_(False), report

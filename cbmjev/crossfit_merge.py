"""Merge complete nested outer folds; not a final responder/deployment bundle."""
from pathlib import Path

import torch

from .config import learning_config, resolve_config
from .contracts import stable_hash
from .crossfit import verify_crossfit_prepared
from .crossfit_artifacts import load_crossfit_head, save_crossfit_head
from .crossfit_cache import crossfit_cache_source_binding, load_crossfit_cache
from .crossfit_targets import open_target_package
from .crossfit_training import (_controller_digest, _merge_provenance, _rows_hash,
                               _validate_target_package,
                               _validate_target_package_metadata,
                               _validate_target_package_lineage,
                               fit_controller_from_targets, fit_head_only)
from .io import file_hash, fresh_dir, read_json, write_json
from .learning import ActionController, normalize_config, schema_signature
from .pipeline import code_fingerprint, load_prepared


def _files(directory):
    return {str(p.relative_to(directory)): file_hash(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file() and p != directory / "receipt.json"}


def _outer_receipt(directory):
    receipt = read_json(directory / "receipt.json")
    unsigned = dict(receipt)
    if unsigned.pop("receipt_hash", None) != stable_hash(unsigned):
        raise ValueError("outer receipt hash mismatch")
    if (receipt.get("format") != "cbmjev-outer-fold-execution-v1"
            or receipt.get("status") != "COMPLETE"
            or receipt.get("scope") != "SINGLE_OUTER_FOLD_NOT_FINAL_DEPLOYMENT"):
        raise ValueError("outer execution is not complete")
    # Enumerating actual files also rejects omitted hashes and unexpected files.
    if receipt.get("files_sha256") != _files(directory):
        raise ValueError("outer execution files changed or missing")
    return receipt


def load_outer_folds(prepared, planned, outer_dirs, *, responder_source_dir=None,
                     validate_target_records=True):
    """Validate full plan coverage and return bound packages and OOF head inputs.

    All folds must share the same training source/configuration. The merger may
    be newer source: its own fingerprint is recorded separately when training.
    Non-observed task labels follow the cache's explicit exclusion policy.
    """
    prepared, planned = Path(prepared), Path(planned)
    verified = verify_crossfit_prepared(prepared, planned)
    plan = read_json(planned / "plan.json")
    schema, _, _ = load_prepared(prepared)
    directories = [Path(p).resolve() for p in outer_dirs]
    if len(directories) != plan["outer_folds"] or len(set(directories)) != len(directories):
        raise ValueError("exactly all distinct outer folds are required")
    expected = {"plan_hash": verified["plan_hash"], "schema_hash": schema.hash,
                "plan_sha256": file_hash(planned / "plan.json"),
                "fold_manifest_sha256": file_hash(planned / "fold_manifest.jsonl"),
                "prepared_files_sha256": plan["receipt"]["prepared_files_sha256"]}
    entries, seen, common = [], set(), None
    for directory in directories:
        receipt = _outer_receipt(directory)
        fold = receipt.get("outer_fold")
        if type(fold) is not int or not 0 <= fold < plan["outer_folds"] or fold in seen:
            raise ValueError("duplicate or invalid outer fold")
        seen.add(fold)
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise ValueError("outer receipt differs from current plan: " + key)
        config = read_json(directory / "config.json")
        options = read_json(directory / "responder_options.json")
        if (receipt.get("config_hash") != stable_hash(config)
                or receipt.get("responder_options_hash") != stable_hash(options)
                or config != resolve_config(config)):
            raise ValueError("outer configuration hash mismatch")
        identity = {key: receipt.get(key) for key in (
            "config_hash", "responder_options_hash", "source_code_hash",
            "responder_seed_policy", "responder_seed", "task_seed")}
        if common is not None and identity != common:
            raise ValueError("outer folds use different source/configuration/seeds")
        common = identity
        cfg = normalize_config(learning_config(config), schema)
        package = open_target_package(directory / "action_targets.json")
        if validate_target_records:
            _validate_target_package(package, schema, cfg)
        else:
            _validate_target_package_metadata(package, schema, cfg)
            _validate_target_package_lineage(package)
        _, rows, manifest = load_crossfit_cache(
            directory / "outer/cache", prepared, planned, directory / "outer/responder",
            responder_source_dir=responder_source_dir)
        if (manifest["crossfit"]["stage"] != "outer"
                or manifest["crossfit"]["outer_fold"] != fold):
            raise ValueError("outer cache is assigned to a different fold")
        _, report = load_crossfit_head(directory / "head", schema)
        expected_package = {
            "target_group_ids": sorted({r["group_id"] for r in rows}),
            "target_sample_ids": sorted(r["sample_id"] for r in rows),
            "target_rows_sha256": _rows_hash(rows),
            "response_artifact_by_group": manifest["response_artifact_by_group"],
            "head_artifact_id": report["head_artifact_id"],
            "head_component_sha256": report["head_component_sha256"],
        }
        if any(package.get(k) != v for k, v in expected_package.items()):
            raise ValueError("outer targets differ from persisted head/cache")
        if (receipt.get("head_artifact_id") != package["head_artifact_id"]
                or receipt.get("target_artifact_id") != package["target_artifact_id"]
                or receipt.get("num_target_records") != package["record_count"]
                or report["config"] != cfg):
            raise ValueError("outer artifact receipt mismatch")
        # Reject provenance identities rewritten independently of source receipts.
        merged = _merge_provenance(package["provenance"], report["provenance"],
                                   manifest["provenance"])
        if len(merged) != len(package["provenance"]):
            raise ValueError("outer package omits source provenance")
        entries.append((fold, directory, receipt, package, rows, manifest, config))
    entries.sort(key=lambda entry: entry[0])
    # Tensor-identical legacy artifacts with conflicting fit histories must fail
    # closed, not have their ancestry silently unioned or overwritten.
    _merge_provenance(*(entry[3]["provenance"] for entry in entries))
    rows, assignments, provenance, samples, groups = [], {}, [], set(), set()
    for _, _, _, package, fold_rows, manifest, _ in entries:
        if samples.intersection(package["target_sample_ids"]) or groups.intersection(package["target_group_ids"]):
            raise ValueError("overlap across outer targets")
        samples.update(package["target_sample_ids"])
        groups.update(package["target_group_ids"])
        rows.extend(fold_rows)
        assignments.update(manifest["response_artifact_by_group"])
        provenance.extend(manifest["provenance"])
    rows.sort(key=lambda row: row["sample_id"])
    bridge_bindings = None
    if responder_source_dir is not None:
        bridge_bindings = [{"outer_fold": entry[0], "directory": str(entry[1]),
            "outer_cache": crossfit_cache_source_binding(entry[1] / "outer/cache",
                entry[1] / "outer/responder", responder_source_dir)}
            for entry in entries]
    return schema, {"config": entries[0][-1],
                    "packages": [entry[3] for entry in entries], "rows": rows,
                    "response_artifact_by_group": assignments,
                    "provenance": _merge_provenance(provenance),
                    "sources": [{"outer_fold": e[0], "directory": str(e[1]),
                                 "receipt_sha256": file_hash(e[1] / "receipt.json"),
                                 "receipt_hash": e[2]["receipt_hash"]} for e in entries],
                    "plan_binding": expected, "outer_training_identity": common,
                    "response_source_bindings": bridge_bindings}


def load_merged_controller(directory, schema, *, device="cpu"):
    """Read a tensor-only controller artifact and verify all component bindings."""
    directory = Path(directory)
    receipt = read_json(directory / "receipt.json")
    core = dict(receipt)
    if core.pop("receipt_hash", None) != stable_hash(core):
        raise ValueError("merged controller receipt hash mismatch")
    if (receipt.get("format") != "cbmjev-crossfit-merge-v1"
            or receipt.get("status") != "COMPLETE"
            or receipt.get("schema_signature") != schema_signature(schema)
            or receipt.get("files_sha256") != _files(directory)):
        raise ValueError("merged controller format/schema/files mismatch")
    payload = torch.load(directory / "controller.pt", map_location="cpu", weights_only=True)
    report = read_json(directory / "controller_report.json")
    if (payload.get("format") != "cbmjev-crossfit-controller-v1"
            or payload.get("schema_signature") != schema_signature(schema)
            or payload.get("config") != report.get("config")):
        raise ValueError("merged controller checkpoint metadata mismatch")
    controller = ActionController(schema, {**payload["config"], "device": device})
    controller.network.load_state_dict(payload["state_dict"], strict=True)
    controller.config = dict(payload["config"])
    controller.network.eval().requires_grad_(False)
    digest = _controller_digest(controller, schema)
    if (report.get("controller_component_sha256") != digest
            or report.get("controller_artifact_id") != "controller:" + digest
            or receipt.get("controller_artifact_id") != report["controller_artifact_id"]):
        raise ValueError("merged controller tensor identity mismatch")
    from .provenance import make_fit_record
    provenance = _merge_provenance(report["provenance"])
    expected_record = make_fit_record(report["controller_artifact_id"],
        supervised_group_ids=(), parent_ids=report["target_artifact_ids"],
        fit_kind="derived", metadata={"protocol": "cbmjev-nested-oof-controller-v1",
            "component_sha256": digest,
            "target_package_sha256": report["target_package_sha256"]})
    matching = [r for r in provenance if r["artifact_id"] == report["controller_artifact_id"]]
    if (matching != [expected_record]
            or report.get("training_protocol") != "cbmjev-nested-oof-controller-v1"
            or report.get("schema_signature") != schema_signature(schema)
            or report.get("objective") != controller.objective):
        raise ValueError("merged controller provenance/report mismatch")
    return controller, report


def merge_outer_folds(prepared, planned, outer_dirs, out, *, fit_final_head=True,
                      responder_source_dir=None, controller_batch_size=None):
    """Fit unified controller and optionally final head on all outer OOF rows.

    Output deliberately lacks a final responder, deployment costs/static order,
    calibration and evaluation; it is NOT a deployable model bundle. Output is
    new-only and completion is committed only after source revalidation. Legacy
    tensor-only fit identities that collide across distinct fit histories are
    rejected; this API never resolves collisions by discarding ancestry.
    """
    if type(fit_final_head) is not bool:
        raise ValueError("fit_final_head must be boolean")
    if (controller_batch_size is not None
            and (type(controller_batch_size) is not int or controller_batch_size < 1)):
        raise ValueError("controller_batch_size must be a positive integer")
    out = Path(out)
    if out.exists():
        raise ValueError("merge output must be new")
    schema, data = load_outer_folds(prepared, planned, outer_dirs,
                                    responder_source_dir=responder_source_dir,
                                    validate_target_records=False)
    source_before = code_fingerprint()
    cfg = normalize_config(learning_config(data["config"]), schema)
    out = fresh_dir(out)
    controller, report = fit_controller_from_targets(data["packages"], schema, cfg,
        validate_packages=False, training_batch_size=controller_batch_size)
    with (out / "controller.pt").open("xb") as stream:
        torch.save({"format": "cbmjev-crossfit-controller-v1",
                    "schema_signature": schema_signature(schema), "config": cfg,
                    "state_dict": {k: v.detach().cpu() for k, v in controller.network.state_dict().items()}}, stream)
    write_json(out / "controller_report.json", report)
    write_json(out / "config.json", data["config"])
    if fit_final_head:
        head, head_report = fit_head_only(data["rows"], schema, cfg,
            response_artifact_by_group=data["response_artifact_by_group"],
            provenance_records=data["provenance"])
        save_crossfit_head(out / "head", head, schema, head_report)
    _, checked = load_outer_folds(prepared, planned,
                                  [item["directory"] for item in data["sources"]],
                                  responder_source_dir=responder_source_dir,
                                  validate_target_records=False)
    if (checked["sources"] != data["sources"]
            or checked["plan_binding"] != data["plan_binding"]
            or checked["response_source_bindings"] != data["response_source_bindings"]
            or code_fingerprint() != source_before):
        raise ValueError("merge sources changed during training")
    receipt = {"format": "cbmjev-crossfit-merge-v1", "status": "COMPLETE",
               "scope": "OOF_CONTROLLER_AND_HEAD_NOT_FINAL_DEPLOYMENT",
               "evidence_status": "OFFLINE_TRAINING_NOT_PAPER_OR_LATENCY_EVIDENCE",
               "schema_signature": schema_signature(schema),
               "controller_artifact_id": report["controller_artifact_id"],
               "final_oof_head_fitted": fit_final_head,
               "sources": data["sources"], "plan_binding": data["plan_binding"],
               "outer_training_identity": data["outer_training_identity"],
               "controller_training_batch_size": report["controller_training_batch_size"],
               "merge_source_code_hash": source_before, "files_sha256": _files(out)}
    if data["response_source_bindings"] is not None:
        receipt["response_source_bindings"] = data["response_source_bindings"]
    receipt["receipt_hash"] = stable_hash(receipt)
    write_json(out / "receipt.json", receipt)
    load_merged_controller(out, schema)
    return receipt

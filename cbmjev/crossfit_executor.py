"""Execute one nested OOF outer fold, not a complete deployment experiment."""
import inspect
from pathlib import Path

from .config import learning_config, resolve_config
from .contracts import stable_hash
from .crossfit import verify_crossfit_prepared
from .crossfit_artifacts import load_crossfit_head, save_crossfit_head
from .crossfit_cache import cache_crossfit_responses, load_crossfit_cache
from .crossfit_targets import open_target_package
from .crossfit_training import (construct_action_targets, fit_head_only,
                                _validate_target_package)
from .io import file_hash, read_json, write_json
from .learning import normalize_config
from .pipeline import (code_fingerprint, load_prepared, train_crossfit_responder,
                       train_responder)


def execute_outer_fold(prepared, planned, out, *, outer_fold, config,
                       responder_options=None, raw_root=None):
    """Train outer/inner responders, an inner-OOF head, and outer action targets.

    No resume or overwrite is supported. A failed run retains diagnostic partial
    artifacts but has no top-level receipt. Seed is shared unchanged across all
    responder fits; task-head/target seed is the resolved task config seed.
    """
    prepared, planned, out = map(Path, (prepared, planned, out))
    verified = verify_crossfit_prepared(prepared, planned)
    plan = read_json(planned / "plan.json")
    if type(outer_fold) is not int or not 0 <= outer_fold < plan["outer_folds"]:
        raise ValueError("outer_fold must identify a planned outer fold")
    if out.exists():
        raise ValueError("outer-fold output must be new; no resume or overwrite")
    schema, _, _ = load_prepared(prepared)
    resolved = resolve_config(config)
    cfg = normalize_config(learning_config(resolved), schema)
    options = dict(responder_options or {})
    if set(options) & {"prepared", "out", "raw_root"}:
        raise ValueError("responder_options cannot override input/output paths")
    options.setdefault("seed", resolved["seed"])
    options.setdefault("device", resolved["device"])
    # Bind defaults explicitly so the receipt does not silently omit actual knobs.
    try:
        bound = inspect.signature(train_responder).bind(prepared, out, **options)
    except TypeError as exc:
        raise ValueError("invalid responder_options: " + str(exc)) from exc
    bound.apply_defaults()
    options = {k: v for k, v in bound.arguments.items()
               if k not in {"prepared", "out", "raw_root"}}
    stable_hash(options)  # Require finite JSON configuration before creating output.

    def sources():
        current = verify_crossfit_prepared(prepared, planned)
        return {"plan_hash": current["plan_hash"],
                "plan_sha256": file_hash(planned / "plan.json"),
                "fold_manifest_sha256": file_hash(planned / "fold_manifest.jsonl"),
                "prepared_files_sha256": read_json(planned / "plan.json")["receipt"]["prepared_files_sha256"],
                "source_code_hash": code_fingerprint()}

    before = sources()
    if before["plan_hash"] != verified["plan_hash"]:
        raise ValueError("plan changed during validation")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", resolved)
    write_json(out / "responder_options.json", options)

    def produce(name, inner=None):
        responder, cache = out / name / "responder", out / name / "cache"
        train_crossfit_responder(prepared, planned, responder, outer_fold=outer_fold,
                                 inner_fold=inner, raw_root=raw_root, **options)
        cache_crossfit_responses(prepared, planned, responder, cache,
                                raw_root=raw_root, device=options["device"])
        return load_crossfit_cache(cache, prepared, planned, responder)[1:]

    outer_rows, outer_manifest = produce("outer")
    rows, assignments, provenance = [], {}, []
    for inner in range(plan["inner_folds"]):
        inner_rows, manifest = produce("inner_%03d" % inner, inner)
        if set(assignments) & set(manifest["response_artifact_by_group"]):
            raise ValueError("inner caches overlap in target groups")
        rows.extend(inner_rows)
        assignments.update(manifest["response_artifact_by_group"])
        provenance.extend(manifest["provenance"])
    rows.sort(key=lambda row: row["sample_id"])
    head, report = fit_head_only(rows, schema, cfg,
                                response_artifact_by_group=assignments,
                                provenance_records=provenance)
    save_crossfit_head(out / "head", head, schema, report)
    # Use the persisted component, not an unverified in-memory shortcut.
    head, report = load_crossfit_head(out / "head", schema, device=resolved["device"])
    targets = construct_action_targets(
        outer_rows, head, schema, cfg, head_report=report,
        response_artifact_by_group=outer_manifest["response_artifact_by_group"],
        provenance_records=outer_manifest["provenance"],
        manifest_path=out / "action_targets.json")
    _validate_target_package(open_target_package(out / "action_targets.json"), schema, cfg)
    if sources() != before:
        raise ValueError("sources changed during outer-fold execution")
    receipt = {"format": "cbmjev-outer-fold-execution-v1", "status": "COMPLETE",
               "scope": "SINGLE_OUTER_FOLD_NOT_FINAL_DEPLOYMENT",
               "evidence_status": "OFFLINE_TRAINING_NOT_LATENCY_OR_PAPER_EVIDENCE",
               "outer_fold": outer_fold, "schema_hash": schema.hash, **before,
               "config_hash": stable_hash(resolved),
               "responder_options_hash": stable_hash(options),
               "responder_seed_policy": "SAME_EXPLICIT_SEED_FOR_ALL_FITS",
               "responder_seed": options["seed"], "task_seed": resolved["seed"],
               "head_artifact_id": report["head_artifact_id"],
               "target_artifact_id": targets["target_artifact_id"],
               "num_target_records": targets["record_count"],
               "files_sha256": {str(path.relative_to(out)): file_hash(path)
                                for path in sorted(out.rglob("*")) if path.is_file()}}
    receipt["receipt_hash"] = stable_hash(receipt)
    write_json(out / "receipt.json", receipt)  # Sole execution completion marker.
    return receipt

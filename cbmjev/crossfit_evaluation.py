"""Validation-only offline evaluation of the actual nested-OOF system.

No conversion to the legacy split-fit bundle, no test/calibration access, and
no latency or paper-evidence claim. Declared lineage cannot certify unknown
external pretraining membership or authenticate a maliciously rewritten history.
"""
from pathlib import Path
import sys

from .config import learning_config, resolve_config
from .contracts import DeclaredCost, stable_hash
from .crossfit_artifacts import load_crossfit_head
from .crossfit_cache import crossfit_cache_source_binding, load_crossfit_cache
from .crossfit_merge import _files, load_merged_controller, load_outer_folds
from .crossfit_static import load_crossfit_static_order
from .crossfit_training import _merge_provenance, _rows_hash
from .evaluation import summarize_traces
from .io import file_hash, fresh_dir, read_json, write_json, write_jsonl
from .learning import normalize_config
from .pipeline import code_fingerprint, load_prepared
from .provenance import validate_target_exclusion
from .runtime import ReplayEnvironment, run_episode


def _log_progress(message):
    print(message, file=sys.stderr, flush=True)


def _verify_binding_files_unchanged(binding, merged_dir, responder_dir, cache_dir,
                                    static_order_dir=None, *, planned_dir=None):
    """Lightweight end-of-run drift check for immutable source artifacts.

    Full semantic source loading is intentionally done once before inference.
    Repeating it after replay doubles evaluation wall time and creates heavy
    I/O contention when policies are evaluated in parallel.  The replay itself
    only depends on the already-loaded rows/models plus the immutable artifacts
    whose content hashes are captured in ``binding``.
    """
    checks = {
        "merged_receipt_sha256": Path(merged_dir) / "receipt.json",
        "responder_receipt_sha256": Path(responder_dir) / "receipt.json",
        "cache_manifest_sha256": Path(cache_dir) / "manifest.json",
    }
    for key, path in checks.items():
        if binding.get(key) != file_hash(path):
            raise ValueError("evaluation source changed during inference: " + str(path))
    # Receipt and plan hashes are cheap to recheck. The complete outer file
    # trees were validated during _load_sources; large tensor files are not
    # rehashed here solely for this end-of-replay drift guard.
    for source in binding.get("outer_sources", ()):
        path = Path(source["directory"]) / "receipt.json"
        if file_hash(path) != source["receipt_sha256"]:
            raise ValueError("outer receipt changed during inference: " + str(path))
    if planned_dir is not None:
        plan = binding["plan_binding"]
        for name, key in (("plan.json", "plan_sha256"),
                          ("fold_manifest.jsonl", "fold_manifest_sha256")):
            path = Path(planned_dir) / name
            if file_hash(path) != plan[key]:
                raise ValueError("outer plan changed during inference: " + str(path))
    cache_manifest = read_json(Path(cache_dir) / "manifest.json")
    expected_cache_files = cache_manifest.get("files_sha256")
    actual_cache_files = {name: file_hash(Path(cache_dir) / name) for name in
                          ("schema.json", "responses.jsonl", "exclusions.jsonl")}
    if expected_cache_files != actual_cache_files:
        raise ValueError("cache file hash mismatch")
    if static_order_dir is not None and "static_order" in binding:
        static_binding = binding["static_order"]
        static_checks = {
            "receipt_sha256": Path(static_order_dir) / "receipt.json",
            "report_sha256": Path(static_order_dir) / "static_order.json",
        }
        for key, path in static_checks.items():
            if static_binding.get(key) != file_hash(path):
                raise ValueError("static report hash changed during inference: " + str(path))


def _load_sources(prepared, planned, merged_dir, responder_dir, cache_dir, device,
                  static_order_dir=None, responder_source_dir=None):
    schema, _, _ = load_prepared(prepared)
    controller, controller_report = load_merged_controller(merged_dir, schema, device=device)
    receipt = read_json(merged_dir / "receipt.json")
    if receipt.get("final_oof_head_fitted") is not True:
        raise ValueError("evaluation requires a final OOF head")
    _, data = load_outer_folds(prepared, planned,
                               [s["directory"] for s in receipt["sources"]],
                               responder_source_dir=responder_source_dir,
                               validate_target_records=False)
    for key in ("sources", "plan_binding", "outer_training_identity"):
        if receipt.get(key) != data[key]:
            raise ValueError("merged source binding mismatch: " + key)
    config = read_json(merged_dir / "config.json")
    if config != data["config"]:
        raise ValueError("merged configuration differs from outer sources")
    cfg = normalize_config(learning_config(config), schema)
    head, head_report = load_crossfit_head(merged_dir / "head", schema, device=device)
    if head_report["config"] != cfg or controller_report["config"] != cfg:
        raise ValueError("component configuration mismatch")
    packages = sorted(data["packages"], key=lambda p: p["target_artifact_id"])
    samples = sorted(r["sample_id"] for r in data["rows"])
    groups = sorted({r["group_id"] for r in data["rows"]})
    expected_controller = {
        "target_artifact_ids": [p["target_artifact_id"] for p in packages],
        "target_package_sha256": [p["package_sha256"] for p in packages],
        "target_sample_ids": samples, "target_group_ids": groups,
    }
    expected_head = {"fit_sample_ids": samples, "fit_group_ids": groups,
                     "fit_rows_sha256": _rows_hash(data["rows"]),
                     "response_artifact_by_group": data["response_artifact_by_group"]}
    for report, expected in ((controller_report, expected_controller), (head_report, expected_head)):
        if any(report.get(k) != v for k, v in expected.items()):
            raise ValueError("component report differs from complete outer sources")
    # Source ancestry must be present verbatim, not only target/package IDs.
    for report, source_records in (
            (controller_report, _merge_provenance(*(p["provenance"] for p in packages))),
            (head_report, data["provenance"])):
        if len(_merge_provenance(report["provenance"], source_records)) != len(report["provenance"]):
            raise ValueError("component omits source ancestry")
    cache_schema, rows, manifest = load_crossfit_cache(
        cache_dir, prepared, planned, responder_dir, responder_source_dir=responder_source_dir)
    if (cache_schema != schema or manifest["crossfit"]["stage"] != "final"
            or manifest["split"] != "validation" or not rows):
        raise ValueError("evaluation requires nonempty final validation cache")
    responder_receipt = read_json(responder_dir / "receipt.json")
    # Receipts carry effective defaults (e.g. backend-selected learning rate),
    # whereas the invocation may intentionally specify learning_rate=None.
    for source in data["sources"]:
        outer_receipt = read_json(Path(source["directory"]) / "outer/responder/receipt.json")
        if responder_receipt.get("training_options") != outer_receipt.get("training_options"):
            raise ValueError("final responder configuration differs from outer responders")
    provenance = _merge_provenance(head_report["provenance"], controller_report["provenance"],
                                   manifest["provenance"])
    artifact_ids = [
        head_report["head_artifact_id"], controller_report["controller_artifact_id"],
        manifest["responder_artifact_id"]]
    static_binding = None
    if static_order_dir is not None:
        order, static_report = load_crossfit_static_order(static_order_dir, prepared, planned,
            [s["directory"] for s in data["sources"]], device=device,
            responder_source_dir=responder_source_dir)
        provenance = _merge_provenance(provenance, static_report["provenance"])
        artifact_ids.append(static_report["static_order_artifact_id"])
        static_binding = {"receipt_sha256": file_hash(Path(static_order_dir) / "receipt.json"),
            "report_sha256": file_hash(Path(static_order_dir) / "static_order.json"),
            "artifact_id": static_report["static_order_artifact_id"], "order": list(order),
            "ranking_source_code_hash": static_report["source_code_hash"],
            "loss_definition": static_report["loss_definition"],
            "cost_assumptions": static_report["cost_assumptions"]}
    exclusion = validate_target_exclusion(provenance, artifact_ids=artifact_ids,
                                          target_group_ids=manifest["target_group_ids"])
    binding = {"merged_receipt_sha256": file_hash(merged_dir / "receipt.json"),
               "responder_receipt_sha256": file_hash(responder_dir / "receipt.json"),
               "cache_manifest_sha256": file_hash(cache_dir / "manifest.json"),
               "outer_sources": data["sources"], "plan_binding": data["plan_binding"],
               "head_component_sha256": head_report["head_component_sha256"],
               "controller_component_sha256": controller_report["controller_component_sha256"],
               "responder_checkpoint_sha256": manifest["responder_checkpoint_sha256"],
               "ancestry_exclusion": exclusion}
    if responder_source_dir is not None:
        binding["response_source_bindings"] = {"outer": data["response_source_bindings"],
            "final_validation_cache": crossfit_cache_source_binding(
                cache_dir, responder_dir, responder_source_dir)}
    if static_binding is not None:
        binding["static_order"] = static_binding
    return schema, rows, config, head, controller, binding


def evaluate_crossfit_validation(prepared, planned, merged_dir, responder_dir, cache_dir, out,
                                *, methods=None, cost_weight=None, max_groups=None, device="cpu",
                                static_order_dir=None, responder_source_dir=None):
    """Evaluate supported policies on a bound final-responder validation cache.

    ``all`` always means all groups in one call; requesting it together with a
    smaller global budget is rejected, not silently turned into a partial query.
    Static policies require a separately validated training-only OOF order.
    """
    prepared, planned, merged_dir, responder_dir, cache_dir, out = map(
        Path, (prepared, planned, merged_dir, responder_dir, cache_dir, out))
    if out.exists():
        raise ValueError("evaluation output must be new")
    methods = ["stop", "all", "fixed", "random", "value", "value_singleton"] if methods is None else list(methods)
    if (not methods or len(set(methods)) != len(methods)
            or set(methods) - {"stop", "all", "fixed", "random", "value", "value_singleton", "static", "static_value"}):
        raise ValueError("unsupported, empty or duplicate evaluation methods")
    if set(methods) & {"static", "static_value"} and static_order_dir is None:
        raise ValueError("static policies require static_order_dir")
    source_before = code_fingerprint()
    args = (prepared, planned, merged_dir, responder_dir, cache_dir, device,
            static_order_dir, responder_source_dir)
    _log_progress(f"[evaluate-crossfit] loading sources for methods={methods} out={out}")
    schema, rows, config, head, controller, binding = _load_sources(*args)
    _log_progress(f"[evaluate-crossfit] loaded {len(rows)} validation rows for methods={methods}")
    config["device"] = device
    if cost_weight is not None:
        config["policy"]["cost_weight"] = cost_weight
    if max_groups is not None:
        config["policy"]["max_groups"] = max_groups
    config = resolve_config(config)
    budget = config["policy"]["max_groups"]
    if budget is not None and budget > schema.num_groups:
        raise ValueError("max_groups exceeds schema")
    if set(methods) & {"value", "value_singleton", "static_value"} and controller.objective != "value":
        raise ValueError("value policies require value objective")
    cost = DeclaredCost(**config["cost"])
    if "all" in methods and (budget not in (None, schema.num_groups)
            or not config["learning"]["include_all"]
            or (config["policy"]["max_cost"] is not None and
                cost(schema.empty_state(), tuple(range(schema.num_groups))) > config["policy"]["max_cost"] + 1e-10)):
        raise ValueError("all requires feasible full budget and include_all")
    out = fresh_dir(out)
    settings, traces, reports = {}, [], {}
    for method in methods:
        _log_progress(f"[evaluate-crossfit] replay start method={method} rows={len(rows)}")
        spec = {"method": method, "config": config, "mode": "offline_replay",
                "split": "validation", "source_binding": binding}
        spec["system_hash"] = stable_hash(spec)
        settings[method] = spec
        current = []
        for row in rows:
            trace = run_episode(ReplayEnvironment(row["z"], schema), schema, head,
                method=method, controller=controller, cost=cost, **config["policy"],
                include_pairs=config["learning"]["include_pairs"],
                include_all=config["learning"]["include_all"], pairs=controller.pairs,
                order=binding["static_order"]["order"] if method in ("static", "static_value") else None,
                seed=config["seed"], device=device)
            # Labels and identities never enter the inference interface.
            trace.update(sample_id=row["sample_id"], group_id=row["group_id"], y=row["y"],
                         split="validation", policy_id=method, system_hash=spec["system_hash"],
                         seed=config["seed"], num_query_groups=schema.num_groups,
                         num_atoms=schema.num_atoms)
            current.append(trace)
        traces.extend(current)
        reports[method] = summarize_traces(current, num_classes=schema.num_classes)
        _log_progress(f"[evaluate-crossfit] replay done method={method} metrics={reports[method]}")
    evidence = "OFFLINE_VALIDATION_DEVELOPMENT_NOT_PAPER_OR_LATENCY_EVIDENCE"
    write_jsonl(out / "traces.jsonl", traces)
    write_json(out / "settings.json", settings)
    write_json(out / "metrics.json", {"split": "validation", "mode": "offline_replay",
        "seed": config["seed"], "policies": reports, "paper_evidence": False,
        "evidence_status": evidence, "source_binding": binding, "source_code_hash": source_before})
    _verify_binding_files_unchanged(binding, merged_dir, responder_dir, cache_dir,
                                    static_order_dir, planned_dir=planned)
    if code_fingerprint() != source_before:
        raise ValueError("evaluation sources changed during inference")
    receipt = {"format": "cbmjev-crossfit-validation-v1", "status": "COMPLETE",
               "evidence_status": evidence, "paper_evidence": False,
               "source_binding": binding, "source_code_hash": source_before,
               "files_sha256": _files(out), "samples_per_policy": len(rows),
               "num_policies": len(methods)}
    receipt["receipt_hash"] = stable_hash(receipt)
    write_json(out / "receipt.json", receipt)
    return receipt

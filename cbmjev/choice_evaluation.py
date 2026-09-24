"""Paired structured Choice validation using the existing nested-OOF loader.

No legacy-bundle conversion, test access, source-check bypass, or timing claim.
"""
from pathlib import Path

from .choice_artifacts import load_choice_pair, verify_choice_binding_unchanged
from .choice_runtime import StructuredChoiceController
from .contracts import DeclaredCost, stable_hash
from .crossfit_evaluation import _load_sources, _verify_binding_files_unchanged
from .crossfit_merge import _files
from .evaluation import summarize_traces
from .io import file_hash, fresh_dir, read_json, write_json, write_jsonl
from .pipeline import code_fingerprint
from .runtime import ReplayEnvironment, run_episode


def _bound_sources(prepared, planned, merged, responder, cache, choice, targets, target_config,
                   device, responder_source_dir=None):
    schema, rows, config, head, _, base = _load_sources(
        prepared, planned, merged, responder, cache, device,
        responder_source_dir=responder_source_dir)
    *choice_heads, settings, report = load_choice_pair(choice, prepared=prepared,
        targets=targets, config=target_config, device=device, split="validation",
        reuse_sealed_training_validation=True)
    # _load_sources already validated the complete outer folds against the
    # merged controller report. Reuse that verified report instead of loading
    # all outer sources a second time solely to recover these two lists.
    controller_report = read_json(merged / "controller_report.json")
    merged_receipt = read_json(merged / "receipt.json")
    if (merged_receipt["files_sha256"].get("controller_report.json")
            != file_hash(merged / "controller_report.json")):
        raise ValueError("merged controller report changed during Choice source loading")
    expected = sorted(zip(controller_report["target_artifact_ids"],
                          controller_report["target_package_sha256"]))
    actual = sorted((p["target_artifact_id"], p["package_sha256"]) for p in report["package_bindings"])
    if actual != expected:
        raise ValueError("Choice training packages differ from complete merged outer sources")
    # The original loader validates exclusion for the same packages, head and
    # final responder. Choice adds no training examples or new external source.
    binding = {"nested_system": base, "choice_receipt_sha256": file_hash(choice / "receipt.json"),
        "choice_files_sha256": read_json(choice / "receipt.json")["files"],
        "choice_training_packages": report["package_bindings"],
        "same_complete_outer_training_packages_verified": True}
    return schema, rows, config, head, choice_heads, settings, report, binding


def _verify_outer_plan_unchanged(binding, planned):
    """Check bytes of the already-validated outer/plan snapshot, not its semantics."""
    plan = binding["plan_binding"]
    for name, key in (("plan.json", "plan_sha256"),
                      ("fold_manifest.jsonl", "fold_manifest_sha256")):
        if file_hash(planned / name) != plan[key]:
            raise ValueError("Choice outer plan changed during inference: " + name)
    for source in binding["outer_sources"]:
        directory = Path(source["directory"])
        if file_hash(directory / "receipt.json") != source["receipt_sha256"]:
            raise ValueError("Choice outer receipt changed during inference: " + str(directory))
        if read_json(directory / "receipt.json")["files_sha256"] != _files(directory):
            raise ValueError("Choice outer files changed during inference: " + str(directory))


def evaluate_choice_crossfit_validation(prepared, planned, merged_dir, responder_dir, cache_dir,
        choice_dir, targets, target_config, out, *, cost_weight=None, max_groups=None,
        device="cpu", split="validation", responder_source_dir=None):
    if split != "validation":
        raise ValueError("Choice crossfit evaluation permits validation only, not test/calibration")
    prepared, planned, merged, responder, cache, choice, target_config, out = map(Path,
        (prepared, planned, merged_dir, responder_dir, cache_dir, choice_dir, target_config, out))
    targets = tuple(Path(path) for path in targets)
    if out.exists():
        raise ValueError("Choice evaluation output must be new")
    source = code_fingerprint()
    args = (prepared, planned, merged, responder, cache, choice, targets, target_config,
            device, responder_source_dir)
    schema, rows, config, head, choice_heads, settings, training, binding = _bound_sources(*args)
    weight = settings["cost_weight"] if cost_weight is None else cost_weight
    if type(weight) not in (int, float) or weight != settings["cost_weight"]:
        raise ValueError("evaluation cost weight must match Choice training lambda")
    budget_mode = settings.get("budget_mode", "fixed")
    budget = (schema.num_groups if budget_mode == "uniform_remaining" else settings["remaining_groups"]) if max_groups is None else max_groups
    if type(budget) is not int or not 0 <= budget <= schema.num_groups:
        raise ValueError("max_groups outside schema bounds")
    if budget_mode == "fixed" and budget != settings["remaining_groups"]:
        raise ValueError("fixed-budget evaluation must match initial training budget")
    if budget_mode not in ("fixed", "uniform_remaining"):
        raise ValueError("unknown Choice training budget mode")
    counts = training.get("selected_budget_counts",
        {str(settings["remaining_groups"]): training["selected_occurrences"]})
    covered = {int(key) for key, count in counts.items() if count > 0}
    if budget_mode == "uniform_remaining" and budget not in covered:
        raise ValueError("initial evaluation budget absent from selected training occurrences")
    cost = DeclaredCost(**config["cost"])
    if (cost.setup, cost.call, cost.per_group) != (0., 0., 1.):
        raise ValueError("Choice teacher requires zero setup/call and unit singleton cost")
    if config["policy"]["max_cost"] is not None and config["policy"]["max_cost"] < budget:
        raise ValueError("additional cost cap removes teacher singleton candidates")
    out = fresh_dir(out)
    all_traces, metrics, specs, coverage = [], {}, {}, {}
    for head_name, choice_head in zip(settings.get("head_names", ("scalar", "attention")), choice_heads):
        name = "structured_choice_" + head_name
        controller = StructuredChoiceController(schema, choice_head)
        spec = {"method": name, "runtime_method": "structured_choice", "split": "validation",
            "mode": "offline_replay", "max_groups": budget, "cost_weight": weight,
            "cost": cost.to_dict(), "source_binding": binding,
            "head_weights_sha256": training["fit"]["heads"][head_name]["final_weights_sha256"]}
        spec["system_hash"] = stable_hash(spec)
        specs[name], current = spec, []
        seen, unseen, forced = {}, {}, {}
        for row in rows:
            trace = run_episode(ReplayEnvironment(row["z"], schema), schema, head,
                method="structured_choice", controller=controller, cost=cost, cost_weight=weight,
                max_groups=budget, max_cost=config["policy"]["max_cost"], include_pairs=False,
                include_all=False, seed=config["seed"], device=device)
            for step in trace["steps"]:
                remaining = budget - sum(schema.group_mask(step["before"]))
                bucket = forced if step["legal_candidate_count"] == 1 else seen if remaining in covered else unseen
                bucket[str(remaining)] = bucket.get(str(remaining), 0) + 1
            # Identities and target labels enter only the post-inference metric layer.
            trace.update(sample_id=row["sample_id"], group_id=row["group_id"], y=row["y"],
                split="validation", policy_id=name, system_hash=spec["system_hash"], seed=config["seed"],
                num_query_groups=schema.num_groups, num_atoms=schema.num_atoms)
            current.append(trace)
        all_traces.extend(current)
        metrics[name] = summarize_traces(current, num_classes=schema.num_classes)
        coverage[name] = {"covered_nontrivial_decisions_by_budget": seen,
                         "uncovered_nontrivial_decisions_by_budget": unseen,
                         "forced_single_candidate_decisions_by_budget": forced}
    status = "OFFLINE_VALIDATION_DEVELOPMENT_NOT_PAPER_OR_LATENCY_EVIDENCE"
    coverage_report = {"training_budget_mode": budget_mode, "training_counts": counts,
        "policies": coverage, "scope": "Marginal numeric-budget coverage only, not joint state/budget or distribution-shift validation",
        "fixed_budget_limitation": "Later decreasing budgets can be outside fixed training support; forced STOP-only states listed separately",
        "cost_penalty_applied_again_at_inference": False}
    write_json(out / "settings.json", specs)
    write_jsonl(out / "traces.jsonl", all_traces)
    write_json(out / "metrics.json", {"split": "validation", "mode": "offline_replay",
        "seed": config["seed"], "policies": metrics, "coverage": coverage_report,
        "source_binding": binding, "source_code_hash": source,
        "paper_evidence": False, "evidence_status": status})
    _verify_binding_files_unchanged(binding["nested_system"], merged, responder, cache)
    _verify_outer_plan_unchanged(binding["nested_system"], planned)
    if read_json(merged / "receipt.json")["files_sha256"] != _files(merged):
        raise ValueError("merged model files changed during Choice inference")
    verify_choice_binding_unchanged(choice, prepared=prepared, targets=targets,
                                    config=target_config)
    if (file_hash(choice / "receipt.json") != binding["choice_receipt_sha256"]
            or code_fingerprint() != source):
        raise ValueError("Choice evaluation sources changed during inference")
    receipt = {"format": "cbmjev-choice-crossfit-validation-v1", "status": "COMPLETE",
        "source_binding": binding, "source_code_hash": source, "files_sha256": _files(out),
        "samples_per_policy": len(rows), "num_policies": len(choice_heads), "coverage": coverage_report,
        "paper_evidence": False, "evidence_status": status}
    receipt["receipt_hash"] = stable_hash(receipt)
    write_json(out / "receipt.json", receipt)
    return receipt

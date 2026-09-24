"""Validation-only K16 replay for the objective-matched Choice risk control."""

import argparse
from pathlib import Path

from cbmjev.choice_evaluation import _bound_sources
from cbmjev.choice_runtime import StructuredChoiceController
from cbmjev.contracts import DeclaredCost, stable_hash
from cbmjev.crossfit_merge import _files
from cbmjev.evaluation import summarize_traces
from cbmjev.io import file_hash, fresh_dir, write_json, write_jsonl
from cbmjev.pipeline import code_fingerprint
from cbmjev.runtime import ReplayEnvironment, run_episode
from scripts.train_choice_risk_pair import load_choice_risk_pair


def evaluate_choice_risk_pair(prepared, planned, merged, responder, cache,
        soft_choice, risk_choice, targets, target_config, out, *, max_groups=16,
        device="cpu", responder_source_dir=None):
    prepared, planned, merged, responder, cache, soft_choice, risk_choice, target_config, out = map(
        Path, (prepared, planned, merged, responder, cache, soft_choice, risk_choice,
               target_config, out))
    targets = tuple(map(Path, targets))
    if out.exists():
        raise ValueError("risk-control evaluation output must be new")
    source = {"cbmjev": code_fingerprint(), "evaluation_script": file_hash(Path(__file__))}
    args = (prepared, planned, merged, responder, cache, soft_choice, targets,
            target_config, device, responder_source_dir)
    schema, rows, config, head, _, soft_settings, _, soft_binding = _bound_sources(*args)
    scalar, attention, settings, training = load_choice_risk_pair(risk_choice,
        prepared=prepared, targets=targets, config=target_config,
        soft_choice=soft_choice, device=device)
    if (settings["schema_hash"] != soft_settings["schema_hash"]
            or settings["cost_weight"] != soft_settings["cost_weight"]):
        raise ValueError("risk/soft Choice settings differ")
    if type(max_groups) is not int or not 0 <= max_groups <= schema.num_groups:
        raise ValueError("max_groups outside schema bounds")
    budget_mode = settings["budget_mode"]
    if budget_mode == "fixed" and max_groups != settings["remaining_groups"]:
        raise ValueError("fixed-budget evaluation must match training budget")
    counts = training["selected_budget_counts"]
    covered = {int(key) for key, count in counts.items() if count > 0}
    if budget_mode == "uniform_remaining" and max_groups not in covered:
        raise ValueError("initial evaluation budget absent from training occurrences")
    cost = DeclaredCost(**config["cost"])
    if (cost.setup, cost.call, cost.per_group) != (0., 0., 1.):
        raise ValueError("Choice utility target requires unit singleton costs")
    if config["policy"]["max_cost"] is not None and config["policy"]["max_cost"] < max_groups:
        raise ValueError("cost cap removes teacher singleton candidates")
    binding = {"nested_system": soft_binding["nested_system"],
        "soft_choice_receipt_sha256": file_hash(soft_choice / "receipt.json"),
        "risk_choice_receipt_sha256": file_hash(risk_choice / "receipt.json"),
        "risk_choice_training_binding": training["source_binding"],
        "same_complete_outer_training_packages_verified": True}
    out = fresh_dir(out)
    all_traces, metrics, specs, coverage = [], {}, {}, {}
    for name, choice_head in (("structured_choice_risk_scalar", scalar),
                              ("structured_choice_risk_attention", attention)):
        controller = StructuredChoiceController(schema, choice_head)
        weight_key = "scalar" if name.endswith("scalar") else "attention"
        spec = {"method": name, "runtime_method": "structured_choice",
            "objective": settings["objective"], "split": "validation",
            "mode": "offline_replay", "max_groups": max_groups,
            "cost_weight": settings["cost_weight"], "cost": cost.to_dict(),
            "source_binding": binding,
            "head_weights_sha256": training["fit"]["heads"][weight_key]["final_weights_sha256"]}
        spec["system_hash"] = stable_hash(spec)
        specs[name], current = spec, []
        seen, unseen, forced = {}, {}, {}
        for row in rows:
            trace = run_episode(ReplayEnvironment(row["z"], schema), schema, head,
                method="structured_choice", controller=controller, cost=cost,
                cost_weight=settings["cost_weight"], max_groups=max_groups,
                max_cost=config["policy"]["max_cost"], include_pairs=False,
                include_all=False, seed=config["seed"], device=device)
            for step in trace["steps"]:
                remaining = max_groups - sum(schema.group_mask(step["before"]))
                bucket = forced if step["legal_candidate_count"] == 1 else seen if remaining in covered else unseen
                bucket[str(remaining)] = bucket.get(str(remaining), 0) + 1
            trace.update(sample_id=row["sample_id"], group_id=row["group_id"],
                y=row["y"], split="validation", policy_id=name,
                system_hash=spec["system_hash"], seed=config["seed"],
                num_query_groups=schema.num_groups, num_atoms=schema.num_atoms)
            current.append(trace)
        all_traces.extend(current)
        metrics[name] = summarize_traces(current, num_classes=schema.num_classes)
        coverage[name] = {"covered_nontrivial_decisions_by_budget": seen,
                          "uncovered_nontrivial_decisions_by_budget": unseen,
                          "forced_single_candidate_decisions_by_budget": forced}
    coverage_report = {"training_budget_mode": budget_mode, "training_counts": counts,
        "policies": coverage,
        "scope": "Marginal numeric-budget coverage only, not joint state/budget coverage",
        "cost_penalty_applied_again_at_inference": False}
    status = "OFFLINE_VALIDATION_DEVELOPMENT_NOT_PAPER_OR_LATENCY_EVIDENCE"
    write_json(out / "settings.json", specs)
    write_jsonl(out / "traces.jsonl", all_traces)
    write_json(out / "metrics.json", {"split": "validation", "mode": "offline_replay",
        "seed": config["seed"], "policies": metrics, "coverage": coverage_report,
        "source_binding": binding, "source_code_hash": source,
        "paper_evidence": False, "evidence_status": status})
    if (_bound_sources(*args)[-1] != soft_binding
            or file_hash(soft_choice / "receipt.json") != binding["soft_choice_receipt_sha256"]
            or file_hash(risk_choice / "receipt.json") != binding["risk_choice_receipt_sha256"]
            or {"cbmjev": code_fingerprint(), "evaluation_script": file_hash(Path(__file__))} != source):
        raise ValueError("risk-control evaluation source changed during inference")
    receipt = {"format": "cbmjev-choice-risk-crossfit-validation-v1",
        "status": "COMPLETE", "source_binding": binding, "source_code_hash": source,
        "files_sha256": _files(out), "samples_per_policy": len(rows),
        "num_policies": 2, "coverage": coverage_report, "paper_evidence": False,
        "evidence_status": status}
    receipt["receipt_hash"] = stable_hash(receipt)
    write_json(out / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "planned", "merged", "responder", "cache",
                 "soft-choice", "risk-choice", "target-config", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--max-groups", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--responder-source-dir")
    args = parser.parse_args()
    receipt = evaluate_choice_risk_pair(args.prepared, args.planned,
        args.merged, args.responder, args.cache, args.soft_choice,
        args.risk_choice, args.targets, args.target_config, args.out,
        max_groups=args.max_groups, device=args.device,
        responder_source_dir=args.responder_source_dir)
    print(receipt["receipt_hash"])


if __name__ == "__main__":
    main()

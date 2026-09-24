#!/usr/bin/env python3
"""Minimum-acquisition stopping diagnostic for CUB/CBMJev value policies.

This script is a diagnostic wrapper around the already trained value controller.
It does not modify ``cbmjev/*.py`` and therefore does not invalidate concurrent
core-package fingerprinted runs.

For each ``max_budget`` and ``min_groups`` pair, value acquisition is forced
until at least ``min_groups`` concept groups have been observed. The default
singleton family preserves the original diagnostic. The configured family
retains the original value policy's legal pair/all candidates, enabling a
same-controller stopping intervention on that policy. After the floor, the
normal value-vs-STOP objective applies.
"""
import argparse
import math
import time
from pathlib import Path

from cbmjev.config import resolve_config
from cbmjev.contracts import DeclaredCost, candidate_actions, stable_hash
from cbmjev.crossfit_evaluation import _load_sources, _verify_binding_files_unchanged
from cbmjev.crossfit_merge import _files
from cbmjev.evaluation import summarize_traces
from cbmjev.io import file_hash, fresh_dir, write_json, write_jsonl
from cbmjev.pipeline import code_fingerprint
from cbmjev.runtime import ReplayEnvironment, run_episode, synchronize


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--planned", required=True)
    parser.add_argument("--merged", required=True)
    parser.add_argument("--responder", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-budgets", type=int, nargs="+", required=True)
    parser.add_argument("--min-groups", type=int, nargs="+", required=True)
    parser.add_argument("--action-family", choices=("singleton", "configured"),
                        default="singleton")
    parser.add_argument("--matched-fixed-count", action="store_true",
                        help="also replay canonical fixed order at each adaptive trace's acquired count")
    parser.add_argument("--cost-weight", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--responder-source-dir")
    return parser.parse_args()


def choose_value_minstop(observed, actions, *, schema, controller, cost,
                         cost_weight, min_groups, action_family="singleton"):
    if getattr(controller, "objective", None) != "value":
        raise ValueError("min-stop diagnostics require a value-objective controller")
    if action_family not in ("singleton", "configured"):
        raise ValueError("unknown min-stop action family")
    acquired = sum(1 for present in schema.group_mask(observed) if present)
    if action_family == "singleton":
        actions = tuple(a for a in actions if len(a) <= 1)
    if acquired < min_groups:
        actions = tuple(a for a in actions if a)
    if not actions:
        if acquired < min_groups:
            raise ValueError("min-stop floor infeasible under remaining budget or cost")
        return (), {}
    scores = tuple(float(v) for v in controller.predict(observed, actions))
    if len(scores) != len(actions) or any(not math.isfinite(v) for v in scores):
        raise ValueError("controller returned invalid action scores")
    objectives = [-v + cost_weight * cost(observed, a) for a, v in zip(actions, scores)]
    best = min(range(len(actions)), key=lambda i: (
        objectives[i], bool(actions[i]), cost(observed, actions[i]), actions[i]))
    return actions[best], {",".join(map(str, a)) if a else "STOP": v
                           for a, v in zip(actions, scores)}


def choose_value_singleton_minstop(observed, actions, *, schema, controller, cost,
                                   cost_weight, min_groups):
    """Retain the original singleton helper for existing diagnostic callers."""
    return choose_value_minstop(observed, actions, schema=schema,
        controller=controller, cost=cost, cost_weight=cost_weight,
        min_groups=min_groups, action_family="singleton")


def run_episode_minstop(env, schema, head, *, controller, cost, cost_weight,
                        max_groups, min_groups, max_cost, include_pairs,
                        include_all, pairs, device, action_family="singleton"):
    if action_family not in ("singleton", "configured"):
        raise ValueError("unknown min-stop action family")
    if not 0 <= min_groups <= max_groups <= schema.num_groups:
        raise ValueError("invalid min-stop acquisition budget")
    synchronize(device)
    started = time.perf_counter()
    declared, policy_ms, steps = 0.0, 0.0, []
    for _ in range(schema.num_groups + 1):
        observed = env.state()
        acquired = sum(schema.group_mask(observed))
        if acquired >= max_groups:
            steps.append({"before": list(observed), "action": [], "scores": {},
                          "legal_candidate_count": 1, "scored_candidate_count": 0,
                          "declared_cost": 0.0, "decision": "STOP",
                          "max_budget_stop": True})
            break
        synchronize(device)
        begin = time.perf_counter()
        actions = candidate_actions(observed, schema, include_pairs, include_all, pairs)
        remaining = math.inf if max_cost is None else max(0.0, max_cost - declared)
        actions = tuple(a for a in actions if len(a) + acquired <= max_groups and
                        cost(observed, a) <= remaining + 1e-10)
        action, scores = choose_value_minstop(
            observed, actions, schema=schema, controller=controller, cost=cost,
            cost_weight=cost_weight, min_groups=min_groups,
            action_family=action_family)
        synchronize(device)
        policy_ms += (time.perf_counter() - begin) * 1000
        step = {"before": list(observed), "action": list(action), "scores": scores,
                "legal_candidate_count": len(actions),
                "scored_candidate_count": len(scores),
                "declared_cost": cost(observed, action),
                "min_groups": min_groups, "max_groups": max_groups}
        if not action:
            step["decision"] = "STOP"
            steps.append(step)
            break
        values = env.query(action)
        declared += step["declared_cost"]
        step.update(decision="ACQUIRE", atom_ids=list(schema.expand(action)),
                    values=list(values))
        steps.append(step)
    else:
        raise RuntimeError("min-stop episode did not terminate within K+1 steps")
    synchronize(device)
    begin = time.perf_counter()
    probabilities = tuple(float(p) for p in head.probabilities(env.state()))
    if len(probabilities) != schema.num_classes or any(
            not math.isfinite(p) or p < 0 or p > 1 for p in probabilities):
        raise ValueError("task head returned invalid probabilities")
    if abs(sum(probabilities) - 1.0) > 1e-5:
        raise ValueError("task probabilities must sum to one")
    prediction = max(range(len(probabilities)), key=lambda i: probabilities[i])
    synchronize(device)
    method = "value_singleton_minstop" if action_family == "singleton" else "value_minstop"
    return {"method": method, "prediction": prediction,
            "probabilities": list(probabilities), "final_state": list(env.state()),
            "queried_groups": [i for i, present in enumerate(schema.group_mask(env.state())) if present],
            "queried_atoms": [i for i, v in enumerate(env.state()) if v != -1],
            "calls": env.calls, "declared_cost": declared, "cost_units": cost.units,
            "mode": env.mode, "steps": steps, "backend_stats": env.backend_stats,
            "policy_ms": policy_ms, "head_ms": (time.perf_counter() - begin) * 1000,
            "replay_wall_ms": (time.perf_counter() - started) * 1000,
            "evidence_status": "OFFLINE_REPLAY_NOT_LATENCY"}


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        raise ValueError("output must be new: " + str(out))
    source_before = code_fingerprint()
    wrapper_before = file_hash(__file__)
    print(f"[minstop-grid] loading sources max_budgets={args.max_budgets} "
          f"min_groups={args.min_groups}", flush=True)
    schema, rows, config, head, controller, binding = _load_sources(
        Path(args.prepared), Path(args.planned), Path(args.merged),
        Path(args.responder), Path(args.cache), args.device, None,
        args.responder_source_dir)
    if controller.objective != "value":
        raise ValueError("min-stop diagnostics require a value-objective merged controller")
    print(f"[minstop-grid] loaded rows={len(rows)} groups={schema.num_groups}", flush=True)
    config["device"] = args.device
    config = resolve_config(config)
    if args.cost_weight is not None:
        config["policy"]["cost_weight"] = args.cost_weight
    cost = DeclaredCost(**config["cost"])
    include_pairs = config["learning"]["include_pairs"]
    include_all = config["learning"]["include_all"]
    traces, reports, settings, comparisons = [], {}, {}, {}
    for max_budget in args.max_budgets:
        if not 0 <= max_budget <= schema.num_groups:
            raise ValueError("max budget outside valid group range")
        for min_groups in args.min_groups:
            if not 0 <= min_groups <= max_budget:
                continue
            method = "value_singleton_minstop" if args.action_family == "singleton" else "value_minstop"
            key = f"{method}_min{min_groups}_K{max_budget}"
            spec = {"method": method, "action_family": args.action_family,
                    "min_groups": min_groups, "max_budget": max_budget,
                    "config": config, "mode": "offline_replay",
                    "split": "validation", "source_binding": binding,
                    "diagnostic": "minimum_acquisition_stopping_floor"}
            spec["system_hash"] = stable_hash(spec)
            settings[key] = spec
            fixed_key = "fixed_at_" + key
            if args.matched_fixed_count:
                fixed_spec = {"method": "fixed_at_adaptive_count", "adaptive_policy_id": key,
                    "count_source": "adaptive_trace_queried_groups_no_labels",
                    "config": config, "mode": "offline_replay", "split": "validation",
                    "source_binding": binding,
                    "diagnostic": "matched_realized_count_selection_control"}
                fixed_spec["system_hash"] = stable_hash(fixed_spec)
                settings[fixed_key] = fixed_spec
            print(f"[minstop-grid] replay start {key}", flush=True)
            current, fixed_current = [], []
            for row_index, row in enumerate(rows, 1):
                trace = run_episode_minstop(
                    ReplayEnvironment(row["z"], schema), schema, head,
                    controller=controller, cost=cost,
                    cost_weight=config["policy"]["cost_weight"],
                    max_groups=max_budget, min_groups=min_groups,
                    max_cost=config["policy"]["max_cost"],
                    include_pairs=include_pairs, include_all=include_all,
                    pairs=controller.pairs, device=args.device,
                    action_family=args.action_family)
                trace.update(sample_id=row["sample_id"], group_id=row["group_id"],
                             y=row["y"], split="validation", policy_id=key,
                             system_hash=spec["system_hash"], seed=config["seed"],
                             num_query_groups=schema.num_groups,
                             num_atoms=schema.num_atoms,
                             budget_groups=max_budget,
                             min_groups=min_groups)
                current.append(trace)
                if args.matched_fixed_count:
                    realized_count = len(trace["queried_groups"])
                    fixed = run_episode(
                        ReplayEnvironment(row["z"], schema), schema, head,
                        method="fixed", controller=controller, cost=cost,
                        cost_weight=config["policy"]["cost_weight"],
                        max_groups=realized_count,
                        max_cost=config["policy"]["max_cost"],
                        include_pairs=include_pairs, include_all=include_all,
                        pairs=controller.pairs, seed=config["seed"],
                        device=args.device)
                    if len(fixed["queried_groups"]) != realized_count:
                        raise ValueError("matched fixed count infeasible under cost/action constraints")
                    fixed.update(sample_id=row["sample_id"], group_id=row["group_id"],
                                 y=row["y"], split="validation", policy_id=fixed_key,
                                 system_hash=fixed_spec["system_hash"], seed=config["seed"],
                                 num_query_groups=schema.num_groups,
                                 num_atoms=schema.num_atoms,
                                 budget_groups=max_budget,
                                 matched_adaptive_policy_id=key,
                                 adaptive_realized_groups=realized_count)
                    fixed_current.append(fixed)
                if row_index % 100 == 0 or row_index == len(rows):
                    print(f"[minstop-grid] replay progress {key} "
                          f"{row_index}/{len(rows)}", flush=True)
            traces.extend(current)
            reports[key] = summarize_traces(current, num_classes=schema.num_classes)
            reports[key]["budget_groups"] = max_budget
            reports[key]["min_groups"] = min_groups
            if args.matched_fixed_count:
                traces.extend(fixed_current)
                reports[fixed_key] = summarize_traces(
                    fixed_current, num_classes=schema.num_classes)
                reports[fixed_key]["budget_groups"] = max_budget
                reports[fixed_key]["matched_adaptive_policy_id"] = key
                if (reports[fixed_key]["total_queried_groups"] !=
                        reports[key]["total_queried_groups"]):
                    raise ValueError("matched fixed count aggregate differs from adaptive")
                paired = {"both_correct": 0, "adaptive_only_correct": 0,
                          "fixed_only_correct": 0, "both_wrong": 0}
                for adaptive, fixed in zip(current, fixed_current):
                    if adaptive["sample_id"] != fixed["sample_id"] or adaptive["y"] != fixed["y"]:
                        raise ValueError("matched fixed comparison lost sample alignment")
                    adaptive_ok = adaptive["prediction"] == adaptive["y"]
                    fixed_ok = fixed["prediction"] == fixed["y"]
                    if adaptive_ok and fixed_ok:
                        paired["both_correct"] += 1
                    elif adaptive_ok:
                        paired["adaptive_only_correct"] += 1
                    elif fixed_ok:
                        paired["fixed_only_correct"] += 1
                    else:
                        paired["both_wrong"] += 1
                comparisons[key] = {"adaptive_policy_id": key,
                    "fixed_policy_id": fixed_key, "num_samples": len(current),
                    "adaptive_minus_fixed_accuracy":
                        reports[key]["accuracy"] - reports[fixed_key]["accuracy"],
                    "adaptive_minus_fixed_mean_declared_cost":
                        reports[key]["mean_declared_cost"] -
                        reports[fixed_key]["mean_declared_cost"],
                    "paired_correctness": paired,
                    "selection_control_scope":
                        "same_sample_same_realized_group_count_not_necessarily_same_declared_cost"}
            print(f"[minstop-grid] replay done {key} "
                  f"acc={reports[key]['accuracy']:.6f} "
                  f"macro_f1={reports[key]['macro_f1']:.6f} "
                  f"mean_groups={reports[key]['mean_queried_groups']:.3f}",
                  flush=True)
    out = fresh_dir(out)
    write_jsonl(out / "traces.jsonl", traces)
    write_json(out / "settings.json", settings)
    write_json(out / "metrics.json", {"split": "validation",
        "mode": "offline_replay", "seed": config["seed"],
        "max_budgets": args.max_budgets, "min_groups": args.min_groups,
        "action_family": args.action_family,
        "matched_fixed_count": args.matched_fixed_count,
        "policies": reports, "matched_selection_comparisons": comparisons,
        "paper_evidence": False,
        "diagnostic": "minimum_acquisition_stopping_floor",
        "evidence_status": "OFFLINE_VALIDATION_DIAGNOSTIC_NOT_PAPER_EVIDENCE",
        "source_binding": binding, "source_code_hash": source_before,
        "wrapper_source_sha256": wrapper_before})
    _verify_binding_files_unchanged(binding, Path(args.merged), Path(args.responder),
                                    Path(args.cache), None, planned_dir=Path(args.planned))
    if code_fingerprint() != source_before or file_hash(__file__) != wrapper_before:
        raise ValueError("source code changed during min-stop evaluation")
    receipt = {"format": "cbmjev-crossfit-minstop-grid-v1",
               "status": "COMPLETE", "paper_evidence": False,
               "evidence_status": "OFFLINE_VALIDATION_DIAGNOSTIC_NOT_PAPER_EVIDENCE",
               "source_binding": binding, "source_code_hash": source_before,
               "wrapper_source_sha256": wrapper_before,
               "samples_per_policy": len(rows), "num_policies": len(reports),
               "max_budgets": args.max_budgets, "min_groups": args.min_groups,
               "action_family": args.action_family,
               "matched_fixed_count": args.matched_fixed_count,
               "diagnostic": "minimum_acquisition_stopping_floor",
               "files_sha256": _files(out)}
    receipt["receipt_hash"] = stable_hash(receipt)
    write_json(out / "receipt.json", receipt)
    print(f"[minstop-grid] complete out={out} policies={len(reports)}", flush=True)


if __name__ == "__main__":
    main()

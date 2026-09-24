#!/usr/bin/env python3
"""Forced-budget diagnostic for CUB/CBMJev value policies.

This script intentionally lives outside ``cbmjev.runtime`` so that long-running
audited evaluations that fingerprint the core package are not invalidated.
It reuses the same bound validation sources as ``evaluate_crossfit_budget_grid``
but evaluates variants that are not allowed to stop before the requested concept
budget when a legal non-empty action remains.

The diagnostic separates two failure modes:
  * forced value catches up -> learned STOP / cost calibration is the bottleneck;
  * forced value still lags -> action ranking / value target is the bottleneck.
"""
import argparse
import math
import time
from pathlib import Path

from cbmjev.config import resolve_config
from cbmjev.contracts import DeclaredCost, candidate_actions, stable_hash
from cbmjev.crossfit_evaluation import _load_sources, _verify_binding_files_unchanged
from cbmjev.evaluation import summarize_traces
from cbmjev.io import fresh_dir, write_json, write_jsonl
from cbmjev.pipeline import code_fingerprint
from cbmjev.runtime import ReplayEnvironment, synchronize


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--planned", required=True)
    parser.add_argument("--merged", required=True)
    parser.add_argument("--responder", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--static-order-dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--methods", nargs="+",
                        default=["value_force", "value_singleton_force"])
    parser.add_argument("--cost-weight", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--responder-source-dir")
    return parser.parse_args()


def _choose_forced_value(observed, actions, *, method, controller, cost, cost_weight):
    if getattr(controller, "objective", None) != "value":
        raise ValueError("forced value diagnostics require a value-objective controller")
    if method == "value_singleton_force":
        actions = tuple(a for a in actions if len(a) <= 1)
    elif method != "value_force":
        raise ValueError("unsupported forced method: " + method)
    non_stop = tuple(a for a in actions if a)
    if non_stop:
        actions = non_stop
    scores = tuple(float(v) for v in controller.predict(observed, actions))
    if len(scores) != len(actions) or any(not math.isfinite(v) for v in scores):
        raise ValueError("controller returned invalid action scores")
    objectives = [-v + cost_weight * cost(observed, a) for a, v in zip(actions, scores)]
    best = min(range(len(actions)), key=lambda i: (
        objectives[i], cost(observed, actions[i]), actions[i]))
    return actions[best], {",".join(map(str, a)) if a else "STOP": v
                           for a, v in zip(actions, scores)}


def run_forced_episode(env, schema, head, *, method, controller, cost,
                       cost_weight, max_groups, max_cost, include_pairs,
                       include_all, pairs, device):
    synchronize(device)
    started = time.perf_counter()
    declared, policy_ms, steps = 0.0, 0.0, []
    for _ in range(schema.num_groups + 1):
        observed = env.state()
        acquired = sum(schema.group_mask(observed))
        if acquired >= max_groups:
            steps.append({"before": list(observed), "action": [],
                          "scores": {}, "legal_candidate_count": 1,
                          "scored_candidate_count": 0,
                          "declared_cost": 0.0, "decision": "STOP",
                          "forced_budget_stop": True})
            break
        synchronize(device)
        begin = time.perf_counter()
        actions = candidate_actions(observed, schema, include_pairs, include_all, pairs)
        remaining = math.inf if max_cost is None else max(0.0, max_cost - declared)
        actions = tuple(a for a in actions if len(a) + acquired <= max_groups and
                        cost(observed, a) <= remaining + 1e-10)
        action, scores = _choose_forced_value(
            observed, actions, method=method, controller=controller,
            cost=cost, cost_weight=cost_weight)
        synchronize(device)
        policy_ms += (time.perf_counter() - begin) * 1000
        step = {"before": list(observed), "action": list(action), "scores": scores,
                "legal_candidate_count": len(actions),
                "scored_candidate_count": len(scores),
                "declared_cost": cost(observed, action)}
        if not action:
            step["decision"] = "STOP"
            step["forced_budget_stop"] = False
            steps.append(step)
            break
        values = env.query(action)
        declared += step["declared_cost"]
        step.update(decision="ACQUIRE", atom_ids=list(schema.expand(action)),
                    values=list(values))
        steps.append(step)
    else:
        raise RuntimeError("forced episode did not terminate within K+1 steps")
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
    methods = list(args.methods)
    allowed = {"value_force", "value_singleton_force"}
    if not methods or len(set(methods)) != len(methods) or set(methods) - allowed:
        raise ValueError("unsupported, empty or duplicate methods")
    source_before = code_fingerprint()
    print(f"[forced-grid] loading sources once for budgets={args.budgets} methods={methods}", flush=True)
    schema, rows, config, head, controller, binding = _load_sources(
        Path(args.prepared), Path(args.planned), Path(args.merged),
        Path(args.responder), Path(args.cache), args.device,
        Path(args.static_order_dir) if args.static_order_dir else None,
        args.responder_source_dir)
    if controller.objective != "value":
        raise ValueError("forced value diagnostics require a value-objective merged controller")
    print(f"[forced-grid] loaded rows={len(rows)} groups={schema.num_groups}", flush=True)
    config["device"] = args.device
    config = resolve_config(config)
    if args.cost_weight is not None:
        config["policy"]["cost_weight"] = args.cost_weight
    cost = DeclaredCost(**config["cost"])
    include_pairs = config["learning"]["include_pairs"]
    include_all = config["learning"]["include_all"]
    traces, reports, settings = [], {}, {}
    for budget in args.budgets:
        if not 0 <= budget <= schema.num_groups:
            raise ValueError("budget outside valid group range")
        for method in methods:
            key = f"{method}_K{budget}"
            spec = {"method": method, "budget": budget, "config": config,
                    "mode": "offline_replay", "split": "validation",
                    "source_binding": binding,
                    "diagnostic": "forced_budget_no_early_stop"}
            spec["system_hash"] = stable_hash(spec)
            settings[key] = spec
            print(f"[forced-grid] replay start {key}", flush=True)
            current = []
            for row_index, row in enumerate(rows, 1):
                trace = run_forced_episode(
                    ReplayEnvironment(row["z"], schema), schema, head,
                    method=method, controller=controller, cost=cost,
                    cost_weight=config["policy"]["cost_weight"],
                    max_groups=budget, max_cost=config["policy"]["max_cost"],
                    include_pairs=include_pairs, include_all=include_all,
                    pairs=controller.pairs, device=args.device)
                trace.update(sample_id=row["sample_id"], group_id=row["group_id"],
                             y=row["y"], split="validation", policy_id=key,
                             system_hash=spec["system_hash"], seed=config["seed"],
                             num_query_groups=schema.num_groups,
                             num_atoms=schema.num_atoms, budget_groups=budget)
                current.append(trace)
                if row_index % 100 == 0 or row_index == len(rows):
                    print(f"[forced-grid] replay progress {key} {row_index}/{len(rows)}", flush=True)
            traces.extend(current)
            reports[key] = summarize_traces(current, num_classes=schema.num_classes)
            reports[key]["budget_groups"] = budget
            print(f"[forced-grid] replay done {key} "
                  f"acc={reports[key]['accuracy']:.6f} "
                  f"macro_f1={reports[key]['macro_f1']:.6f} "
                  f"mean_groups={reports[key]['mean_queried_groups']:.3f}",
                  flush=True)
    out = fresh_dir(out)
    write_jsonl(out / "traces.jsonl", traces)
    write_json(out / "settings.json", settings)
    write_json(out / "metrics.json", {"split": "validation",
        "mode": "offline_replay", "seed": config["seed"],
        "budgets": args.budgets, "methods": methods, "policies": reports,
        "paper_evidence": False,
        "diagnostic": "forced_budget_no_early_stop",
        "evidence_status": "OFFLINE_VALIDATION_DIAGNOSTIC_NOT_PAPER_EVIDENCE",
        "source_binding": binding, "source_code_hash": source_before})
    _verify_binding_files_unchanged(binding, Path(args.merged), Path(args.responder),
                                    Path(args.cache),
                                    Path(args.static_order_dir) if args.static_order_dir else None,
                                    planned_dir=Path(args.planned))
    if code_fingerprint() != source_before:
        raise ValueError("source code changed during forced-budget evaluation")
    receipt = {"format": "cbmjev-crossfit-forced-budget-grid-v1",
               "status": "COMPLETE", "paper_evidence": False,
               "evidence_status": "OFFLINE_VALIDATION_DIAGNOSTIC_NOT_PAPER_EVIDENCE",
               "source_binding": binding, "source_code_hash": source_before,
               "samples_per_policy": len(rows), "num_policies": len(reports),
               "budgets": args.budgets, "methods": methods,
               "diagnostic": "forced_budget_no_early_stop"}
    receipt["receipt_hash"] = stable_hash(receipt)
    write_json(out / "receipt.json", receipt)
    print(f"[forced-grid] complete out={out} policies={len(reports)}", flush=True)


if __name__ == "__main__":
    main()

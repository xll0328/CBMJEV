#!/usr/bin/env python3
"""Exact finite-empirical 0/1-risk E0 control for CEBaB singleton queries.

The dynamic and all 24 fixed-order policies use the same policy-fit empirical
response/label distribution, terminal 0/1 loss, frozen head, and query cost.
Only policy_fit objective selects the fixed order; validation is descriptive.
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import torch

from cbmjev.baselines import EmpiricalLookaheadPolicy
from cbmjev.contracts import DeclaredCost, candidate_actions, stable_hash
from cbmjev.evaluation import summarize_traces
from cbmjev.io import file_hash, fresh_dir, write_json, write_jsonl
from cbmjev.pipeline import load_model_bundle
from cbmjev.runtime import ReplayEnvironment


def eligible_rows(rows, role, limit):
    selected = [row for row in rows if row["split"] == role]
    if not selected:
        raise ValueError("missing " + role)
    return selected if limit is None else selected[:limit]


def fit_policy(rows, head, schema, order, *, max_states):
    return EmpiricalLookaheadPolicy.fit(
        rows, head, schema, depth=None, max_groups=4,
        include_pairs=False, include_all=False, max_states=max_states,
        order=order)


class CachedDecisions:
    """Cache decisions, not fitted statistics; equivalent histories share actions."""

    def __init__(self, policy, cost, weight, max_cost):
        self.policy, self.cost, self.weight, self.max_cost = policy, cost, weight, max_cost
        self.actions = {}

    def choose(self, state, accrued):
        key = (tuple(state), accrued)
        if key not in self.actions:
            schema = self.policy.schema
            options = candidate_actions(state, schema, include_pairs=False, include_all=False)
            if self.policy.order is not None:
                options = tuple(action for action in options
                                if action in self.policy._allowed_order_actions(state))
            remaining = math.inf if self.max_cost is None else max(0.0, self.max_cost - accrued)
            options = tuple(action for action in options if self.cost(state, action) <= remaining + 1e-10)
            if not options or () not in options:
                raise ValueError("STOP must remain feasible")
            selected = self.policy.choose(state, options, remaining, self.cost,
                                          cost_weight=self.weight,
                                          remaining_groups=sum(not v for v in schema.group_mask(state)))
            self.actions[key] = (selected, len(options))
        return self.actions[key]


def replay(row, head, policy, decisions, family):
    schema = policy.schema
    env = ReplayEnvironment(row["z"], schema)
    steps, accrued = [], 0.0
    while True:
        before = env.state()
        action, legal_count = decisions.choose(before, accrued)
        charge = decisions.cost(before, action)
        step = {"before": list(before), "action": list(action),
                "legal_candidate_count": legal_count, "declared_cost": charge}
        if not action:
            step["decision"] = "STOP"
            steps.append(step)
            break
        values = env.query(action)
        accrued += charge
        step.update(decision="ACQUIRE", atom_ids=list(schema.expand(action)),
                    values=list(values))
        steps.append(step)
        if len(steps) > schema.num_groups:
            raise RuntimeError("policy failed to stop within K queries")
    probabilities = [float(p) for p in head.probabilities(env.state())]
    prediction = max(range(schema.num_classes), key=lambda index: probabilities[index])
    return {"sample_id": row["sample_id"], "group_id": row["group_id"],
            "split": row["split"], "y": row["y"], "method": family,
            "family": family, "prediction": prediction,
            "probabilities": probabilities, "final_state": list(env.state()),
            "queried_groups": [g for g, present in enumerate(schema.group_mask(env.state())) if present],
            "queried_atoms": [a for a, value in enumerate(env.state()) if value != -1],
            "calls": env.calls, "declared_cost": accrued,
            "cost_units": decisions.cost.units, "mode": "offline_replay", "steps": steps}


def evaluate(rows, head, policy, *, family, cost, weight, max_cost):
    decisions = CachedDecisions(policy, cost, weight, max_cost)
    traces = [replay(row, head, policy, decisions, family) for row in rows]
    return traces, summarize_traces(traces, num_classes=policy.schema.num_classes), len(decisions.actions)


def empirical_objective(report, weight):
    return report["error"] + weight * report["mean_declared_cost"]


def select_fixed_order(fit_rows, weight):
    if len(fit_rows) != 24 or len({tuple(row["order"]) for row in fit_rows}) != 24:
        raise ValueError("all 24 policy-fit fixed orders are required")
    return min(fit_rows, key=lambda row: (empirical_objective(row["report"], weight),
                                          tuple(row["order"])))


def paired_comparison(static, dynamic):
    a = {row["sample_id"]: row for row in static}
    b = {row["sample_id"]: row for row in dynamic}
    if not a or set(a) != set(b):
        raise ValueError("paired policies require matching nonempty sample sets")
    result = {"num_samples": len(a), "path_different": 0, "queryset_different": 0,
              "prediction_different": 0, "dynamic_correct_static_wrong": 0,
              "static_correct_dynamic_wrong": 0}
    for sample_id, left in a.items():
        right = b[sample_id]
        if (left["group_id"], left["split"], left["y"]) != (right["group_id"], right["split"], right["y"]):
            raise ValueError("paired metadata mismatch")
        path = lambda row: [step["action"] for step in row["steps"] if step["decision"] == "ACQUIRE"]
        result["path_different"] += path(left) != path(right)
        result["queryset_different"] += left["queried_groups"] != right["queried_groups"]
        result["prediction_different"] += left["prediction"] != right["prediction"]
        result["dynamic_correct_static_wrong"] += right["prediction"] == right["y"] and left["prediction"] != left["y"]
        result["static_correct_dynamic_wrong"] += left["prediction"] == left["y"] and right["prediction"] != right["y"]
    return result


def run(models, cache, out, *, cpu_threads=2, limit=None, max_states=20000):
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be positive")
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("output exists and is not empty")
    torch.set_num_threads(cpu_threads)
    schema, rows, manifest, config, receipt, head, _, _ = load_model_bundle(models, cache, device="cpu")
    if schema.dataset != "cebab" or schema.num_groups != 4:
        raise ValueError("requires four-group CEBaB schema")
    weight = config["policy"]["cost_weight"]
    max_cost = config["policy"]["max_cost"]
    cost = DeclaredCost(**config["cost"])
    fit = eligible_rows(rows, "policy_fit", limit)
    validation = eligible_rows(rows, "validation", limit)
    # Only this role enters the empirical model; validation and test never fit it.
    fit_source = [row for row in rows if row["split"] == "policy_fit"]
    if limit is not None:
        fit_source = fit_source[:limit]
    policies = {}
    fit_reports, validation_reports, validation_traces = [], [], []
    selected_traces = None
    out = fresh_dir(out)
    for order in itertools.permutations(range(4)):
        family = "empirical_dp_fixed_" + "".join(map(str, order))
        policy = fit_policy(fit_source, head, schema, order, max_states=max_states)
        policies[order] = policy
        _, fit_report, fit_states = evaluate(fit, head, policy, family=family,
                                             cost=cost, weight=weight, max_cost=max_cost)
        fit_reports.append({"order": list(order), "family": family,
                            "decision_states": fit_states, "report": fit_report})
    selected = select_fixed_order(fit_reports, weight)
    for order, policy in policies.items():
        family = "empirical_dp_fixed_" + "".join(map(str, order))
        traces, summary, states = evaluate(validation, head, policy, family=family,
                                           cost=cost, weight=weight, max_cost=max_cost)
        validation_traces.extend(traces)
        validation_reports.append({"order": list(order), "family": family,
                                   "selected_on_policy_fit": list(order) == selected["order"],
                                   "decision_states": states, "objective": empirical_objective(summary, weight),
                                   "report": summary})
        if list(order) == selected["order"]:
            selected_traces = traces
    dynamic = fit_policy(fit_source, head, schema, None, max_states=max_states)
    dynamic_traces, dynamic_report, dynamic_states = evaluate(validation, head, dynamic,
        family="empirical_dp_dynamic", cost=cost, weight=weight, max_cost=max_cost)
    validation_traces.extend(dynamic_traces)
    report = {"format": "cbmjev-cebab-ordered-empirical-dp-v1",
              "evidence_status": "PILOT_SUBSAMPLE_NOT_CLAIM_EVIDENCE" if limit else "EXPLORATORY_VALIDATION_NOT_TEST",
              "seed": config["seed"], "test_evaluated": False,
              "method": {"terminal_loss": "frozen_head_0_1_error", "dp_depth": "exhaustive_empirical",
                         "empirical_fit_role": "policy_fit", "candidate_actions": "STOP_OR_SINGLETON",
                         "fixed_order_selection": "policy_fit_mean_0_1_error_plus_lambda_mean_declared_cost",
                         "training_data_in_current_patient_observation": False,
                         "unseen_history_rule": dynamic.report["unseen_history_rule"],
                         "cost": cost.to_dict(), "cost_weight": weight, "max_cost": max_cost,
                         "max_states": max_states, "cpu_threads": cpu_threads},
              "population": {"policy_fit_samples": len(fit), "validation_samples": len(validation),
                             "limit": limit,
                             "policy_fit_group_ids_hash": stable_hash(sorted({r["group_id"] for r in fit})),
                             "validation_group_ids_hash": stable_hash(sorted({r["group_id"] for r in validation}))},
              "selection": {"role": "policy_fit", "selected_fixed_order": selected["order"],
                            "selected_fixed_fit_objective": empirical_objective(selected["report"], weight),
                            "all_24_fit": fit_reports},
              "validation": {"all_24_fixed_orders": validation_reports,
                             "selected_fixed": next(row for row in validation_reports if row["selected_on_policy_fit"]),
                             "dynamic": {"family": "empirical_dp_dynamic", "decision_states": dynamic_states,
                                         "objective": empirical_objective(dynamic_report, weight),
                                         "report": dynamic_report},
                             "paired_dynamic_vs_selected_fixed": paired_comparison(selected_traces, dynamic_traces)},
              "bindings": {"models_receipt_sha256": file_hash(Path(models) / "receipt.json"),
                           "model_weights_sha256": receipt["models_sha256"],
                           "cache_manifest_sha256": file_hash(Path(cache) / "manifest.json"),
                           "cache_responses_sha256": manifest["responses_sha256"],
                           "script_sha256": file_hash(__file__)},
              "notes": ["Exact only for finite empirical policy-fit distribution and restricted action set.",
                        "Validation scores never select the fixed order; all 24 are descriptive.",
                        "No test data evaluated and no deployment latency inferred from replay."]}
    report["report_hash"] = stable_hash(report)
    write_jsonl(out / "validation_traces.jsonl", validation_traces)
    write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "cache", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--max-states", type=int, default=20000)
    parser.add_argument("--limit", type=int, default=None, help="pilot only, per role")
    args = parser.parse_args()
    report = run(args.models, args.cache, args.out, cpu_threads=args.cpu_threads,
                 limit=args.limit, max_states=args.max_states)
    print(json.dumps({"out": args.out, "seed": report["seed"],
                      "selected_order": report["selection"]["selected_fixed_order"],
                      "dynamic_objective": report["validation"]["dynamic"]["objective"],
                      "fixed_objective": report["validation"]["selected_fixed"]["objective"]}, sort_keys=True))


if __name__ == "__main__":
    main()

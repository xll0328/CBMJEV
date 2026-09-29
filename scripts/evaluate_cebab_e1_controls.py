#!/usr/bin/env python3
"""Frozen CEBaB E1 controls on policy_fit and validation, never test.

This is a same-responder/same-head comparison of confidence-only STOP gates,
the existing signed CE-gain controller, and an entropy-cap *sensitivity* of that
controller. The last family is NOT a reproduction of LAVOIR's trained target.
Each external J_lambda selects its own configuration using policy_fit only.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from cbmjev.contracts import DeclaredCost, stable_hash
from cbmjev.evaluation import summarize_traces
from cbmjev.io import file_hash, fresh_dir, write_json, write_jsonl
from cbmjev.pipeline import load_model_bundle
from cbmjev.runtime import ReplayEnvironment


LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.40)
THRESHOLDS = (0.25, 0.35, 0.45, 0.55, 0.65, 0.80, 1.0)
PENALTIES = (0.0, 0.01, 0.03, 0.10, 0.20)
METHODS = ("uncertainty_static", "uncertainty_value_order",
           "signed_value", "entropy_cap_sensitivity")


class Scores:
    def __init__(self, schema, head, controller):
        if controller.objective != "value":
            raise ValueError("E1 controls require a signed CE-value controller")
        self.schema, self.head, self.controller = schema, head, controller
        self.prob_cache, self.value_cache = {}, {}

    def probabilities(self, state):
        state = tuple(state)
        if state not in self.prob_cache:
            probs = tuple(float(x) for x in self.head.probabilities(state))
            if (len(probs) != self.schema.num_classes or
                    any(not math.isfinite(x) or not 0 <= x <= 1 for x in probs) or
                    abs(sum(probs) - 1.0) > 1e-5):
                raise ValueError("invalid frozen head probabilities")
            self.prob_cache[state] = probs
        return self.prob_cache[state]

    def values(self, state, actions):
        key = (tuple(state), tuple(actions))
        if key not in self.value_cache:
            values = tuple(float(x) for x in self.controller.predict(*key))
            if len(values) != len(actions) or any(not math.isfinite(x) for x in values):
                raise ValueError("invalid frozen controller scores")
            self.value_cache[key] = values
        return self.value_cache[key]


def choose(state, schema, scores, cost, order, method, parameter):
    seen = schema.group_mask(state)
    remaining = tuple(g for g in range(schema.num_groups) if not seen[g])
    if not remaining:
        return (), {}, "complete"
    if method.startswith("uncertainty_"):
        confidence = max(scores.probabilities(state))
        if confidence >= parameter:
            return (), {"confidence": confidence}, "confidence_gate"
        if method == "uncertainty_static":
            return (next(g for g in order if not seen[g]),), {"confidence": confidence}, "acquire"
        values = scores.values(state, tuple((g,) for g in remaining))
        best = min(range(len(remaining)), key=lambda i: (-values[i], remaining[i]))
        return (remaining[best],), {"confidence": confidence, **
                {str(g): value for g, value in zip(remaining, values)}}, "acquire"
    actions = tuple((g,) for g in remaining)
    values = scores.values(state, actions)
    if method == "entropy_cap_sensitivity":
        probs = scores.probabilities(state)
        entropy = -sum(p * math.log(max(p, 1e-300)) for p in probs)
        values = tuple(min(value, entropy) for value in values)
    weighted = tuple(value - parameter * cost(state, action)
                     for value, action in zip(values, actions))
    best = min(range(len(actions)), key=lambda i: (-weighted[i], actions[i]))
    detail = {str(g): value for g, value in zip(remaining, values)}
    if weighted[best] <= 0:
        return (), detail, "nonpositive_net_value"
    return actions[best], detail, "acquire"


def replay(row, schema, scores, cost, order, method, parameter):
    if method not in METHODS:
        raise ValueError("unknown E1 method")
    env = ReplayEnvironment(row["z"], schema)
    steps, declared = [], 0.0
    for _ in range(schema.num_groups + 1):
        state = env.state()
        action, detail, reason = choose(state, schema, scores, cost, order, method, parameter)
        step = {"before": list(state), "action": list(action), "scores": detail,
                "reason": reason, "declared_cost": cost(state, action),
                "legal_candidate_count": 1 + schema.num_groups - sum(schema.group_mask(state))}
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
        raise RuntimeError("E1 episode failed to stop")
    probabilities = scores.probabilities(env.state())
    prediction = max(range(schema.num_classes), key=lambda i: probabilities[i])
    return {"sample_id": row["sample_id"], "group_id": row["group_id"],
            "split": row["split"], "y": row["y"], "method": method,
            "prediction": prediction, "probabilities": list(probabilities),
            "queried_groups": [g for g, present in enumerate(schema.group_mask(env.state())) if present],
            "queried_atoms": [a for a, v in enumerate(env.state()) if v != -1],
            "calls": env.calls, "declared_cost": declared, "cost_units": cost.units,
            "mode": "offline_replay", "steps": steps, "final_state": list(env.state())}


def candidates(method):
    if method not in METHODS:
        raise ValueError("unknown E1 method")
    return THRESHOLDS if method.startswith("uncertainty_") else PENALTIES


def external_j(report, lam, num_groups):
    return report["error"] + lam * report["mean_queried_groups"] / num_groups


def select_configuration(fit_reports, method, lam, num_groups):
    available = [(p, r) for (m, p), r in fit_reports.items() if m == method]
    if {p for p, _ in available} != set(candidates(method)):
        raise ValueError("incomplete policy_fit sweep")
    return min(available, key=lambda item: (external_j(item[1], lam, num_groups),
                    item[1]["mean_queried_groups"], item[0]))[0]


def paired(a, b, lam, num_groups):
    left, right = {r["sample_id"]: r for r in a}, {r["sample_id"]: r for r in b}
    if not left or left.keys() != right.keys():
        raise ValueError("paired traces require identical nonempty samples")
    changes = {"path_different": 0, "query_set_different": 0,
               "prediction_different": 0, "a_correct_b_wrong": 0,
               "a_wrong_b_correct": 0, "mean_b_minus_a_j": 0.0}
    for key in left:
        x, y = left[key], right[key]
        if (x["group_id"], x["y"], x["split"]) != (y["group_id"], y["y"], y["split"]):
            raise ValueError("paired metadata mismatch")
        path = lambda row: tuple(tuple(s["action"]) for s in row["steps"] if s["action"])
        changes["path_different"] += path(x) != path(y)
        changes["query_set_different"] += x["queried_groups"] != y["queried_groups"]
        changes["prediction_different"] += x["prediction"] != y["prediction"]
        changes["a_correct_b_wrong"] += x["prediction"] == x["y"] and y["prediction"] != y["y"]
        changes["a_wrong_b_correct"] += x["prediction"] != x["y"] and y["prediction"] == y["y"]
        changes["mean_b_minus_a_j"] += (int(y["prediction"] != y["y"]) -
            int(x["prediction"] != x["y"]) + lam *
            (len(y["queried_groups"]) - len(x["queried_groups"])) / num_groups)
    changes["num_samples"] = len(left)
    changes["mean_b_minus_a_j"] /= len(left)
    return changes


def run(models, cache, out, *, limit=None, cpu_threads=2):
    if limit is not None and limit < 1 or cpu_threads < 1:
        raise ValueError("limit and cpu_threads must be positive")
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("output must be new or empty")
    torch.set_num_threads(cpu_threads)
    schema, rows, manifest, config, receipt, head, controller, order = (
        load_model_bundle(models, cache, device="cpu"))
    if schema.dataset != "cebab" or schema.num_groups != 4:
        raise ValueError("E1 controls require four-group CEBaB")
    if controller.objective != "value":
        raise ValueError("E1 controls require CE signed value")
    if sorted(order) != list(range(schema.num_groups)):
        raise ValueError("fitted static order is not a permutation")
    if config["policy"]["max_cost"] is not None:
        raise ValueError("E1 controls do not implement a hard max_cost constraint")
    fit = [r for r in rows if r["split"] == "policy_fit"]
    val = [r for r in rows if r["split"] == "validation"]
    if not fit or not val:
        raise ValueError("policy_fit and validation are required")
    role_counts = {"policy_fit": len(fit), "validation": len(val)}
    if limit is not None:
        fit, val = fit[:limit], val[:limit]
    cost = DeclaredCost(**config["cost"])
    scores = Scores(schema, head, controller)
    fit_reports = {}
    for method in METHODS:
        for parameter in candidates(method):
            traces = [replay(row, schema, scores, cost, order, method, parameter) for row in fit]
            fit_reports[(method, parameter)] = summarize_traces(traces, num_classes=schema.num_classes)
    selected, val_traces, comparisons = {}, [], {}
    for lam in LAMBDAS:
        selected[str(lam)] = {}
        selected_traces = {}
        for method in METHODS:
            parameter = select_configuration(fit_reports, method, lam, schema.num_groups)
            traces = [replay(row, schema, scores, cost, order, method, parameter) for row in val]
            report = summarize_traces(traces, num_classes=schema.num_classes)
            selected[str(lam)][method] = {"parameter": parameter,
                 "policy_fit_j": external_j(fit_reports[(method, parameter)], lam, schema.num_groups),
                 "validation_j": external_j(report, lam, schema.num_groups),
                 "validation": report}
            selected_traces[method] = traces
            for trace in traces:
                val_traces.append({**trace, "external_lambda": lam,
                                   "selected_parameter": parameter})
        comparisons[str(lam)] = {method: paired(selected_traces["signed_value"],
            selected_traces[method], lam, schema.num_groups) for method in METHODS
            if method != "signed_value"}
    report = {"format": "cbmjev-cebab-e1-frozen-controls-v1",
        "evidence_status": "PILOT_SUBSAMPLE_NOT_CLAIM_EVIDENCE" if limit is not None else
                           "EXPLORATORY_VALIDATION_NOT_TEST",
        "test_evaluated": False, "seed": config["seed"],
        "roles": {"policy_fit_samples": len(fit), "validation_samples": len(val),
                  "original_role_counts": role_counts, "limit": limit},
        "protocol": {"external_j": "error_rate + lambda * mean_queried_groups / 4",
            "lambdas": LAMBDAS, "thresholds": THRESHOLDS, "penalties": PENALTIES,
            "parameter_selection_role": "policy_fit", "evaluation_role": "validation",
            "same_frozen_responder_head_controller": True,
            "uncertainty": "maximum task-head class probability; gate only",
            "entropy_cap": "min(predicted signed CE gain, predictive entropy); posthoc sensitivity, NOT LAVOIR reproduction",
            "fitted_static_order": list(order), "cost": cost.to_dict()},
        "policy_fit_grid": [{"method": m, "parameter": p, "report": r} for (m, p), r in fit_reports.items()],
        "selected_validation": selected, "paired_vs_signed": comparisons,
        "bindings": {"models_receipt_sha256": file_hash(Path(models) / "receipt.json"),
            "model_weights_sha256": receipt["models_sha256"],
            "cache_manifest_sha256": file_hash(Path(cache) / "manifest.json"),
            "cache_responses_sha256": manifest["responses_sha256"],
            "script_sha256": file_hash(__file__)}}
    report["report_hash"] = stable_hash(report)
    out = fresh_dir(out)
    write_jsonl(out / "validation_traces.jsonl", val_traces)
    write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "cache", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--limit", type=int, help="pilot only: first N per role")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    result = run(args.models, args.cache, args.out, limit=args.limit,
                 cpu_threads=args.cpu_threads)
    print(json.dumps({"out": args.out, "seed": result["seed"],
        "validation_samples": result["roles"]["validation_samples"]}))


if __name__ == "__main__":
    main()

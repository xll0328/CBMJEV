#!/usr/bin/env python3
"""Validation diagnostic: compare adaptive accuracy to a cost-matched static mixture.

An independent coin flip chooses one of the two neighboring static-budget
policies for each case. Its *expected* accuracy and declared cost are linear
in the mixture weight. Group-level bootstrap intervals, when group metrics
are present, are conditional development diagnostics, not confirmatory tests
or evidence of a deployable tuned baseline or actual latency.
Macro-F1 is deliberately not interpolated because it is nonlinear in the
confusion matrix.
"""
import argparse
import json
import math
import random
import statistics
from pathlib import Path

from cbmjev.io import file_hash, write_json


def _finite(report, key, low=None, high=None):
    value = report.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"invalid {key}")
    value = float(value)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{key} outside valid range")
    return value


def _group_vectors(policies, names, samples):
    groups = [policies[name].get("group_metrics") for name in names]
    if any(not isinstance(group, dict) or not group for group in groups):
        return None
    keys = sorted(groups[0])
    if any(sorted(group) != keys for group in groups[1:]):
        raise ValueError("paired policies have different image groups")
    vectors = []
    for name, group in zip(names, groups):
        policy = policies[name]
        values = []
        for key in keys:
            item = group[key]
            count = item.get("num_samples")
            if type(count) is not int or count < 1:
                raise ValueError("invalid image-group sample count")
            error = _finite(item, "error", 0, 1)
            cost = _finite(item, "declared_cost", 0)
            values.append((count, error, cost))
        if sum(value[0] for value in values) != samples:
            raise ValueError("image-group sample counts disagree with policy")
        if not math.isclose(1 - sum(n * e for n, e, _ in values) / samples,
                            policy["accuracy"], abs_tol=1e-9):
            raise ValueError("image-group accuracy disagrees with policy")
        if not math.isclose(sum(n * c for n, _, c in values) / samples,
                            policy["mean_declared_cost"], abs_tol=1e-9):
            raise ValueError("image-group cost disagrees with policy")
        if vectors and any(a[0] != b[0] for a, b in zip(vectors[0], values)):
            raise ValueError("paired policies have different group sizes")
        vectors.append(values)
    return vectors


def _quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _paired_group_bootstrap(policies, row, baseline_method, *, repeats=2000):
    control_ids = sorted((name for name in policies
                          if name.startswith(baseline_method + "_K")),
                         key=lambda name: policies[name]["mean_declared_cost"])
    names = [row["adaptive_policy_id"], *control_ids]
    vectors = _group_vectors(policies, names, row["num_samples"])
    if vectors is None:
        return None
    rng = random.Random(row["seed"] + 73001)
    deltas = []
    for _ in range(repeats):
        chosen = [rng.randrange(len(vectors[0])) for _ in vectors[0]]
        means = []
        for policy in vectors:
            n = sum(policy[i][0] for i in chosen)
            error = sum(policy[i][0] * policy[i][1] for i in chosen) / n
            cost = sum(policy[i][0] * policy[i][2] for i in chosen) / n
            means.append((cost, 1 - error))
        target, adaptive_accuracy = means[0]
        controls = means[1:]
        if not all(a[0] < b[0] for a, b in zip(controls, controls[1:])):
            raise ValueError("bootstrap baseline costs are not ordered")
        pair = next(((a, b) for a, b in zip(controls, controls[1:])
                     if a[0] - 1e-10 <= target <= b[0] + 1e-10), None)
        if pair is None:
            raise ValueError("bootstrap adaptive cost outside static range")
        low, high = pair
        weight = (target - low[0]) / (high[0] - low[0])
        deltas.append(adaptive_accuracy - ((1 - weight) * low[1] + weight * high[1]))
    return {"unit": "image_group", "groups": len(vectors[0]), "repeats": repeats,
            "resamples_validation_cost_match": True,
            "percentile_95_interval": [_quantile(deltas, .025), _quantile(deltas, .975)],
            "positive_replicate_fraction": sum(delta > 0 for delta in deltas) / repeats,
            "interpretation": "Descriptive validation uncertainty conditional on fitted policies; not a confirmatory p-value or training-seed interval."}


def analyze_one(metrics, *, adaptive_policy="value_K16", baseline_method="static"):
    if metrics.get("split") != "validation" or metrics.get("mode") != "offline_replay":
        raise ValueError("cost-matched analysis requires validation offline replay")
    seed = metrics.get("seed")
    if type(seed) is not int or not metrics.get("source_binding"):
        raise ValueError("seed and source binding required")
    policies = metrics.get("policies")
    if not isinstance(policies, dict) or adaptive_policy not in policies:
        raise ValueError("adaptive policy missing")
    adaptive = policies[adaptive_policy]
    samples = adaptive.get("num_samples")
    if type(samples) is not int or samples < 1:
        raise ValueError("invalid sample count")
    target_cost = _finite(adaptive, "mean_declared_cost", 0)
    dynamic_accuracy = _finite(adaptive, "accuracy", 0, 1)
    controls = []
    for key, report in policies.items():
        if not key.startswith(baseline_method + "_K"):
            continue
        if report.get("num_samples") != samples or report.get("split") != "validation":
            raise ValueError("baseline sample count or split mismatch")
        controls.append({"policy_id": key,
                         "cost": _finite(report, "mean_declared_cost", 0),
                         "accuracy": _finite(report, "accuracy", 0, 1)})
    controls.sort(key=lambda row: (row["cost"], row["policy_id"]))
    if len(controls) < 2 or any(a["cost"] >= b["cost"]
                                for a, b in zip(controls, controls[1:])):
        raise ValueError("baseline costs must be distinct and increasing")
    if target_cost < controls[0]["cost"] - 1e-10 or target_cost > controls[-1]["cost"] + 1e-10:
        raise ValueError("adaptive cost lies outside baseline cost range")
    low, high = next(((a, b) for a, b in zip(controls, controls[1:])
                      if a["cost"] - 1e-10 <= target_cost <= b["cost"] + 1e-10),
                     (None, None))
    if low is None:
        raise ValueError("no adjacent baseline budgets bracket adaptive cost")
    high_probability = max(0.0, min(1.0,
        (target_cost - low["cost"]) / (high["cost"] - low["cost"])))
    expected_cost = (1 - high_probability) * low["cost"] + high_probability * high["cost"]
    expected_accuracy = ((1 - high_probability) * low["accuracy"] +
                         high_probability * high["accuracy"])
    if abs(expected_cost - target_cost) > 1e-8:
        raise ValueError("mixture failed to match declared cost")
    static_order = metrics["source_binding"].get("static_order", {}).get("order")
    paired_budgets = [key[len("fixed_K"):] for key in policies
                      if key.startswith("fixed_K") and
                      "static_K" + key[len("fixed_K"):] in policies]
    fixed_static_identical = bool(paired_budgets) and all(
        policies["fixed_K" + budget].get("group_metrics") ==
        policies["static_K" + budget].get("group_metrics")
        for budget in paired_budgets)
    return {"seed": seed, "split": "validation", "num_samples": samples,
            "adaptive_policy_id": adaptive_policy,
            "adaptive_mean_declared_cost": target_cost,
            "adaptive_mean_queried_groups": _finite(adaptive, "mean_queried_groups", 0),
            "adaptive_accuracy": dynamic_accuracy,
            "low_static": low, "high_static": high,
            "high_static_probability": high_probability,
            "static_mixture_expected_declared_cost": expected_cost,
            "static_mixture_expected_accuracy": expected_accuracy,
            "adaptive_minus_static_mixture_accuracy": dynamic_accuracy - expected_accuracy,
            "fixed_static_group_metrics_identical_all_shared_budgets": fixed_static_identical,
            "shared_fixed_static_budgets": sorted(int(k) for k in paired_budgets),
            "static_order_noncanonical": (static_order != list(range(len(static_order)))
                                          if isinstance(static_order, list) else None)}


def summarize(paths, *, adaptive_policy="value_K16", baseline_method="static"):
    if not paths:
        raise ValueError("at least one metric file required")
    rows, sources = [], []
    for path in paths:
        path = Path(path)
        metrics = json.loads(path.read_text(encoding="utf-8"))
        row = analyze_one(metrics, adaptive_policy=adaptive_policy,
                          baseline_method=baseline_method)
        row["paired_group_bootstrap"] = _paired_group_bootstrap(
            metrics["policies"], row, baseline_method)
        rows.append(row)
        sources.append({"path": str(path), "sha256": file_hash(path)})
    if len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("duplicate seed")
    rows.sort(key=lambda row: row["seed"])
    deltas = [row["adaptive_minus_static_mixture_accuracy"] for row in rows]
    return {"format": "cbmjev-cost-matched-static-mixture-diagnostic-v3",
            "evidence_status": "OFFLINE_VALIDATION_DIAGNOSTIC_NOT_PAPER_EVIDENCE",
            "adaptive_policy_id": adaptive_policy,
            "baseline_method": baseline_method,
            "baseline_construction": "independent_per_case_randomized_adjacent_static_budgets",
            "metric": "sample_accuracy_only_macro_f1_not_linear",
            "sources": sources, "rows": rows,
            "num_seeds": len(rows), "mean_accuracy_delta": statistics.mean(deltas),
            "sample_std_accuracy_delta": statistics.stdev(deltas) if len(deltas) > 1 else None,
            "positive_seed_count": sum(delta > 0 for delta in deltas),
            "limitations": ["Paired group bootstrap is conditional on fitted policies and reused development validation images, not independent training variability or a confirmatory interval",
                "Baseline mixture weights use inspected validation costs",
                "Historical budget-grid router passed fitted static order to fixed; inspect per-seed order audit before interpreting fixed as canonical schema order",
                "Declared cost and acquired count do not establish deployment latency"]}


def summarize_grid(paths, *, adaptive_policies, baseline_method="static"):
    """Cost-normalize a predeclared policy grid with one read per seed.

    Full-grid point estimates are cheap diagnostics; selected operating points
    can be passed to ``summarize`` for conditional image-group bootstraps.
    """
    policies_to_check = tuple(adaptive_policies)
    if not paths or not policies_to_check or len(set(policies_to_check)) != len(policies_to_check):
        raise ValueError("nonempty metric paths and unique adaptive policies required")
    if any(not (name.startswith("value_K") or name.startswith("value_singleton_K"))
           for name in policies_to_check):
        raise ValueError("grid adaptive policies must be value or value_singleton")
    if baseline_method != "static":
        raise ValueError("grid analysis requires an explicit fitted static comparator")
    by_policy = {name: [] for name in policies_to_check}
    sources, seeds = [], set()
    for item in paths:
        path = Path(item)
        metrics = json.loads(path.read_text(encoding="utf-8"))
        seed = metrics.get("seed")
        if seed in seeds:
            raise ValueError("duplicate seed")
        seeds.add(seed)
        sources.append({"path": str(path), "sha256": file_hash(path)})
        for name in policies_to_check:
            row = analyze_one(metrics, adaptive_policy=name,
                              baseline_method=baseline_method)
            row["paired_group_bootstrap"] = None
            by_policy[name].append(row)
    summaries = {}
    for name, rows in by_policy.items():
        rows.sort(key=lambda row: row["seed"])
        deltas = [row["adaptive_minus_static_mixture_accuracy"] for row in rows]
        summaries[name] = {
            "rows": rows,
            "num_seeds": len(rows),
            "mean_accuracy_delta": statistics.mean(deltas),
            "sample_std_accuracy_delta": statistics.stdev(deltas) if len(deltas) > 1 else None,
            "positive_seed_count": sum(delta > 0 for delta in deltas),
        }
    return {
        "format": "cbmjev-cost-matched-static-full-grid-diagnostic-v1",
        "evidence_status": "OFFLINE_VALIDATION_DIAGNOSTIC_NOT_PAPER_EVIDENCE",
        "baseline_method": baseline_method,
        "baseline_construction": "independent_per_case_randomized_adjacent_fitted_static_budgets",
        "metric": "sample_accuracy_only_macro_f1_not_linear",
        "sources": sources,
        "adaptive_policies": list(policies_to_check),
        "policies": summaries,
        "limitations": [
            "Point estimates only; no full-grid bootstrap or multiple-look correction",
            "Mixture weights use inspected validation costs and are not frozen deployable operating points",
            "Expected declared cost is matched, not each case's realized cost or deployment latency",
            "Canonical fixed and strong adaptive baselines are not represented by this fitted-static comparator",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--adaptive-policy", default="value_K16")
    parser.add_argument("--adaptive-policies", nargs="+",
                        help="Analyze a complete predeclared grid in one source pass; point estimates only")
    parser.add_argument("--baseline-method", default="static", choices=("fixed", "static"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise ValueError("output must be new: " + str(out))
    if args.adaptive_policies:
        report = summarize_grid(args.metrics, adaptive_policies=args.adaptive_policies,
                                baseline_method=args.baseline_method)
    else:
        report = summarize(args.metrics, adaptive_policy=args.adaptive_policy,
                           baseline_method=args.baseline_method)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    if args.adaptive_policies:
        print(json.dumps({"out": str(out), "seeds": sorted({row["seed"]
            for policy in report["policies"].values() for row in policy["rows"]}),
            "num_policies": len(report["policies"])}, sort_keys=True))
    else:
        print(json.dumps({"out": str(out), "seeds": [r["seed"] for r in report["rows"]],
                          "mean_accuracy_delta": report["mean_accuracy_delta"]}, sort_keys=True))


if __name__ == "__main__":
    main()

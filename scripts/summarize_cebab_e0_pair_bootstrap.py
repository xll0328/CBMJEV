#!/usr/bin/env python3
"""Family-cluster uncertainty for the E0 policy-fit-selected STOP comparison.

This is exploratory validation uncertainty for fixed trained models, not a
training-seed confidence interval or an independent-family certification.
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from cbmjev.io import file_hash, fresh_dir, write_json


def quantile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def compare(report, traces_path, *, resamples=5000, seed=721):
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if report["format"] != "cbmjev-cebab-order-stop-factorial-v1" or report["test_evaluated"]:
        raise ValueError("expected non-test E0 factorial report")
    order = "".join(map(str, report["selection"]["selected_order_by_cell"]["stop_capK4"]))
    static_name = f"fixed_order_{order}_stop_capK4"
    dynamic_name = "dynamic_singleton_stop_capK4"
    selected = {static_name: {}, dynamic_name: {}}
    with open(traces_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["family"] in selected:
                if row["split"] != "validation":
                    raise ValueError("non-validation trace in E0 comparison")
                family = selected[row["family"]]
                if row["sample_id"] in family:
                    raise ValueError("duplicate sample ID")
                family[row["sample_id"]] = row
    fixed, dynamic = selected[static_name], selected[dynamic_name]
    if not fixed or set(fixed) != set(dynamic):
        raise ValueError("E0 traces are missing or unpaired")
    expected = report["population"]["validation_samples"]
    if len(fixed) != expected:
        raise ValueError("trace count does not match E0 report")
    weight = report["configuration"]["cost_weight"]
    by_family = defaultdict(list)
    for sample_id, left in fixed.items():
        right = dynamic[sample_id]
        if (left["group_id"], left["y"]) != (right["group_id"], right["y"]):
            raise ValueError("paired metadata mismatch")
        error_delta = int(right["prediction"] != right["y"]) - int(left["prediction"] != left["y"])
        cost_delta = right["declared_cost"] - left["declared_cost"]
        by_family[left["group_id"]].append((error_delta, cost_delta,
                                              error_delta + weight * cost_delta))
    families = sorted(by_family)
    group_means = [[statistics.mean(v[i] for v in by_family[f]) for i in range(3)]
                   for f in families]
    rng = random.Random(seed)
    draws = [[], [], []]
    for _ in range(resamples):
        chosen = [group_means[rng.randrange(len(families))] for _ in families]
        for i in range(3):
            draws[i].append(statistics.mean(group[i] for group in chosen))
    keys = ("error", "declared_cost", "J")
    estimates = {key: {"dynamic_minus_static_equal_family_mean":
                           statistics.mean(group[i] for group in group_means),
                       "ci95_low": quantile(draws[i], .025),
                       "ci95_high": quantile(draws[i], .975)}
                 for i, key in enumerate(keys)}
    return {"format": "cbmjev-cebab-e0-paired-family-bootstrap-v1",
            "evidence_status": "EXPLORATORY_VALIDATION_FIXED_TRAINED_MODELS_ONLY",
            "seed": report["seed"], "test_evaluated": False,
            "static_family": static_name, "dynamic_family": dynamic_name,
            "samples": len(fixed), "families": len(families),
            "resamples": resamples, "bootstrap_seed": seed,
            "cost_weight": weight, "estimates": estimates,
            "notes": ["Family resampling does not include training-seed uncertainty.",
                      "CEBaB family independence is not verified; intervals are descriptive." ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e0", required=True, help="one seed's E0 output directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    args = parser.parse_args()
    source = Path(args.e0)
    report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    result = compare(report, source / "validation_traces.jsonl", resamples=args.resamples)
    result["source_report_sha256"] = file_hash(source / "report.json")
    out = fresh_dir(args.out)
    write_json(out / "report.json", result)
    print(json.dumps({"out": str(out), "seed": result["seed"],
                      "J": result["estimates"]["J"]}, sort_keys=True))


if __name__ == "__main__":
    main()

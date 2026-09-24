#!/usr/bin/env python3
"""Compare a forced dynamic concept path with its matched fixed-budget path.

This is an offline validation diagnostic, not a significance test or a paper
speed measurement. Both trace files must use the same validation examples.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from cbmjev.io import file_hash, write_json


def load_policy(path, policy_id):
    rows = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("policy_id") != policy_id:
                continue
            sample_id = row["sample_id"]
            if sample_id in rows:
                raise ValueError(f"duplicate sample_id {sample_id} in {path}")
            rows[sample_id] = row
    if not rows:
        raise ValueError(f"policy {policy_id} not found in {path}")
    return rows


def first_action(row):
    for step in row["steps"]:
        if step["action"]:
            return tuple(step["action"])
    return ()


def analyze(dynamic, fixed):
    if set(dynamic) != set(fixed):
        raise ValueError("dynamic and fixed sample IDs do not match")
    sets, firsts, fixed_sets = Counter(), Counter(), Counter()
    step_actions = []
    outcomes = Counter()
    group_overlap = []
    for sample_id, d in dynamic.items():
        f = fixed[sample_id]
        if d["y"] != f["y"] or d["split"] != f["split"]:
            raise ValueError(f"target or split mismatch for {sample_id}")
        ds = frozenset(d["queried_groups"])
        fs = frozenset(f["queried_groups"])
        sets[tuple(sorted(ds))] += 1
        fixed_sets[tuple(sorted(fs))] += 1
        firsts[first_action(d)] += 1
        acquisitions = [tuple(step["action"]) for step in d["steps"]
                        if step["action"]]
        if not step_actions:
            step_actions = [Counter() for _ in acquisitions]
        if len(acquisitions) != len(step_actions):
            raise ValueError("dynamic traces do not use a common exact budget")
        for counter, action in zip(step_actions, acquisitions):
            counter[action] += 1
        group_overlap.append(len(ds & fs))
        outcomes[(d["prediction"] == d["y"],
                  f["prediction"] == f["y"])] += 1
    n = len(dynamic)
    probabilities = [count / n for count in sets.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities)
    top_set, top_count = sets.most_common(1)[0]
    steps = []
    for index, counter in enumerate(step_actions, 1):
        action, count = counter.most_common(1)[0]
        steps.append({"step": index, "distinct_actions": len(counter),
                      "modal_action": list(action), "modal_fraction": count / n,
                      "entropy_bits": -sum((v / n) * math.log2(v / n)
                                           for v in counter.values())})
    return {
        "samples": n,
        "split": next(iter(dynamic.values()))["split"],
        "dynamic_unique_group_sets": len(sets),
        "fixed_unique_group_sets": len(fixed_sets),
        "dynamic_top_group_set": list(top_set),
        "dynamic_top_group_set_count": top_count,
        "dynamic_top_group_set_fraction": top_count / n,
        "dynamic_group_set_entropy_bits": entropy,
        "dynamic_first_action_counts": [
            {"action": list(action), "count": count}
            for action, count in firsts.most_common()],
        "dynamic_step_action_diversity": steps,
        "mean_group_overlap_with_fixed": sum(group_overlap) / n,
        "dynamic_accuracy": (outcomes[(True, True)] + outcomes[(True, False)]) / n,
        "fixed_accuracy": (outcomes[(True, True)] + outcomes[(False, True)]) / n,
        "paired_correctness": {
            "both_correct": outcomes[(True, True)],
            "dynamic_only_correct": outcomes[(True, False)],
            "fixed_only_correct": outcomes[(False, True)],
            "both_wrong": outcomes[(False, False)],
        },
        "evidence_status": "OFFLINE_VALIDATION_DIAGNOSTIC_NOT_PAPER_EVIDENCE",
    }


def audit_saved_baselines(report_paths, metric_paths):
    """Identify the actual comparator behind historical ``fixed`` labels.

    Saved trajectory reports do not contain the underlying traces locally, so
    this binds their declared trace path, sample count and accuracy to the
    corresponding source-hashed budget-grid metrics. It does not re-audit the
    unavailable trace contents.
    """
    if not report_paths or len(report_paths) != len(metric_paths):
        raise ValueError("report and metric paths must be nonempty matched lists")
    rows = []
    for report_path, metric_path in zip(report_paths, metric_paths):
        report_path, metric_path = Path(report_path), Path(metric_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        fixed_id = report.get("fixed_policy")
        if not isinstance(fixed_id, str) or not fixed_id.startswith("fixed_K"):
            raise ValueError("saved report does not name a fixed budget")
        if Path(report.get("fixed_traces", "")).parent.resolve() != metric_path.parent.resolve():
            raise ValueError("saved fixed trace path is not bound to metrics directory")
        if metrics.get("split") != "validation" or metrics.get("mode") != "offline_replay":
            raise ValueError("expected validation offline-replay metrics")
        policies = metrics.get("policies", {})
        static_id = "static_K" + fixed_id[len("fixed_K"):]
        if fixed_id not in policies or static_id not in policies:
            raise ValueError("fixed/static comparator pair absent")
        fixed = policies[fixed_id]
        saved_accuracy = report.get("fixed_accuracy")
        source_accuracy = fixed.get("accuracy")
        if (report.get("samples") != fixed.get("num_samples") or
                any(isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in (saved_accuracy, source_accuracy)) or
                not math.isclose(saved_accuracy, source_accuracy, abs_tol=1e-10)):
            raise ValueError("saved trajectory count/accuracy disagrees with source metrics")
        shared = sorted((name for name in policies if name.startswith("fixed_K")
                         and "static_K" + name[len("fixed_K"):] in policies),
                        key=lambda name: int(name[len("fixed_K"):]))
        if not shared or any(not isinstance(policies[name].get("group_metrics"), dict)
                             or not policies[name]["group_metrics"]
                             or {k: v for k, v in policies[name].items() if k != "method"} !=
                             {k: v for k, v in policies["static_K" + name[len("fixed_K"):]].items()
                              if k != "method"}
                             for name in shared):
            raise ValueError("fixed/static per-group alias is not verified")
        order = metrics.get("source_binding", {}).get("static_order", {}).get("order")
        if not isinstance(order, list) or order == list(range(len(order))):
            raise ValueError("fitted static order is absent or canonical")
        rows.append({"seed": metrics.get("seed"),
                     "saved_report": {"path": str(report_path), "sha256": file_hash(report_path)},
                     "budget_metrics": {"path": str(metric_path), "sha256": file_hash(metric_path)},
                     "declared_fixed_trace_path": report["fixed_traces"],
                     "raw_fixed_trace_available": Path(report["fixed_traces"]).exists(),
                     "fixed_policy_id": fixed_id, "actual_comparator": "fitted_static_prefix",
                     "static_policy_id": static_id, "shared_aliased_budgets":
                     [int(name[len("fixed_K"):]) for name in shared],
                     "sample_count": report["samples"], "accuracy": report["fixed_accuracy"]})
    if any(type(row["seed"]) is not int for row in rows) or len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("missing or duplicate seed")
    rows.sort(key=lambda row: row["seed"])
    return {"format": "cbmjev-forced-trajectory-baseline-provenance-v2",
            "status": "HISTORICAL_FIXED_IS_FITTED_STATIC",
            "audit_script_sha256": file_hash(__file__),
            "alias_check": "all_policy_fields_except_method_equal_at_each_shared_budget",
            "rows": rows,
            "limitation": "Raw fixed trace contents are not re-audited when unavailable locally; path, count, accuracy, and source group metrics are bound."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-traces")
    parser.add_argument("--fixed-traces")
    parser.add_argument("--dynamic-policy", default="value_singleton_force_K16")
    parser.add_argument("--fixed-policy", default="fixed_K16")
    parser.add_argument("--audit-existing-reports", nargs="+")
    parser.add_argument("--baseline-metrics", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if args.audit_existing_reports or args.baseline_metrics:
        if args.dynamic_traces or args.fixed_traces:
            raise ValueError("audit mode does not accept trace inputs")
        report = audit_saved_baselines(args.audit_existing_reports, args.baseline_metrics)
    else:
        if not args.dynamic_traces or not args.fixed_traces:
            raise ValueError("both trace inputs required")
        report = analyze(load_policy(args.dynamic_traces, args.dynamic_policy),
                         load_policy(args.fixed_traces, args.fixed_policy))
        report.update(dynamic_traces=args.dynamic_traces,
                      fixed_traces=args.fixed_traces,
                      dynamic_policy=args.dynamic_policy,
                      fixed_policy=args.fixed_policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

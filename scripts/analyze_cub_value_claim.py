#!/usr/bin/env python3
"""Analyze a CUB value-budget aggregate for evidence-scoped claim decisions.

The input is ``frontier_summary.json`` produced by
``scripts/figures/summarize_cub_budget_grids.py``.  The output is intentionally
conservative: it reports matched-budget deltas and a suggested evidence status,
but it does not edit the paper or promote claims automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


DYNAMIC = ("value", "value_singleton")
STATIC_OR_FIXED = ("static", "static_value", "fixed")
DEFAULT_BASELINES = ("static",)
SANITY = ("random", "all", "stop")


def label(method: str) -> str:
    return {
        "value": "Dynamic value",
        "value_singleton": "Dynamic singleton value",
        "static": "Static learned order",
        "static_value": "Static-value stopping",
        "fixed": "Fixed prefix",
        "random": "Random prefix",
        "all": "All concepts",
        "stop": "No concepts",
    }.get(method, method)


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric_value(row: dict, metric: str) -> float | None:
    return finite(row.get(metric + "_mean"))


def collect_by_budget(policies: dict) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in policies.values():
        budget = int(row["budget_groups"])
        grouped.setdefault(budget, []).append(row)
    return grouped


def best(rows: list[dict], methods: tuple[str, ...], metric: str) -> dict | None:
    candidates = [row for row in rows if row["method"] in methods and metric_value(row, metric) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: metric_value(row, metric) or float("-inf"))


def reject_duplicate_fixed_static(policies: dict) -> None:
    """Do not silently count indistinguishable fixed/static rows as distinct controls."""
    budgets = sorted({int(row["budget_groups"]) for row in policies.values()
                      if row["method"] in ("fixed", "static")})
    pairs = [(policies.get(f"fixed_K{k}"), policies.get(f"static_K{k}"))
             for k in budgets]
    if len(pairs) < 2 or any(a is None or b is None for a, b in pairs):
        return
    compared = ("accuracy_mean", "macro_f1_mean", "mean_queried_groups_mean",
                "mean_calls_mean", "mean_declared_cost_mean", "num_seeds")
    if all(all(a.get(key) == b.get(key) for key in compared) for a, b in pairs):
        raise ValueError("fixed/static aggregate rows are indistinguishable; audit source order and analyze static separately")


def analyze(summary: dict, metric: str, min_delta: float,
            baseline_methods: tuple[str, ...] = DEFAULT_BASELINES) -> dict:
    if not baseline_methods or set(baseline_methods) - set(STATIC_OR_FIXED):
        raise ValueError("baseline methods must be a nonempty static/fixed subset")
    if "fixed" in baseline_methods:
        reject_duplicate_fixed_static(summary["policies"])
    rows_by_budget = collect_by_budget(summary["policies"])
    comparisons = []
    positive = []
    for budget in sorted(rows_by_budget):
        rows = rows_by_budget[budget]
        dyn = best(rows, DYNAMIC, metric)
        base = best(rows, baseline_methods, metric)
        rand = best(rows, ("random",), metric)
        if dyn is None or base is None:
            continue
        dyn_value = metric_value(dyn, metric)
        base_value = metric_value(base, metric)
        rand_value = metric_value(rand, metric) if rand else None
        assert dyn_value is not None and base_value is not None
        delta = dyn_value - base_value
        comparisons.append({
            "budget_groups": budget,
            "dynamic_method": dyn["method"],
            "dynamic_label": label(dyn["method"]),
            "dynamic_mean": dyn_value,
            "baseline_method": base["method"],
            "baseline_label": label(base["method"]),
            "baseline_mean": base_value,
            "delta_vs_best_baseline": delta,
            "delta_vs_best_static_or_fixed": delta,
            "random_mean": rand_value,
            "delta_vs_random": (dyn_value - rand_value) if rand_value is not None else None,
            "num_seeds": dyn.get("num_seeds"),
        })
        if delta >= min_delta:
            positive.append(delta)
    if not comparisons:
        status = "insufficient_dynamic_or_static_rows"
        rationale = "No matched budget contains both a dynamic value policy and a requested baseline."
    else:
        best_delta = max(item["delta_vs_best_static_or_fixed"] for item in comparisons)
        worst_delta = min(item["delta_vs_best_static_or_fixed"] for item in comparisons)
        if best_delta >= min_delta and len(positive) >= 2:
            status = "candidate_positive_validation_claim"
            rationale = (
                "The selected dynamic policy has at least two descriptive matched-budget "
                f"improvements over the requested baselines of >= {min_delta:.4f}; "
                "this is not a significance test or a locked operating-point result."
            )
        elif best_delta > 0:
            status = "weak_or_budget_local_dynamic_gain"
            rationale = "Dynamic value improves at some budgets, but evidence is not broad enough for a strong claim."
        elif worst_delta < 0:
            status = "no_observed_matched_budget_gain"
            rationale = "No selected dynamic policy exceeds the requested baseline at any saved matched budget; statistical equivalence is not established."
        else:
            status = "tie_or_uninformative"
            rationale = "Matched dynamic and baseline rows are essentially tied under the configured threshold."
    return {
        "schema_version": "cbmjev-cub-value-claim-analysis-v2",
        "metric": metric,
        "baseline_methods": list(baseline_methods),
        "min_delta": min_delta,
        "seeds": summary.get("seeds"),
        "num_seeds": summary.get("num_seeds"),
        "status": status,
        "rationale": rationale,
        "comparisons": comparisons,
    }


def pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100.0 * value:.2f}"


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# CUB value-policy claim analysis",
        "",
        f"- Metric: `{report['metric']}`",
        f"- Seeds: {report.get('seeds')}",
        f"- Status: `{report['status']}`",
        f"- Rationale: {report['rationale']}",
        "",
        "| Budget | Dynamic | Dyn. | Best requested baseline | Base | Δ dyn-base | Random | Δ dyn-rand |",
        "|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for item in report["comparisons"]:
        lines.append(
            f"| {item['budget_groups']} | {item['dynamic_label']} | "
            f"{pct(item['dynamic_mean'])} | {item['baseline_label']} | "
            f"{pct(item['baseline_mean'])} | {pct(item['delta_vs_best_baseline'])} | "
            f"{pct(item['random_mean'])} | {pct(item['delta_vs_random'])} |"
        )
    lines.extend([
        "",
        "Interpretation is validation-scoped.  Promote to a paper claim only after",
        "the evidence state, claim matrix, and operating-point protocol are updated.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--metric", choices=["accuracy", "macro_f1"], default="accuracy")
    parser.add_argument("--min-delta", type=float, default=0.005,
                        help="Absolute metric delta required for a local positive result.")
    parser.add_argument("--baseline-methods", nargs="+", choices=STATIC_OR_FIXED,
                        default=list(DEFAULT_BASELINES),
                        help="Explicit comparator set; exclude fixed for audited historical fixed/static aliases.")
    args = parser.parse_args()

    source_path = Path(args.summary)
    summary = json.loads(source_path.read_text(encoding="utf-8"))
    report = analyze(summary, args.metric, args.min_delta, tuple(args.baseline_methods))
    report["input_source"] = {"path": str(source_path),
                              "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()}
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, Path(args.markdown_output))
    print(json.dumps({
        "status": report["status"],
        "metric": report["metric"],
        "num_comparisons": len(report["comparisons"]),
        "json_output": str(json_path),
        "markdown_output": args.markdown_output,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

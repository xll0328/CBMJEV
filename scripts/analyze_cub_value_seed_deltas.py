#!/usr/bin/env python3
"""Paired-seed delta analysis for CUB value-policy budget frontiers.

This consumes ``frontier_long.csv`` from
``scripts/figures/summarize_cub_budget_grids.py``.  Unlike the aggregate
claim analyzer, which compares mean frontiers, this script asks whether a
dynamic value policy beats the best static/fixed baseline within the same
seed and matched concept budget.  The output is intended as validation-scope
evidence for paper wording and ablation triage, not as an automatic claim
promotion gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


DYNAMIC = ("value", "value_singleton")
STATIC_OR_FIXED = ("static", "static_value", "fixed")
DEFAULT_BASELINES = ("static",)


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("frontier CSV is empty")
    required = {"seed", "method", "budget_groups", "policy_id", "accuracy", "macro_f1"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"frontier CSV missing columns: {sorted(missing)}")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        metric_accuracy = finite_float(row.get("accuracy"))
        metric_macro_f1 = finite_float(row.get("macro_f1"))
        if metric_accuracy is None or metric_macro_f1 is None:
            raise ValueError("frontier CSV contains non-finite metrics")
        parsed.append({
            **row,
            "seed": int(row["seed"]),
            "budget_groups": int(row["budget_groups"]),
            "accuracy": metric_accuracy,
            "macro_f1": metric_macro_f1,
        })
    return parsed


def best(rows: list[dict[str, Any]], methods: tuple[str, ...], metric: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["method"] in methods]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row[metric]))


def reject_duplicate_fixed_static(rows: list[dict[str, Any]]) -> None:
    by_key = {(row["seed"], row["method"], row["budget_groups"]): row for row in rows}
    seeds = {row["seed"] for row in rows}
    budgets = {row["budget_groups"] for row in rows}
    pairs = [(by_key.get((seed, "fixed", budget)), by_key.get((seed, "static", budget)))
             for seed in seeds for budget in budgets]
    if len(pairs) < 2 or any(a is None or b is None for a, b in pairs):
        return
    fields = ("accuracy", "macro_f1", "mean_queried_groups",
              "mean_calls", "mean_declared_cost")
    if all(all(a.get(key) == b.get(key) for key in fields) for a, b in pairs):
        raise ValueError("fixed/static seed rows are indistinguishable; audit source order and analyze static separately")


def summarize_deltas(deltas: list[float]) -> dict[str, Any]:
    if not deltas:
        return {
            "num_seed_pairs": 0,
            "mean_delta": None,
            "std_delta": None,
            "stderr_delta": None,
            "min_delta": None,
            "max_delta": None,
            "positive_seed_pairs": 0,
            "negative_seed_pairs": 0,
            "tied_seed_pairs": 0,
        }
    std = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    return {
        "num_seed_pairs": len(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "std_delta": std,
        "stderr_delta": std / math.sqrt(len(deltas)) if deltas else None,
        "min_delta": min(deltas),
        "max_delta": max(deltas),
        "positive_seed_pairs": sum(1 for value in deltas if value > 0),
        "negative_seed_pairs": sum(1 for value in deltas if value < 0),
        "tied_seed_pairs": sum(1 for value in deltas if value == 0),
    }


def analyze(rows: list[dict[str, Any]], metric: str, min_delta: float,
            baseline_methods: tuple[str, ...] = DEFAULT_BASELINES) -> dict[str, Any]:
    if not baseline_methods or set(baseline_methods) - set(STATIC_OR_FIXED):
        raise ValueError("baseline methods must be a nonempty static/fixed subset")
    if "fixed" in baseline_methods:
        reject_duplicate_fixed_static(rows)
    budgets = sorted({row["budget_groups"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    budget_reports = []
    strong_budgets = []
    weak_budgets = []
    for budget in budgets:
        paired = []
        for seed in seeds:
            subset = [row for row in rows if row["seed"] == seed and row["budget_groups"] == budget]
            dyn = best(subset, DYNAMIC, metric)
            base = best(subset, baseline_methods, metric)
            if dyn is None or base is None:
                continue
            delta = float(dyn[metric]) - float(base[metric])
            paired.append({
                "seed": seed,
                "dynamic_policy_id": dyn["policy_id"],
                "dynamic_method": dyn["method"],
                "dynamic_value": float(dyn[metric]),
                "baseline_policy_id": base["policy_id"],
                "baseline_method": base["method"],
                "baseline_value": float(base[metric]),
                "delta": delta,
            })
        stats = summarize_deltas([row["delta"] for row in paired])
        row = {"budget_groups": budget, "paired_deltas": paired, **stats}
        budget_reports.append(row)
        if stats["num_seed_pairs"] >= 2 and stats["mean_delta"] is not None:
            if stats["mean_delta"] >= min_delta and stats["negative_seed_pairs"] == 0:
                strong_budgets.append(budget)
            elif stats["mean_delta"] > 0:
                weak_budgets.append(budget)
    if not any(row["num_seed_pairs"] for row in budget_reports):
        status = "insufficient_paired_rows"
        rationale = "No matched seed/budget row contains both dynamic and requested baseline policies."
    elif strong_budgets:
        status = "paired_positive_candidate"
        rationale = (
            "At least one matched budget has a mean paired dynamic gain above "
            f"{min_delta:.4f} with no negative seed-pair deltas."
        )
    elif weak_budgets:
        status = "paired_weak_or_mixed_gain"
        rationale = "Some matched budgets have positive mean paired gain, but evidence is mixed or small."
    else:
        status = "paired_negative_or_tie"
        rationale = "Dynamic policies do not show positive matched seed/budget deltas."
    return {
        "schema_version": "cbmjev-cub-value-paired-seed-deltas-v2",
        "metric": metric,
        "baseline_methods": list(baseline_methods),
        "min_delta": min_delta,
        "seeds": seeds,
        "budgets": budgets,
        "status": status,
        "rationale": rationale,
        "budget_reports": budget_reports,
    }


def pct(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.2f}"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# CUB paired-seed value-policy deltas",
        "",
        f"- Metric: `{report['metric']}`",
        f"- Seeds: {report['seeds']}",
        f"- Status: `{report['status']}`",
        f"- Rationale: {report['rationale']}",
        "",
        "| Budget | Pairs | Mean Δ | Std Δ | SE Δ | Min Δ | Max Δ | + / 0 / - |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["budget_reports"]:
        lines.append(
            f"| {row['budget_groups']} | {row['num_seed_pairs']} | "
            f"{pct(row['mean_delta'])} | {pct(row['std_delta'])} | "
            f"{pct(row['stderr_delta'])} | {pct(row['min_delta'])} | "
            f"{pct(row['max_delta'])} | "
            f"{row['positive_seed_pairs']} / {row['tied_seed_pairs']} / {row['negative_seed_pairs']} |"
        )
    lines.extend([
        "",
        "Deltas are computed within the same seed and matched concept budget:",
        "best dynamic value policy minus best requested baseline.  This is",
        "validation-scoped evidence and should be interpreted together with the",
        "aggregate frontier and claim matrix.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-csv", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--metric", choices=["accuracy", "macro_f1"], default="accuracy")
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--baseline-methods", nargs="+", choices=STATIC_OR_FIXED,
                        default=list(DEFAULT_BASELINES),
                        help="Explicit comparator set; exclude fixed for audited historical fixed/static aliases.")
    args = parser.parse_args()

    source_path = Path(args.frontier_csv)
    report = analyze(load_rows(source_path), args.metric,
                     args.min_delta, tuple(args.baseline_methods))
    report["input_source"] = {"path": str(source_path),
                              "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()}
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.markdown_output))
    print(json.dumps({
        "status": report["status"],
        "metric": report["metric"],
        "json_output": str(json_path),
        "markdown_output": args.markdown_output,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

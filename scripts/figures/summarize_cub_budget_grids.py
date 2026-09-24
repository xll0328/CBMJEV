#!/usr/bin/env python3
"""Aggregate one or more CUB crossfit budget-grid metric files.

Inputs are produced by ``scripts/evaluate_crossfit_budget_grid.py``.  The
script writes a long-form per-seed CSV, a mean/std summary JSON, a compact
LaTeX table fragment, and optionally a mean frontier plot.  It is intentionally
agnostic to whether the grid is a no-value development run or a value-policy
run; unsupported or absent methods are simply omitted.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_plot_style import COLORS


LABELS = {
    "stop": "No concepts",
    "fixed": "Fixed prefix",
    "random": "Random prefix",
    "static": "Static learned order",
    "value": "Dynamic value",
    "value_singleton": "Dynamic singleton value",
    "static_value": "Static-value stopping",
    "all": "All concepts",
}

MARKERS = {
    "fixed": "s",
    "random": "x",
    "static": "D",
    "value": "o",
    "value_singleton": "^",
    "static_value": "v",
    "all": "*",
    "stop": ".",
}


def parse_policy_key(key: str) -> tuple[str, int]:
    if "_K" not in key:
        raise ValueError(f"policy key lacks budget suffix: {key}")
    method, budget = key.rsplit("_K", 1)
    return method, int(budget)


def load_grid(path: Path) -> list[dict]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if metrics.get("mode") != "offline_replay":
        raise ValueError(f"expected offline_replay metrics: {path}")
    seed = metrics.get("seed")
    rows = []
    for policy_id, report in sorted(metrics["policies"].items()):
        method, budget = parse_policy_key(policy_id)
        rows.append({
            "source": str(path),
            "seed": int(seed) if seed is not None else -1,
            "policy_id": policy_id,
            "method": method,
            "budget_groups": int(report.get("budget_groups", budget)),
            "accuracy": float(report["accuracy"]),
            "macro_f1": float(report["macro_f1"]),
            "mean_queried_groups": float(report["mean_queried_groups"]),
            "mean_calls": float(report["mean_calls"]),
            "mean_declared_cost": float(report["mean_declared_cost"]),
            "num_samples": int(report["num_samples"]),
        })
    return rows


def verify_fixed_static_alias(path: Path) -> None:
    """Fail closed before hiding historical duplicate fixed display rows."""
    metrics = json.loads(path.read_text(encoding="utf-8"))
    policies = metrics.get("policies", {})
    fitted = metrics.get("source_binding", {}).get("static_order", {}).get("order")
    if not isinstance(fitted, list) or fitted == list(range(len(fitted))):
        raise ValueError(f"noncanonical fitted static order required: {path}")
    fixed = {int(key.split("_K", 1)[1]) for key in policies
             if key.startswith("fixed_K")}
    static = {int(key.split("_K", 1)[1]) for key in policies
              if key.startswith("static_K")}
    if not fixed or fixed != static:
        raise ValueError(f"fixed/static budget sets must match: {path}")
    for budget in fixed:
        a = policies[f"fixed_K{budget}"]
        b = policies[f"static_K{budget}"]
        if a.get("group_metrics") != b.get("group_metrics"):
            raise ValueError(f"fixed/static groups differ at K{budget}: {path}")
        for name in ("accuracy", "macro_f1", "mean_declared_cost",
                     "mean_queried_groups", "mean_calls", "num_samples"):
            if a.get(name) != b.get(name):
                raise ValueError(f"fixed/static {name} differs at K{budget}: {path}")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["budget_groups"])].append(row)
    policies = {}
    for (method, budget), items in sorted(grouped.items()):
        key = f"{method}_K{budget}"
        policies[key] = {
            "method": method,
            "budget_groups": budget,
            "num_seeds": len({item["seed"] for item in items}),
            "seeds": sorted({item["seed"] for item in items}),
        }
        for metric in ("accuracy", "macro_f1", "mean_queried_groups",
                       "mean_calls", "mean_declared_cost"):
            values = [float(item[metric]) for item in items]
            policies[key][metric + "_mean"] = mean(values)
            policies[key][metric + "_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0)
    return {"policies": policies,
            "num_rows": len(rows),
            "num_seeds": len({row["seed"] for row in rows}),
            "seeds": sorted({row["seed"] for row in rows})}


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_pct(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{100.0 * value:.2f}"


def write_latex(summary: dict, path: Path, methods: list[str], budgets: list[int]) -> None:
    lines = [
        "% Auto-generated by scripts/figures/summarize_cub_budget_grids.py",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "Policy & Groups & Calls & Acc. & Macro-F1\\\\",
        "\\midrule",
    ]
    policies = summary["policies"]
    for method in methods:
        if method == "stop":
            candidates = [row for row in policies.values() if row["method"] == "stop"]
            if candidates:
                row = sorted(candidates, key=lambda item: item["budget_groups"])[0]
                lines.append(
                    f"{LABELS['stop']} & 0 & {row['mean_calls_mean']:.2f} & "
                    f"{format_pct(row['accuracy_mean'])} & "
                    f"{format_pct(row['macro_f1_mean'])}\\\\")
            continue
        for budget in budgets:
            key = f"{method}_K{budget}"
            if key not in policies:
                continue
            row = policies[key]
            label = LABELS.get(method, method.replace("_", " "))
            calls = row["mean_calls_mean"]
            lines.append(
                f"{label} & {budget} & {calls:.2f} & "
                f"{format_pct(row['accuracy_mean'])} & "
                f"{format_pct(row['macro_f1_mean'])}\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot(summary: dict, path: Path, methods: list[str], metric: str) -> None:
    import matplotlib.pyplot as plt
    from paper_plot_style import apply_style, save_figure

    apply_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    policies = summary["policies"]
    for method in methods:
        points = []
        for key, row in policies.items():
            if row["method"] == method:
                points.append(row)
        points.sort(key=lambda row: row["mean_queried_groups_mean"])
        if not points:
            continue
        x = [row["mean_queried_groups_mean"] for row in points]
        y = [100.0 * row[metric + "_mean"] for row in points]
        yerr = [100.0 * row[metric + "_std"] for row in points]
        ax.errorbar(x, y, yerr=yerr if any(v > 0 for v in yerr) else None,
                    marker=MARKERS.get(method, "o"),
                    color=COLORS.get(method, "#333333"),
                    label=LABELS.get(method, method.replace("_", " ")),
                    linestyle="--" if method == "random" else "-",
                    capsize=2 if any(v > 0 for v in yerr) else 0)
    ax.set_xlabel("Mean acquired concept groups")
    ax.set_ylabel("Validation accuracy (%)" if metric == "accuracy" else "Macro-F1 (%)")
    ax.set_xticks([0, 1, 2, 4, 8, 16, 28])
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    ax.legend(frameon=False, loc="lower right", handlelength=1.6, handletextpad=0.4)
    seeds = ",".join(str(s) for s in summary["seeds"])
    ax.text(0.02, 0.98, f"CUB validation seeds: {seeds}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7, color="#444444")
    save_figure(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--latex-output", required=True)
    parser.add_argument("--plot-output")
    parser.add_argument("--metric", choices=["accuracy", "macro_f1"], default="accuracy")
    parser.add_argument("--methods", nargs="+",
        default=["stop", "fixed", "random", "static", "value",
                 "value_singleton", "static_value", "all"])
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 28])
    parser.add_argument("--hide-fixed-static-alias", action="store_true",
                        help="Verify historical fixed/static identity in every input, then omit fixed from displayed table/plot only")
    args = parser.parse_args()
    if args.hide_fixed_static_alias:
        for item in args.inputs:
            verify_fixed_static_alias(Path(item))
        display_methods = [method for method in args.methods if method != "fixed"]
    else:
        display_methods = args.methods
    rows = []
    for item in args.inputs:
        rows.extend(load_grid(Path(item)))
    if not rows:
        raise ValueError("no rows loaded")
    write_csv(rows, Path(args.csv_output))
    summary = summarize(rows)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_latex(summary, Path(args.latex_output), display_methods, args.budgets)
    if args.plot_output:
        try:
            plot(summary, Path(args.plot_output), display_methods, args.metric)
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            placeholder = Path(args.plot_output).with_suffix(
                Path(args.plot_output).suffix + ".plot_unavailable.txt")
            placeholder.parent.mkdir(parents=True, exist_ok=True)
            placeholder.write_text(
                "Plot not generated because matplotlib is unavailable in this "
                "environment. CSV, summary JSON, and LaTeX outputs were still "
                "generated. Install matplotlib and rerun this script to create "
                f"{Path(args.plot_output).name}.\n",
                encoding="utf-8")
            print(f"[summarize] matplotlib unavailable; wrote {placeholder}",
                  file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

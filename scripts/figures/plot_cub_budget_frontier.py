#!/usr/bin/env python3
"""Plot CUB validation accuracy--concept-budget frontier from grid metrics.

The input is produced by ``scripts/evaluate_crossfit_budget_grid.py``.  This
figure is a single-seed validation diagnostic unless multiple grid outputs are
explicitly aggregated elsewhere.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_plot_style import COLORS, apply_style, save_figure


MARKERS = {"fixed": "s", "random": "x", "all": "o", "stop": "."}
LABELS = {
    "fixed": "Fixed prefix",
    "random": "Random prefix",
    "all": "All concepts",
    "stop": "No concepts",
}


def parse_policy_key(key: str) -> tuple[str, int]:
    if "_K" not in key:
        raise ValueError(f"policy key lacks budget suffix: {key}")
    method, budget = key.rsplit("_K", 1)
    return method, int(budget)


def load_rows(path: Path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("mode") != "offline_replay" or report.get("paper_evidence") is not False:
        raise ValueError("expected offline validation grid metrics with paper_evidence=false")
    rows = []
    for key, metrics in sorted(report["policies"].items()):
        method, budget = parse_policy_key(key)
        rows.append({
            "policy_id": key,
            "method": method,
            "budget_groups": budget,
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "mean_queried_groups": float(metrics["mean_queried_groups"]),
            "mean_calls": float(metrics["mean_calls"]),
            "mean_declared_cost": float(metrics["mean_declared_cost"]),
            "num_samples": int(metrics["num_samples"]),
        })
    if not rows:
        raise ValueError("empty metrics")
    return rows, report.get("seed")


def write_csv(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--metric", choices=["accuracy", "macro_f1"], default="accuracy")
    args = parser.parse_args()
    rows, seed = load_rows(Path(args.input))
    write_csv(rows, Path(args.csv_output))

    apply_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    for method in ("random", "fixed", "all"):
        points = sorted((row for row in rows if row["method"] == method),
                        key=lambda row: row["mean_queried_groups"])
        if not points:
            continue
        x = [row["mean_queried_groups"] for row in points]
        y = [100.0 * row[args.metric] for row in points]
        ax.plot(x, y, marker=MARKERS[method], color=COLORS.get(method, "#333333"),
                label=LABELS[method], linestyle="--" if method == "random" else "-")
    stop = [row for row in rows if row["method"] == "stop"]
    if stop:
        y0 = 100.0 * stop[0][args.metric]
        ax.axhline(y0, color="#bbbbbb", linewidth=0.8, linestyle=":", label=LABELS["stop"])
    ax.set_xlabel("Mean acquired concept groups")
    ax.set_ylabel("Validation accuracy (%)" if args.metric == "accuracy" else "Macro-F1 (%)")
    ax.set_xlim(-0.5, 28.8)
    ax.set_xticks([0, 1, 2, 4, 8, 16, 28])
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    ax.legend(frameon=False, loc="lower right", handlelength=1.8, handletextpad=0.4)
    ax.text(0.02, 0.98, f"CUB validation, seed {seed}; single run",
            transform=ax.transAxes, ha="left", va="top", fontsize=7, color="#444444")
    save_figure(fig, args.output)


if __name__ == "__main__":
    main()

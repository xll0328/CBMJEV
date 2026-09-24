#!/usr/bin/env python3
"""Plot the exploratory CEBaB validation accuracy--concept frontier.

The input must be the immutable output of tools/summarize_cost_sweep.py. This
script deliberately accepts one seed only and labels the figure accordingly;
it is not the final multi-seed paper result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_plot_style import COLORS, apply_style, save_figure


MARKERS = {"fixed": "s", "random": "x", "static": "D", "value": "o"}
LABELS = {"fixed": "Fixed order", "random": "Random", "static": "Static learned", "value": "Dynamic value"}


def load_frontier(path: Path, cost_weight: float):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("format") != "cbmjev-validation-cost-sweep-v1" or report.get("paper_claim") is not False:
        raise ValueError("expected an exploratory validation cost-sweep report")
    rows = [row for row in report["aggregate"]
            if row["method"] in MARKERS and abs(float(row["cost_weight"]) - cost_weight) < 1e-12]
    if not rows or any(row["num_seeds"] != 1 for row in rows):
        raise ValueError("this exploratory figure requires exactly one seed per point")
    seeds = {tuple(row["seeds"]) for row in rows}
    if len(seeds) != 1:
        raise ValueError("all frontier points must use the same seed")
    grouped = {method: sorted((row for row in rows if row["method"] == method),
                              key=lambda row: row["max_groups"])
               for method in MARKERS}
    if any(not points for points in grouped.values()):
        raise ValueError("missing a required baseline frontier")
    return grouped, next(iter(seeds))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cost-weight", type=float, default=0.03)
    args = parser.parse_args()
    grouped, seed = load_frontier(Path(args.input), args.cost_weight)

    apply_style()
    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    for method in ("random", "fixed", "static", "value"):
        points = grouped[method]
        x = [float(row["mean_queried_groups_mean"]) for row in points]
        y = [100.0 * float(row["accuracy_mean"]) for row in points]
        ax.plot(x, y, marker=MARKERS[method], color=COLORS[method], label=LABELS[method],
                linestyle="--" if method == "random" else "-")

    ax.set_xlabel("Mean acquired concept groups")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_xlim(-0.08, 4.08)
    ax.set_xticks(range(5))
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              handlelength=1.8, columnspacing=0.8, handletextpad=0.4)
    ax.text(0.02, 0.98, f"CEBaB, seed {seed}; validation only",
            transform=ax.transAxes, ha="left", va="top", fontsize=7, color="#444444")
    fig.subplots_adjust(bottom=0.28)
    save_figure(fig, args.output)


if __name__ == "__main__":
    main()

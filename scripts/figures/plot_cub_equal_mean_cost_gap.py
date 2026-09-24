#!/usr/bin/env python3
"""Plot the four-seed CUB validation gap at equal expected declared cost."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from paper_plot_style import COLORS, apply_style, save_figure


BUDGETS = (2, 4, 8, 16, 28)
METHODS = (
    ("value", "Value", "o", "-"),
    ("value_singleton", "Value, singleton", "s", "--"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_points(report: dict) -> list[dict]:
    if (report.get("format") != "cbmjev-cost-matched-static-full-grid-diagnostic-v1"
            or report.get("evidence_status") != "OFFLINE_VALIDATION_DIAGNOSTIC_NOT_PAPER_EVIDENCE"
            or report.get("baseline_method") != "static"
            or report.get("baseline_construction") !=
            "independent_per_case_randomized_adjacent_fitted_static_budgets"):
        raise ValueError("unexpected cost-matched diagnostic protocol")
    points = []
    for method, _, _, _ in METHODS:
        for budget in BUDGETS:
            key = f"{method}_K{budget}"
            policy = report["policies"][key]
            rows = policy["rows"]
            if (policy["num_seeds"] != 4 or len(rows) != 4
                    or sorted(row["seed"] for row in rows) != [60, 61, 62, 63]
                    or any(row["split"] != "validation" or row["num_samples"] != 594
                           or row["adaptive_policy_id"] != key
                           or not row["fixed_static_group_metrics_identical_all_shared_budgets"]
                           for row in rows)):
                raise ValueError(f"incomplete or mismatched policy rows: {key}")
            deltas = [row["adaptive_minus_static_mixture_accuracy"] for row in rows]
            mean, sd = statistics.mean(deltas), statistics.stdev(deltas)
            if (not all(math.isfinite(value) for value in deltas)
                    or not math.isclose(mean, policy["mean_accuracy_delta"], abs_tol=1e-12)
                    or not math.isclose(sd, policy["sample_std_accuracy_delta"], abs_tol=1e-12)):
                raise ValueError(f"saved aggregate differs from its seed rows: {key}")
            points.append({"policy": key, "budget": budget,
                           "mean_delta_pp": 100 * mean, "seed_sd_pp": 100 * sd,
                           "seed_deltas_pp": [100 * value for value in deltas]})
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT /
                        "results/main/cub_value_full_grid_equal_mean_cost_static_60_63_v1.json")
    parser.add_argument("--output", type=Path, default=PROJECT /
                        "paper/cvpr2027/figures/cub_equal_mean_cost_gap.pdf")
    parser.add_argument("--manifest", type=Path, default=PROJECT /
                        "paper/cvpr2027/generated/cub_equal_mean_cost_gap_manifest.json")
    args = parser.parse_args()
    report = json.loads(args.source.read_text(encoding="utf-8"))
    points = checked_points(report)

    apply_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    positions = list(range(len(BUDGETS)))
    for method, label, marker, linestyle in METHODS:
        selected = [point for point in points if point["policy"].startswith(method + "_K")]
        selected.sort(key=lambda point: point["budget"])
        ax.errorbar(positions, [point["mean_delta_pp"] for point in selected],
                    yerr=[point["seed_sd_pp"] for point in selected],
                    color=COLORS[method], marker=marker, linestyle=linestyle,
                    linewidth=1.4, markersize=4.3, elinewidth=0.8,
                    capsize=2.2, label=label, zorder=3)
    ax.axhline(0, color="#444444", linewidth=0.8, linestyle=":", zorder=2)
    ax.set_xticks(positions, [str(budget) for budget in BUDGETS])
    ax.set_xlabel("Maximum concept groups (K)")
    ax.set_ylabel("Accuracy difference (pp)")
    ax.set_xlim(-0.32, len(BUDGETS) - 0.68)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
    ax.legend(frameon=False, loc="lower right", handlelength=1.8)
    save_figure(fig, args.output)

    manifest = {
        "format": "cbmjev-cub-equal-mean-cost-gap-figure-v1",
        "evidence_status": "DEVELOPMENT_VALIDATION_DESCRIPTIVE_ONLY",
        "source_path": str(args.source),
        "source_sha256": sha256(args.source),
        "figure_path": str(args.output),
        "figure_sha256": sha256(args.output),
        "seeds": [60, 61, 62, 63],
        "validation_samples_per_seed": 594,
        "budgets": list(BUDGETS),
        "baseline": "independent randomized adjacent fitted-static budgets at equal expected declared cost",
        "error_bars": "sample standard deviation across four training seeds; same validation images reused",
        "points": points,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(json.dumps({"figure": str(args.output), "manifest": str(args.manifest),
                      "source_sha256": manifest["source_sha256"]}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate and summarize the four-seed canonical-fixed CUB replay.

Consumes only completed validation metrics/receipts and their matched historic
value-policy grids. Produces a source-hashed comparison summary, CSV, LaTeX
table, and accuracy-vs-mean-query figure for paper review. It never reads test
data and never promotes development evidence to paper evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


SEEDS = (60, 61, 62, 63)
BUDGETS = (4, 8, 16, 28)
POLICIES = ("canonical_fixed", "static", "value", "value_singleton", "static_value")
HISTORICAL_METHODS = ("static", "value", "value_singleton", "static_value")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def mean_sd(values):
    return {"mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def collect(runs_root: Path):
    rows = []
    sources = {}
    for seed in SEEDS:
        canonical_dir = runs_root / f"cub_eval_canonical_fixed_seed{seed}_v1"
        old_dir = runs_root / f"cub_eval_value_budget_grid_seed{seed}_v1"
        canonical_metrics_path = canonical_dir / "metrics.json"
        canonical_receipt_path = canonical_dir / "receipt.json"
        old_metrics_path = old_dir / "metrics.json"
        for path in (canonical_metrics_path, canonical_receipt_path, old_metrics_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        canonical = read_json(canonical_metrics_path)
        receipt = read_json(canonical_receipt_path)
        old = read_json(old_metrics_path)
        if receipt.get("status") != "COMPLETE" or receipt.get("paper_evidence") is not False:
            raise ValueError(f"seed {seed}: replay is incomplete or mislabeled")
        if receipt.get("samples_per_policy") != 594 or receipt.get("num_policies") != 7:
            raise ValueError(f"seed {seed}: unexpected sample or policy count")
        if (canonical.get("seed") != seed or canonical.get("split") != "validation" or
                canonical.get("mode") != "offline_replay" or
                canonical.get("paper_evidence") is not False):
            raise ValueError(f"seed {seed}: canonical metrics are not validation-only")
        expected = receipt.get("expected_grid_source", {})
        if (expected.get("historical_fixed_static_alias_verified") is not True or
                expected.get("sha256") != sha256(old_metrics_path)):
            raise ValueError(f"seed {seed}: expected-grid provenance mismatch")
        if "static_order" not in old.get("source_binding", {}) or "static_order" in canonical.get("source_binding", {}):
            raise ValueError(f"seed {seed}: canonical/fitted-order distinction is not verified")
        historic_policies = old.get("policies", {})
        canonical_policies = canonical.get("policies", {})
        for budget in BUDGETS:
            fixed = historic_policies[f"fixed_K{budget}"]
            fitted = historic_policies[f"static_K{budget}"]
            if ({k: v for k, v in fixed.items() if k != "method"} !=
                    {k: v for k, v in fitted.items() if k != "method"}):
                raise ValueError(f"seed {seed} K{budget}: historical fixed/static alias not exact")
            metrics = {
                "canonical_fixed": canonical_policies[f"fixed_K{budget}"],
                **{method: historic_policies[f"{method}_K{budget}"]
                   for method in HISTORICAL_METHODS},
            }
            for method, result in metrics.items():
                rows.append({
                    "seed": seed,
                    "budget": budget,
                    "policy": method,
                    "accuracy": float(result["accuracy"]),
                    "macro_f1": float(result["macro_f1"]),
                    "mean_queried_groups": float(result["mean_queried_groups"]),
                })
        sources[str(seed)] = {
            "canonical_metrics": str(canonical_metrics_path),
            "canonical_metrics_sha256": sha256(canonical_metrics_path),
            "canonical_receipt": str(canonical_receipt_path),
            "canonical_receipt_sha256": sha256(canonical_receipt_path),
            "historical_metrics": str(old_metrics_path),
            "historical_metrics_sha256": sha256(old_metrics_path),
            "historical_alias_verified": True,
            "expected_grid_source_sha256": expected["sha256"],
        }
    return rows, sources


def aggregate(rows):
    result = {}
    for policy in POLICIES:
        result[policy] = {}
        for budget in BUDGETS:
            subset = [row for row in rows if row["policy"] == policy and row["budget"] == budget]
            result[policy][str(budget)] = {
                key: mean_sd([row[key] for row in subset])
                for key in ("accuracy", "macro_f1", "mean_queried_groups")
            }
    deltas = {}
    for policy in POLICIES[1:]:
        deltas[policy] = {}
        for budget in BUDGETS:
            by_seed = {}
            for seed in SEEDS:
                method_row = next(r for r in rows if r["seed"] == seed and r["budget"] == budget and r["policy"] == policy)
                fixed_row = next(r for r in rows if r["seed"] == seed and r["budget"] == budget and r["policy"] == "canonical_fixed")
                by_seed[str(seed)] = {
                    "accuracy_pp": (method_row["accuracy"] - fixed_row["accuracy"]) * 100,
                    "macro_f1_pp": (method_row["macro_f1"] - fixed_row["macro_f1"]) * 100,
                }
            acc = [v["accuracy_pp"] for v in by_seed.values()]
            f1 = [v["macro_f1_pp"] for v in by_seed.values()]
            deltas[policy][str(budget)] = {
                "per_seed": by_seed,
                "accuracy_pp": mean_sd(acc),
                "macro_f1_pp": mean_sd(f1),
                "accuracy_wins": sum(value > 1e-12 for value in acc),
                "accuracy_ties": sum(abs(value) <= 1e-12 for value in acc),
            }
    return {"policies": result, "accuracy_deltas_vs_canonical_fixed": deltas}


def write_outputs(rows, summary, sources, out_dir: Path, paper_dir: Path,
                  figures_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cub_canonical_fixed_replay_long.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("seed", "budget", "policy", "accuracy", "macro_f1", "mean_queried_groups"))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = out_dir / "cub_canonical_fixed_replay_summary.json"
    summary_doc = {
        "format": "cbmjev-cub-canonical-fixed-comparison-v1",
        "dataset": "CUB-200-2011",
        "split": "validation",
        "seeds": list(SEEDS),
        "budgets": list(BUDGETS),
        "number_of_validation_rows_per_seed": 594,
        "paper_evidence": False,
        "evidence_status": "OFFLINE_VALIDATION_DEVELOPMENT_NOT_PAPER_OR_LATENCY_EVIDENCE",
        "statistical_note": "Mean and sample SD across four training/evaluation seeds; repeated validation images, descriptive only; no significance/equivalence inference.",
        "summary": summary,
        "sources": sources,
    }
    summary_path.write_text(json.dumps(summary_doc, indent=2, sort_keys=True), encoding="utf-8")
    table_path = paper_dir / "cub_canonical_fixed_vs_dynamic_table.tex"
    labels = {"canonical_fixed": "Schema-order fixed", "static": "Fitted static", "value": "Value", "value_singleton": "Value singleton", "static_value": "Static-value stop"}
    table_lines = [r"\begin{tabular}{@{}lr r r r@{}}", r"\toprule", r"Policy & $K$ & Accuracy (\%) & Macro-F1 (\%) & Groups \\", r"\midrule"]
    for budget in BUDGETS:
        for policy in POLICIES:
            cell = summary["policies"][policy][str(budget)]
            acc, f1, groups = cell["accuracy"], cell["macro_f1"], cell["mean_queried_groups"]
            table_lines.append(f"{labels[policy]} & {budget} & {acc['mean']*100:.2f} $\\pm$ {acc['sd']*100:.2f} & {f1['mean']*100:.2f} $\\pm$ {f1['sd']*100:.2f} & {groups['mean']:.2f} \\\\")
    table_lines += [r"\bottomrule", r"\end{tabular}"]
    table_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    figure_path = figures_dir / "cub_canonical_fixed_vs_dynamic_accuracy.pdf"
    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt
        style = {
            "canonical_fixed": ("Schema-order fixed", "#4C78A8", "o", "-"),
            "static": ("Fitted static", "#F58518", "s", "-"),
            "value": ("Value", "#54A24B", "^", "-"),
            "value_singleton": ("Value singleton", "#E45756", "D", "-"),
            "static_value": ("Fitted static + stop", "#B279A2", "v", "--"),
        }
        fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
        for policy in POLICIES:
            label, color, marker, line = style[policy]
            points = [summary["policies"][policy][str(k)] for k in BUDGETS]
            x = [p["mean_queried_groups"]["mean"] for p in points]
            y = [p["accuracy"]["mean"] * 100 for p in points]
            yerr = [p["accuracy"]["sd"] * 100 for p in points]
            ax.errorbar(x, y, yerr=yerr, label=label, color=color, marker=marker,
                        linestyle=line, linewidth=1.7, markersize=5, capsize=2)
        ax.set(xlabel="Mean acquired concept groups", ylabel="Validation accuracy (%)")
        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(figure_path)
        plt.close(fig)
    except ImportError:
        figure_path.with_suffix(".pdf.plot_unavailable.txt").write_text(
            "Install matplotlib to regenerate this validation-only figure.\n", encoding="utf-8")
        figure_path = None
    manifest_path = paper_dir / "cub_canonical_fixed_vs_dynamic_manifest.json"
    manifest = {
        "format": "cbmjev-cub-canonical-fixed-paper-assets-v1",
        "source_summary": str(summary_path),
        "source_summary_sha256": sha256(summary_path),
        "source_csv": str(csv_path),
        "source_csv_sha256": sha256(csv_path),
        "table_tex": str(table_path),
        "table_tex_sha256": sha256(table_path),
        "figure_pdf": str(figure_path) if figure_path else None,
        "figure_pdf_sha256": sha256(figure_path) if figure_path else None,
        "sources": sources,
        "paper_evidence": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"summary": str(summary_path), "csv": str(csv_path), "table": str(table_path),
            "figure": str(figure_path) if figure_path else None, "manifest": str(manifest_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/main/cub_canonical_fixed_replay_60_63"))
    parser.add_argument("--paper-generated", type=Path, default=Path("paper/cvpr2027/generated"))
    parser.add_argument("--paper-figures", type=Path, default=Path("paper/cvpr2027/figures"))
    args = parser.parse_args()
    rows, sources = collect(args.runs_root)
    summary = aggregate(rows)
    outputs = write_outputs(rows, summary, sources, args.out_dir,
                            args.paper_generated, args.paper_figures)
    print(json.dumps({"complete": True, "outputs": outputs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

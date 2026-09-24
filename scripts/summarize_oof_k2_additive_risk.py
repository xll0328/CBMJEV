#!/usr/bin/env python3
"""Summarize four CUB K2 additive-risk development runs with baseline binding."""
import argparse
import json
import math
from pathlib import Path

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash


SEEDS = (60, 61, 62, 63)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path,
                        default=Path("results/diagnostics"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/main/CUB_OOF_K2_ADDITIVE_RISK_20260923.md"))
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refuse to overwrite existing summary")
    rows, scripts = [], set()
    for seed in SEEDS:
        path = args.diagnostics / f"cub_oof_k2_additive_risk_seed{seed}_v1.json"
        reference = args.diagnostics / f"cub_oof_conditional_second_query_seed{seed}_v2.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        baseline = json.loads(reference.read_text(encoding="utf-8"))
        unsigned = dict(report)
        if unsigned.pop("report_hash", None) != stable_hash(unsigned):
            raise ValueError(f"invalid additive report hash: {path}")
        if (report["seed"] != seed or baseline["seed"] != seed
                or report["evidence_status"]
                != "TRAIN_OOF_MODEL_SELECTION_VALIDATION_EVALUATION_NOT_LOCKED_TEST"
                or report["validation_labels_used_for_fit_or_tuning"] is not False
                or report["training_rows"] != baseline["training_rows"]
                or report["validation_rows"] != baseline["validation_rows"]
                or report["first_group"] != baseline["first_group"]
                or report["selected_alpha"] not in (1.0, 10.0, 100.0, 1000.0)):
            raise ValueError(f"additive/reference binding mismatch: {seed}")
        a = report["comparisons"]["additive_vs_fixed"]
        l = report["comparisons"]["additive_vs_lookup"]
        b = baseline["aggregate"]
        for observed, expected in ((a["fixed_ce"], b["fixed_ce"]),
                                   (a["fixed_accuracy"], b["fixed_accuracy"]),
                                   (l["fixed_ce"], b["adaptive_ce"]),
                                   (l["fixed_accuracy"], b["adaptive_accuracy"])):
            if not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"not the same fixed/lookup comparator: {seed}")
        scripts.add(report["analysis_script_sha256"])
        rows.append((seed, report, path, file_hash(path)))
    if scripts != {file_hash(Path("scripts/evaluate_oof_k2_additive_risk.py"))}:
        raise ValueError("runs do not share the current additive-analysis script")
    lines = ["# CUB OOF K2 additive conditional-risk development diagnostic", "",
        "This is training-OOF-fitted, **development-validation** evidence; it is",
        "not a locked test, a K16/STOP result, a runtime claim, or a JEV effect.",
        "All models use exactly two singleton groups and the same final",
        "automatic-responder cache and task head per seed. Alpha is selected",
        "only by held-out OOF decision CE. The four seeds reuse the same 594",
        "validation images; their average is descriptive, not four independent",
        "dataset replications.", "",
        "| Seed | OOF alpha | Fixed CE | Lookup CE | Additive CE | Additive−lookup CE | Fixed acc. | Lookup acc. | Additive acc. | Additive−lookup acc. |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    delta_ce, delta_acc = [], []
    for seed, report, _, _ in rows:
        a = report["comparisons"]["additive_vs_fixed"]
        l = report["comparisons"]["additive_vs_lookup"]
        delta_ce.append(l["adaptive_minus_fixed_ce"])
        delta_acc.append(100 * l["adaptive_minus_fixed_accuracy"])
        lines.append(f"| {seed} | {report['selected_alpha']:g} | {a['fixed_ce']:.4f} | "
                     f"{l['fixed_ce']:.4f} | {l['adaptive_ce']:.4f} | "
                     f"{l['adaptive_minus_fixed_ce']:+.4f} | "
                     f"{100*a['fixed_accuracy']:.2f}% | "
                     f"{100*l['fixed_accuracy']:.2f}% | "
                     f"{100*l['adaptive_accuracy']:.2f}% | "
                     f"{100*l['adaptive_minus_fixed_accuracy']:+.2f} pp |")
    lines += ["", f"Descriptive seed-mean additive-minus-lookup CE: "
              f"{sum(delta_ce)/len(delta_ce):+.4f}; accuracy: "
              f"{sum(delta_acc)/len(delta_acc):+.2f} pp.", "",
              "The lookup has a fixed shrinkage of 20; the additive model has",
              "four training-OOF-selected ridge strengths. These are not exactly",
              "equal hyperparameter search budgets. All comparisons are",
              "exploratory because previous validation diagnostics informed this",
              "model design. A small CE gain cannot be promoted to a method claim",
              "without matched K16, strong controls, and locked-test evaluation.", "",
              "## Raw report bindings", ""]
    for seed, _, path, digest in rows:
        lines.append(f"- Seed {seed}: `{path}` (SHA-256 `{digest}`).")
    lines.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()

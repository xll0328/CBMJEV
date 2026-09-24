#!/usr/bin/env python3
"""Training-OOF-selected additive conditional-risk baseline at exact K2.

This is an exploratory CUB validation diagnostic, not a JEV-specific or
locked-test result. The first action is fixed by the training-only static
order. The model sees only that action's categorical answer and a candidate
second-group identity; realized counterfactual losses are training targets.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from cbmjev.contracts import stable_hash
from cbmjev.crossfit_artifacts import load_crossfit_head
from cbmjev.crossfit_cache import load_crossfit_cache
from cbmjev.io import file_hash, read_json
from cbmjev.learning import schema_signature
from cbmjev.provenance import validate_target_exclusion
from scripts.diagnose_conditional_second_query import (
    choose_second, paired_bootstrap, summarize,
)
from scripts.evaluate_oof_conditional_second_query import score_rows


ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def feature_matrix(outcomes, category_counts):
    """Intercept plus additive one-hot first-answer atoms; no hidden answer."""
    width = 1 + sum(category_counts)
    x = np.zeros((len(outcomes), width), dtype=np.float64)
    x[:, 0] = 1.0
    offsets = np.cumsum((1,) + tuple(category_counts[:-1]))
    for i, outcome in enumerate(outcomes):
        if len(outcome) != len(category_counts):
            raise ValueError("first-answer width mismatch")
        for value, count, offset in zip(outcome, category_counts, offsets):
            if type(value) is not int or not 0 <= value < count:
                raise ValueError("invalid first-answer category")
            x[i, offset + value] = 1.0
    return x


def fit_ridge(x, y, alpha):
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError("nonempty aligned matrices required")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("positive finite alpha required")
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def loss_matrix(rows, candidates):
    return np.asarray([[row["ce"][g] for g in candidates] for row in rows],
                      dtype=np.float64)


def choose_alpha(x, y, folds, alphas=ALPHAS):
    """Select regularization by held-out OOF decision CE, not MSE or validation."""
    if len(x) != len(y) or len(folds) != len(x) or len(set(folds)) < 2:
        raise ValueError("at least two aligned training folds required")
    fold_ids = sorted(set(folds))
    scores = {}
    for alpha in alphas:
        total = count = 0
        for fold in fold_ids:
            fit = np.asarray([f != fold for f in folds])
            heldout = ~fit
            weights = fit_ridge(x[fit], y[fit], alpha)
            predicted = x[heldout] @ weights
            actions = np.argmin(predicted, axis=1)
            total += float(y[heldout, actions].sum())
            count += int(heldout.sum())
        scores[str(alpha)] = total / count
    return min(alphas, key=lambda a: (scores[str(a)], a)), scores


def evaluate(rows, candidates, predictions, fixed, lookup):
    entries = []
    for row, predicted in zip(rows, predictions):
        additive = candidates[int(np.argmin(predicted))]
        table = lookup.get(row["outcome"], fixed)
        entries.append({"additive_action": additive, "lookup_action": table,
                        "fixed_action": fixed,
                        "additive_ce": row["ce"][additive],
                        "lookup_ce": row["ce"][table],
                        "fixed_ce": row["ce"][fixed],
                        "additive_correct": row["correct"][additive],
                        "lookup_correct": row["correct"][table],
                        "fixed_correct": row["correct"][fixed]})
    def comparison(name, reference):
        paired = [{"adaptive_action": e[name + "_action"],
                   "fixed_action": e[reference + "_action"],
                   "adaptive_ce": e[name + "_ce"],
                   "fixed_ce": e[reference + "_ce"],
                   "adaptive_correct": e[name + "_correct"],
                   "fixed_correct": e[reference + "_correct"]}
                  for e in entries]
        return summarize(paired), paired
    fixed_summary, fixed_pairs = comparison("additive", "fixed")
    lookup_summary, lookup_pairs = comparison("additive", "lookup")
    return {"additive_vs_fixed": fixed_summary,
            "additive_vs_lookup": lookup_summary,
            "additive_vs_fixed_pairs": fixed_pairs,
            "additive_vs_lookup_pairs": lookup_pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prepared", "planned", "merged", "responder", "cache",
                "static-order", "out"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lookup-smoothing", type=float, default=20.0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.lookup_smoothing < 0:
        raise ValueError("invalid batch size or lookup smoothing")
    out = Path(args.out)
    if out.exists():
        raise ValueError("refuse to overwrite existing report")
    prepared, planned, merged, responder, cache = (Path(getattr(args, key))
        for key in ("prepared", "planned", "merged", "responder", "cache"))
    merge_receipt = read_json(merged / "receipt.json")
    sources = sorted(merge_receipt["sources"], key=lambda s: s["outer_fold"])
    if [s["outer_fold"] for s in sources] != list(range(len(sources))) or len(sources) < 2:
        raise ValueError("complete ordered outer folds required")
    schema, validation_raw, val_manifest = load_crossfit_cache(
        cache, prepared, planned, responder)
    if (val_manifest["split"] != "validation"
            or val_manifest["response_source"] != "automatic_model"):
        raise ValueError("requires automatic validation responses")
    static = read_json(args.static_order)
    if (static.get("schema_signature") != schema_signature(schema)
            or static.get("fit_scope") != "ALL_OBSERVED_OUTER_OOF_TRAIN_TARGETS"
            or static.get("evaluation_holdout_labels_used") is not False
            or sorted(static["order"]) != list(range(schema.num_groups))):
        raise ValueError("training-only static first action required")
    first = static["order"][0]
    candidates = tuple(g for g in range(schema.num_groups) if g != first)
    atom_ids = schema.groups[first].atoms
    counts = tuple(schema.num_categories[a] for a in atom_ids)
    train_rows, folds, bindings = [], [], []
    for source in sources:
        outer = Path(source["directory"])
        fold_schema, raw, manifest = load_crossfit_cache(
            outer / "outer/cache", prepared, planned, outer / "outer/responder")
        if (fold_schema != schema or not raw
                or manifest["response_source"] != "automatic_model"
                or manifest["crossfit"]["outer_fold"] != source["outer_fold"]):
            raise ValueError("outer cache mismatch")
        head, head_report = load_crossfit_head(outer / "head", schema, device="cpu")
        groups = sorted({r["group_id"] for r in raw})
        if (set(groups) & set(head_report["fit_group_ids"])
                or head_report["head_artifact_id"] != read_json(outer / "receipt.json")["head_artifact_id"]):
            raise ValueError("outer head trained on policy targets")
        validate_target_exclusion(head_report["provenance"],
            artifact_ids=[head_report["head_artifact_id"]], target_group_ids=groups)
        scored = score_rows(raw, head, schema, first, candidates, args.batch_size)
        train_rows.extend(scored)
        folds.extend([source["outer_fold"]] * len(scored))
        bindings.append({"outer_fold": source["outer_fold"], "rows": len(scored),
            "cache_manifest_sha256": file_hash(outer / "outer/cache/manifest.json"),
            "head_receipt_sha256": file_hash(outer / "head/receipt.json")})
    if len({r["group_id"] for r in train_rows}) != len(train_rows):
        raise ValueError("training groups overlap")
    final_head, final_head_report = load_crossfit_head(merged / "head", schema, device="cpu")
    val_rows = score_rows(validation_raw, final_head, schema, first,
                          candidates, args.batch_size)
    x = feature_matrix([r["outcome"] for r in train_rows], counts)
    y = loss_matrix(train_rows, candidates)
    alpha, cv_scores = choose_alpha(x, y, folds)
    weights = fit_ridge(x, y, alpha)
    val_x = feature_matrix([r["outcome"] for r in val_rows], counts)
    fixed, lookup, _, _ = choose_second(train_rows, candidates,
                                         smoothing=args.lookup_smoothing)
    comparisons = evaluate(val_rows, candidates, val_x @ weights, fixed, lookup)
    fixed_pairs = comparisons.pop("additive_vs_fixed_pairs")
    lookup_pairs = comparisons.pop("additive_vs_lookup_pairs")
    report = {"format": "cbmjev-oof-k2-additive-risk-validation-v1",
        "evidence_status": "TRAIN_OOF_MODEL_SELECTION_VALIDATION_EVALUATION_NOT_LOCKED_TEST",
        "seed": args.seed, "first_group": first, "first_group_id": schema.groups[first].id,
        "fixed_second_group": fixed, "training_rows": len(train_rows),
        "validation_rows": len(val_rows), "candidate_groups": list(candidates),
        "feature_width": x.shape[1], "feature_type": "first_answer_additive_onehot",
        "selected_alpha": alpha, "oof_heldout_decision_ce_by_alpha": cv_scores,
        "lookup_smoothing": args.lookup_smoothing,
        "validation_labels_used_for_fit_or_tuning": False,
        "outer_fold_target_labels_used_for_fit_and_tuning": True,
        "outer_source_bindings": bindings,
        "merged_receipt_sha256": file_hash(merged / "receipt.json"),
        "validation_cache_manifest_sha256": file_hash(cache / "manifest.json"),
        "final_head_receipt_sha256": file_hash(merged / "head/receipt.json"),
        "final_head_artifact_id": final_head_report["head_artifact_id"],
        "static_order_sha256": file_hash(args.static_order),
        "analysis_script_sha256": file_hash(__file__),
        "comparisons": comparisons,
        "paired_bootstrap_95": {
            "additive_vs_fixed": paired_bootstrap(fixed_pairs, seed=args.seed),
            "additive_vs_lookup": paired_bootstrap(lookup_pairs, seed=args.seed + 10000)},
        "caveats": ["Cross-fit training heads differ from the final validation head.",
                    "One adaptive decision at K2 only; no STOP or JEV representation.",
                    "Validation reused during model development; not independent test evidence."]}
    report["report_hash"] = stable_hash(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "alpha": alpha,
        "comparisons": comparisons}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

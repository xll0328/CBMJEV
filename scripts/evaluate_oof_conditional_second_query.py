#!/usr/bin/env python3
"""Train a two-query conditional policy on outer-OOF rows, evaluate validation.

The first query comes from a training-only static order. Each OOF row is
scored with its own excluded-group fold task head. The final validation rows
are scored with the final OOF-trained task head. This is still development
evidence, not a locked test or a universal adaptive-policy result.
"""
import argparse
import json
import math
from pathlib import Path

from cbmjev.contracts import stable_hash
from cbmjev.crossfit_artifacts import load_crossfit_head
from cbmjev.crossfit_cache import load_crossfit_cache
from cbmjev.io import file_hash, read_json
from cbmjev.learning import mask_answers, schema_signature
from cbmjev.provenance import validate_target_exclusion
from scripts.diagnose_conditional_second_query import (
    choose_second, outcome_key, outcome_shuffle_control, paired_bootstrap,
    summarize,
)


def score_rows(raw_rows, head, schema, first, candidates, batch_size):
    """Counterfactual labels are private training/evaluator data, not state."""
    rows, pending = [], []
    for source in raw_rows:
        answers, y = tuple(source["z"]), source["y"]
        schema.validate_state(answers, complete=True)
        if type(y) is not int or not 0 <= y < schema.num_classes:
            raise ValueError("invalid target")
        row = {"group_id": source["group_id"],
               "outcome": outcome_key(answers, schema, first),
               "y": y, "ce": {}, "correct": {}}
        index = len(rows)
        rows.append(row)
        for second in candidates:
            mask = tuple(g in (first, second) for g in range(schema.num_groups))
            pending.append((index, second, mask_answers(answers, mask, schema)))
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        probs = head.probabilities_many([entry[2] for entry in batch])
        for (index, second, _), p in zip(batch, probs):
            y = rows[index]["y"]
            rows[index]["ce"][second] = -math.log(max(p[y], 1e-12))
            rows[index]["correct"][second] = int(max(range(len(p)), key=p.__getitem__) == y)
    return rows


def evaluate(train, validation, candidates, *, smoothing, seed):
    fixed, policy, global_loss, counts = choose_second(
        train, candidates, smoothing=smoothing)
    entries = []
    for row in validation:
        action = policy.get(row["outcome"], fixed)
        entries.append({"adaptive_action": action, "fixed_action": fixed,
                        "adaptive_ce": row["ce"][action],
                        "fixed_ce": row["ce"][fixed],
                        "adaptive_correct": row["correct"][action],
                        "fixed_correct": row["correct"][fixed]})
    return {"aggregate": summarize(entries),
            "paired_bootstrap_95": paired_bootstrap(entries, seed=seed),
            "outcome_shuffle_control": outcome_shuffle_control(
                [(validation, policy, fixed)], seed=seed),
            "fixed_second_group": fixed,
            "training_outcome_counts": counts,
            "training_outcomes_with_at_most_3_cases": sum(v <= 3 for v in counts.values()),
            "unseen_validation_outcome_count": sum(row["outcome"] not in policy
                                                  for row in validation),
            "learned_outcome_actions": {str(k): v for k, v in policy.items()},
            "global_train_ce_by_group": global_loss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "planned", "merged", "responder", "cache",
                 "static-order", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--smoothing", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.smoothing < 0 or not math.isfinite(args.smoothing) or args.batch_size < 1:
        raise ValueError("invalid smoothing or batch size")
    out = Path(args.out)
    if out.exists():
        raise ValueError("refuse to overwrite report")
    prepared, planned, merged, responder, cache = (Path(getattr(args, name))
        for name in ("prepared", "planned", "merged", "responder", "cache"))
    receipt = read_json(merged / "receipt.json")
    source_entries = sorted(receipt["sources"], key=lambda x: x["outer_fold"])
    if len(source_entries) < 2 or [x["outer_fold"] for x in source_entries] != list(range(len(source_entries))):
        raise ValueError("requires complete ordered outer folds")
    static = read_json(args.static_order)
    final_schema, final_raw, final_manifest = load_crossfit_cache(
        cache, prepared, planned, responder)
    if (final_manifest["split"] != "validation"
            or final_manifest["response_source"] != "automatic_model"):
        raise ValueError("requires automatic validation responses")
    if (static.get("schema_signature") != schema_signature(final_schema)
            or static.get("evaluation_holdout_labels_used") is not False
            or static.get("fit_scope") != "ALL_OBSERVED_OUTER_OOF_TRAIN_TARGETS"
            or sorted(static["order"]) != list(range(final_schema.num_groups))):
        raise ValueError("requires training-only schema-bound static order")
    first = static["order"][0]
    candidates = tuple(g for g in range(final_schema.num_groups) if g != first)
    training, bindings = [], []
    for source in source_entries:
        outer = Path(source["directory"])
        fold_schema, raw, manifest = load_crossfit_cache(
            outer / "outer/cache", prepared, planned, outer / "outer/responder")
        if (fold_schema != final_schema or not raw
                or manifest["response_source"] != "automatic_model"
                or manifest["crossfit"]["outer_fold"] != source["outer_fold"]):
            raise ValueError("outer fold cache mismatch")
        head, head_report = load_crossfit_head(outer / "head", fold_schema, device="cpu")
        groups = {row["group_id"] for row in raw}
        if (groups & set(head_report["fit_group_ids"])
                or head_report["head_artifact_id"] != read_json(outer / "receipt.json")["head_artifact_id"]):
            raise ValueError("outer fold task head saw policy training cases")
        validate_target_exclusion(head_report["provenance"],
                                  artifact_ids=[head_report["head_artifact_id"]],
                                  target_group_ids=sorted(groups))
        training.extend(score_rows(raw, head, fold_schema, first,
                                   candidates, args.batch_size))
        bindings.append({"outer_fold": source["outer_fold"],
                         "cache_manifest_sha256": file_hash(outer / "outer/cache/manifest.json"),
                         "head_receipt_sha256": file_hash(outer / "head/receipt.json"),
                         "head_artifact_id": head_report["head_artifact_id"],
                         "rows": len(raw)})
    if len({row["group_id"] for row in training}) != len(training):
        raise ValueError("outer training groups overlap")
    final_head, final_head_report = load_crossfit_head(
        merged / "head", final_schema, device="cpu")
    validation = score_rows(final_raw, final_head, final_schema,
                            first, candidates, args.batch_size)
    result = evaluate(training, validation, candidates,
                      smoothing=args.smoothing, seed=args.seed)
    report = {"format": "cbmjev-oof-conditional-second-query-validation-v1",
              "evidence_status": "TRAIN_OOF_FIT_VALIDATION_EVALUATION_NOT_LOCKED_TEST",
              "first_group": first, "first_group_id": final_schema.groups[first].id,
              "seed": args.seed, "smoothing": args.smoothing,
              "selection_loss": "unweighted_cross_entropy",
              "training_rows": len(training), "validation_rows": len(validation),
              "source_bindings": bindings,
              "merged_receipt_sha256": file_hash(merged / "receipt.json"),
              "validation_cache_manifest_sha256": file_hash(cache / "manifest.json"),
              "final_head_receipt_sha256": file_hash(merged / "head/receipt.json"),
              "final_head_artifact_id": final_head_report["head_artifact_id"],
              "static_order_sha256": file_hash(args.static_order),
              "validation_labels_used_for_fit": False,
              "outer_fold_target_labels_used_for_fit": True,
              "analysis_script_sha256": file_hash(__file__),
              "caveats": ["Training fold heads differ from the final validation head.",
                          "This tests only one adaptive step at a two-group budget.",
                          "Validation has been used for earlier model development; not independent test evidence."],
              **result}
    report["report_hash"] = stable_hash(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "training_rows": len(training),
                      "validation_rows": len(validation),
                      "aggregate": report["aggregate"]}, sort_keys=True))


if __name__ == "__main__":
    main()

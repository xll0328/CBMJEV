#!/usr/bin/env python3
"""Training-OOF-fitted class-conditional last query after fixed K15 prefix.

This is a simple non-JEV adaptive baseline at exact K16. The policy sees
only the current task head's top-class prediction from acquired groups.
It cannot inspect unqueried concepts, image identifiers or target labels.
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
    paired_bootstrap, summarize)
from scripts.evaluate_oof_conditional_second_query import evaluate


def score_rows(raw_rows, head, schema, prefix, candidates, batch_size):
    if (len(prefix) != 15 or len(set(prefix)) != 15 or
            any(g in prefix for g in candidates) or
            sorted((*prefix, *candidates)) != list(range(schema.num_groups))):
        raise ValueError("K15 prefix and remaining candidates must partition groups")
    prefix_set = set(prefix)
    prefix_mask = tuple(g in prefix_set for g in range(schema.num_groups))
    rows = []
    for source in raw_rows:
        answers, y = tuple(source["z"]), source["y"]
        schema.validate_state(answers, complete=True)
        if type(y) is not int or not 0 <= y < schema.num_classes:
            raise ValueError("invalid task target")
        rows.append({"group_id": source["group_id"],
                     "answers": answers, "y": y, "outcome": None,
                     "ce": {}, "correct": {}})
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        states = [mask_answers(row["answers"], prefix_mask, schema) for row in batch]
        probs = head.probabilities_many(states)
        for row, p in zip(batch, probs):
            row["outcome"] = max(range(len(p)), key=p.__getitem__)
    # Gold labels and counterfactual automatic answers are offline targets only.
    # At inference the table only sees the prefix-derived outcome above.
    for candidate in candidates:
        mask = tuple(g in prefix_set or g == candidate
                     for g in range(schema.num_groups))
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            states = [mask_answers(row["answers"], mask, schema) for row in batch]
            probs = head.probabilities_many(states)
            for row, p in zip(batch, probs):
                y = row["y"]
                row["ce"][candidate] = -math.log(max(p[y], 1e-12))
                row["correct"][candidate] = int(
                    max(range(len(p)), key=p.__getitem__) == y)
    for row in rows:
        del row["answers"]
    return rows


def static_comparisons(validation, policy, fixed, static_action, *, seed):
    entries = []
    for row in validation:
        chosen = policy.get(row["outcome"], fixed)
        entries.append({"adaptive_action": chosen, "fixed_action": static_action,
                        "adaptive_ce": row["ce"][chosen],
                        "fixed_ce": row["ce"][static_action],
                        "adaptive_correct": row["correct"][chosen],
                        "fixed_correct": row["correct"][static_action]})
    return {"conditional_vs_original_static": summarize(entries),
            "conditional_vs_original_static_paired_bootstrap_95":
                paired_bootstrap(entries, seed=seed)}


def canonicalize_result(result):
    """Make the reused K2 evaluator's integer-keyed risk table JSON-stable."""
    return {**result, "global_train_ce_by_group": {
        str(group): value for group, value in
        result["global_train_ce_by_group"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "planned", "merged", "responder", "cache",
                 "static-order", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoothing", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if (args.batch_size < 1 or args.smoothing < 0 or
            not math.isfinite(args.smoothing)):
        raise ValueError("invalid batch size or smoothing")
    out = Path(args.out)
    if out.exists():
        raise ValueError("refuse to overwrite report")
    prepared, planned, merged, responder, cache = (
        Path(getattr(args, name)) for name in
        ("prepared", "planned", "merged", "responder", "cache"))
    receipt = read_json(merged / "receipt.json")
    sources = sorted(receipt["sources"], key=lambda source: source["outer_fold"])
    if (len(sources) < 2 or
            [source["outer_fold"] for source in sources] != list(range(len(sources)))):
        raise ValueError("requires complete ordered outer folds")
    schema, val_raw, val_manifest = load_crossfit_cache(
        cache, prepared, planned, responder)
    if (val_manifest["split"] != "validation" or
            val_manifest["response_source"] != "automatic_model"):
        raise ValueError("requires automatic final validation responses")
    static = read_json(args.static_order)
    if (static.get("schema_signature") != schema_signature(schema) or
            static.get("evaluation_holdout_labels_used") is not False or
            static.get("fit_scope") != "ALL_OBSERVED_OUTER_OOF_TRAIN_TARGETS" or
            sorted(static["order"]) != list(range(schema.num_groups)) or
            schema.num_groups <= 15):
        raise ValueError("requires training-only fixed order and more than 15 groups")
    prefix = tuple(static["order"][:15])
    static_action = static["order"][15]
    candidates = tuple(g for g in range(schema.num_groups) if g not in prefix)
    training, bindings = [], []
    for source in sources:
        outer = Path(source["directory"])
        fold_schema, raw, manifest = load_crossfit_cache(
            outer / "outer/cache", prepared, planned, outer / "outer/responder")
        if (fold_schema != schema or not raw or
                manifest["response_source"] != "automatic_model" or
                manifest["crossfit"]["outer_fold"] != source["outer_fold"]):
            raise ValueError("outer fold automatic cache mismatch")
        head, head_report = load_crossfit_head(outer / "head", schema, device="cpu")
        groups = {row["group_id"] for row in raw}
        if (groups & set(head_report["fit_group_ids"]) or
                head_report["head_artifact_id"] !=
                read_json(outer / "receipt.json")["head_artifact_id"]):
            raise ValueError("outer fold task head saw policy training groups")
        validate_target_exclusion(head_report["provenance"],
            artifact_ids=[head_report["head_artifact_id"]],
            target_group_ids=sorted(groups))
        training.extend(score_rows(raw, head, schema, prefix,
                                   candidates, args.batch_size))
        bindings.append({"outer_fold": source["outer_fold"],
                         "rows": len(raw),
                         "cache_manifest_sha256": file_hash(outer / "outer/cache/manifest.json"),
                         "head_receipt_sha256": file_hash(outer / "head/receipt.json"),
                         "head_artifact_id": head_report["head_artifact_id"]})
    if len({row["group_id"] for row in training}) != len(training):
        raise ValueError("outer training groups overlap")
    final_head, final_report = load_crossfit_head(merged / "head", schema,
                                                  device="cpu")
    validation = score_rows(val_raw, final_head, schema, prefix,
                            candidates, args.batch_size)
    result = canonicalize_result(evaluate(training, validation, candidates,
                                         smoothing=args.smoothing, seed=args.seed))
    fixed = result["fixed_second_group"]
    policy = {int(key): value for key, value in
              result["learned_outcome_actions"].items()}
    extra = static_comparisons(validation, policy, fixed, static_action,
                               seed=args.seed + 10000)
    counts = result["training_outcome_counts"]
    report = {"format": "cbmjev-oof-k16-conditional-last-query-validation-v1",
        "evidence_status": "TRAIN_OOF_FIT_VALIDATION_EVALUATION_NOT_LOCKED_TEST",
        "regime": "15_fixed_training_oof_groups_plus_1_topclass_conditioned_group",
        "seed": args.seed, "split": "validation", "test_evaluated": False,
        "selection_loss": "unweighted_cross_entropy", "smoothing": args.smoothing,
        "prefix": list(prefix),
        "prefix_group_ids": [schema.groups[g].id for g in prefix],
        "original_static_sixteenth_group": static_action,
        "original_static_sixteenth_group_id": schema.groups[static_action].id,
        "candidates": list(candidates), "training_rows": len(training),
        "validation_rows": len(validation),
        "outcome_type": "argmax_task_class_from_only_acquired_15_groups",
        "training_outcome_classes": len(counts),
        "training_outcomes_with_at_most_3_cases":
            sum(count <= 3 for count in counts.values()),
        "source_bindings": bindings,
        "validation_labels_used_for_fit": False,
        "outer_fold_target_labels_used_for_fit": True,
        "bindings": {"merged_receipt_sha256": file_hash(merged / "receipt.json"),
                     "validation_cache_manifest_sha256": file_hash(cache / "manifest.json"),
                     "final_head_receipt_sha256": file_hash(merged / "head/receipt.json"),
                     "final_head_artifact_id": final_report["head_artifact_id"],
                     "static_order_sha256": file_hash(args.static_order),
                     "analysis_script_sha256": file_hash(__file__)},
        "caveats": ["Only the final query is adaptive; this is not a full K16 adaptive policy.",
                    "Fold heads used for OOF targets differ from the final validation head.",
                    "Predicted top class is a lossy summary of 15 observed groups.",
                    "Validation has informed earlier research design; intervals are descriptive."],
        **result, **extra}
    report["report_hash"] = stable_hash(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({"out": str(out), "seed": args.seed,
                      "conditional_vs_oof_fixed": result["aggregate"],
                      "conditional_vs_original_static":
                          extra["conditional_vs_original_static"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()

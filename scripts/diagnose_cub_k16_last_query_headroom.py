#!/usr/bin/env python3
"""Hindsight-only K16 last-query headroom for a frozen CUB task head.

This diagnostic deliberately uses validation labels to pick the best final
action per image. It is NOT a legal policy, attainable bound, or paper result.
"""

import argparse
import math
from collections import Counter
from pathlib import Path
from statistics import mean

from cbmjev.contracts import stable_hash
from cbmjev.crossfit_artifacts import load_crossfit_head
from cbmjev.crossfit_cache import load_crossfit_cache
from cbmjev.io import file_hash, read_json, write_json
from cbmjev.learning import schema_signature
from scripts.evaluate_oof_k16_conditional_last_query import score_rows


def summarize_headroom(rows, candidates, fixed):
    if not rows or fixed not in candidates or len(set(candidates)) != len(candidates):
        raise ValueError("invalid rows/candidates/fixed action")
    if any(set(row["ce"]) != set(candidates) or
           set(row["correct"]) != set(candidates) or
           any(not math.isfinite(row["ce"][a]) or row["ce"][a] < 0 or
               row["correct"][a] not in (0, 1) for a in candidates)
           for row in rows):
        raise ValueError("invalid per-action scores")
    fixed_ce = mean(row["ce"][fixed] for row in rows)
    fixed_acc = mean(row["correct"][fixed] for row in rows)
    winners = [min(candidates, key=lambda a: (row["ce"][a], a)) for row in rows]
    hindsight_ce = mean(row["ce"][a] for row, a in zip(rows, winners))
    ce_gain = [row["ce"][fixed] - row["ce"][a]
               for row, a in zip(rows, winners)]
    any_correct = mean(max(row["correct"].values()) for row in rows)
    chosen_acc = mean(row["correct"][a] for row, a in zip(rows, winners))
    return {"samples": len(rows), "candidates": len(candidates),
            "fixed_action": fixed, "fixed_ce": fixed_ce,
            "fixed_accuracy": fixed_acc,
            "label_clairvoyant_min_ce": hindsight_ce,
            "label_clairvoyant_ce_selected_accuracy": chosen_acc,
            "label_clairvoyant_any_correct_accuracy": any_correct,
            "hindsight_ce_headroom": fixed_ce - hindsight_ce,
            "hindsight_any_correct_headroom": any_correct - fixed_acc,
            "fraction_fixed_is_ce_best": mean(a == fixed for a in winners),
            "fraction_at_least_0p1_ce_improvement": mean(g >= .1 for g in ce_gain),
            "ce_winner_counts": {str(action): count for action, count in
                                 sorted(Counter(winners).items())}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "planned", "merged", "responder", "cache",
                 "static-order", "reference", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.seed < 0:
        raise ValueError("invalid batch size or seed")
    out = Path(args.out)
    if out.exists():
        raise ValueError("refuse to overwrite report")
    prepared, planned, merged, responder, cache, static_path = map(Path,
        (args.prepared, args.planned, args.merged, args.responder, args.cache,
         args.static_order))
    schema, val_raw, manifest = load_crossfit_cache(
        cache, prepared, planned, responder)
    if manifest["split"] != "validation" or manifest["response_source"] != "automatic_model":
        raise ValueError("expected automatic validation response cache")
    static = read_json(static_path)
    if (static.get("schema_signature") != schema_signature(schema) or
            static.get("fit_scope") != "ALL_OBSERVED_OUTER_OOF_TRAIN_TARGETS" or
            static.get("evaluation_holdout_labels_used") is not False or
            sorted(static["order"]) != list(range(schema.num_groups))):
        raise ValueError("invalid OOF-fitted static order")
    reference = read_json(args.reference)
    unsigned = dict(reference)
    if unsigned.pop("report_hash", None) != stable_hash(unsigned):
        raise ValueError("reference report hash mismatch")
    binding = reference["bindings"]
    if (reference.get("seed") != args.seed or
            reference.get("format") != "cbmjev-oof-k16-conditional-last-query-validation-v1" or
            reference.get("validation_labels_used_for_fit") is not False or
            binding["static_order_sha256"] != file_hash(static_path) or
            binding["validation_cache_manifest_sha256"] != file_hash(cache / "manifest.json") or
            binding["merged_receipt_sha256"] != file_hash(merged / "receipt.json")):
        raise ValueError("reference or frozen source mismatch")
    head, head_report = load_crossfit_head(merged / "head", schema, device="cpu")
    if (binding["final_head_receipt_sha256"] != file_hash(merged / "head/receipt.json") or
            binding["final_head_artifact_id"] != head_report["head_artifact_id"]):
        raise ValueError("frozen final head mismatch")
    prefix = tuple(static["order"][:15])
    candidates = tuple(g for g in range(schema.num_groups) if g not in prefix)
    fixed = static["order"][15]
    rows = score_rows(val_raw, head, schema, prefix, candidates, args.batch_size)
    summary = summarize_headroom(rows, candidates, fixed)
    aggregate = reference["aggregate"]
    if (len(rows) != reference["validation_rows"] or
            abs(summary["fixed_ce"] - aggregate["fixed_ce"]) > 1e-8 or
            abs(summary["fixed_accuracy"] - aggregate["fixed_accuracy"]) > 1e-8):
        raise ValueError("recomputed fixed baseline disagrees with reference")
    report = {"format": "cbmjev-k16-last-query-hindsight-headroom-v1",
              "evidence_status": "LABEL_CLAIRVOYANT_VALIDATION_DIAGNOSTIC_NOT_POLICY",
              "seed": args.seed, "split": "validation", "test_evaluated": False,
              "prefix": list(prefix), "summary": summary,
              "bindings": {"reference_sha256": file_hash(args.reference),
                           "static_order_sha256": file_hash(static_path),
                           "cache_manifest_sha256": file_hash(cache / "manifest.json"),
                           "head_receipt_sha256": file_hash(merged / "head/receipt.json"),
                           "analysis_script_sha256": file_hash(__file__)},
              "interpretation_limit": "Per-image best action uses the true validation label and all unqueried automatic answers; it is an optimistic hindsight diagnostic for one final query only, not a legal policy, attainable Bayes bound, JEV evidence, or full K16 sequential headroom."}
    report["report_hash"] = stable_hash(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    print(summary, flush=True)


if __name__ == "__main__":
    main()

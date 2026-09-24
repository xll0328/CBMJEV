#!/usr/bin/env python3
"""Inspect paired NLL/confidence tails for saved CUB complete-case heads.

No parameters or order are selected from validation. Gold annotations are
used only on the complete-case subset; this is not an oracle bound.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

from cbmjev.contracts import stable_hash
from cbmjev.crossfit_cache import load_crossfit_cache
from cbmjev.io import file_hash, read_json
from cbmjev.learning import mask_answers, schema_signature
from scripts.evaluate_cub_gold_order_controls import load_matched_heads
from scripts.evaluate_cub_paired_gold_intervention import complete_gold_validation


def quantiles(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty values")
    return {key: ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]
            for key, fraction in (("p50", .5), ("p90", .9), ("p95", .95),
                                  ("p99", .99), ("max", 1.0))}


def score_budget(rows, gold, schema, heads, order, budget, batch_size):
    visible = set(order[:budget])
    mask = tuple(g in visible for g in range(schema.num_groups))
    selected = [row for row in rows if row["sample_id"] in gold]
    if len(selected) != len(gold):
        raise ValueError("gold/cache pairing mismatch")
    scores = {}
    for source in ("automatic", "gold"):
        head = heads[source]
        for offset in range(0, len(selected), batch_size):
            batch = selected[offset:offset + batch_size]
            states = [mask_answers(row["z"] if source == "automatic"
                                   else gold[row["sample_id"]], mask, schema)
                      for row in batch]
            probs = head.probabilities_many(states)
            for row, p in zip(batch, probs):
                y = row["y"]
                confidence = max(p)
                prediction = max(range(len(p)), key=p.__getitem__)
                scores[(row["sample_id"], source)] = {
                    "nll": -math.log(max(p[y], 1e-12)),
                    "correct": int(prediction == y),
                    "confidence": confidence,
                    "brier": sum(value * value for value in p) - 2 * p[y] + 1}
    entries = [{"sample_id": row["sample_id"],
                "automatic": scores[(row["sample_id"], "automatic")],
                "gold": scores[(row["sample_id"], "gold")]} for row in selected]
    return entries


def summarize(entries):
    if not entries:
        raise ValueError("empty paired scores")
    n = len(entries)
    metrics = {}
    for source in ("automatic", "gold"):
        rows = [row[source] for row in entries]
        errors = [row for row in rows if not row["correct"]]
        nlls = [row["nll"] for row in rows]
        metrics[source] = {"accuracy": sum(row["correct"] for row in rows) / n,
                           "ce": statistics.mean(nlls),
                           "brier": statistics.mean(row["brier"] for row in rows),
                           "mean_confidence": statistics.mean(row["confidence"] for row in rows),
                           "mean_confidence_on_errors":
                               statistics.mean(row["confidence"] for row in errors) if errors else None,
                           "errors": len(errors), "nll_quantiles": quantiles(nlls),
                           "top5_nll_fraction_of_total": sum(sorted(nlls, reverse=True)[:5]) / sum(nlls)}
    delta = [row["gold"]["nll"] - row["automatic"]["nll"] for row in entries]
    positive = sorted((value for value in delta if value > 0), reverse=True)
    negative = [value for value in delta if value < 0]
    paired = {"gold_minus_auto_mean_nll": statistics.mean(delta),
              "gold_lower_nll_cases": len(negative),
              "gold_higher_nll_cases": len(positive),
              "nll_delta_quantiles": quantiles(delta),
              "top5_positive_delta_fraction_of_positive_mass":
                  sum(positive[:5]) / sum(positive) if positive else 0.0,
              "top5_positive_delta_sum": sum(positive[:5]),
              "positive_delta_sum": sum(positive),
              "negative_delta_sum": sum(negative),
              "gold_correct_auto_wrong": sum(
                  row["gold"]["correct"] and not row["automatic"]["correct"]
                  for row in entries),
              "auto_correct_gold_wrong": sum(
                  row["automatic"]["correct"] and not row["gold"]["correct"]
                  for row in entries)}
    return {"n": n, "by_source": metrics, "paired": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "planned", "responder", "cache", "static-order",
                 "matched-heads", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() or args.batch_size < 1:
        raise ValueError("output exists or invalid batch size")
    prepared, planned, responder, cache = (
        Path(getattr(args, name)) for name in
        ("prepared", "planned", "responder", "cache"))
    schema, rows, manifest = load_crossfit_cache(cache, prepared, planned, responder)
    if (manifest["split"] != "validation" or
            manifest["response_source"] != "automatic_model"):
        raise ValueError("requires automatic final validation cache")
    gold = complete_gold_validation(prepared, schema, rows)
    heads, parent = load_matched_heads(args.matched_heads, schema)
    if (parent["seed"] != args.seed or
            parent["validation_sample_ids_hash"] != stable_hash(sorted(gold)) or
            parent["bindings"]["validation_cache_manifest_sha256"] != file_hash(cache / "manifest.json") or
            parent["bindings"]["prepared_samples_sha256"] != file_hash(prepared / "samples.jsonl")):
        raise ValueError("matched-head parent differs from validation inputs")
    static = read_json(args.static_order)
    if (static.get("schema_signature") != schema_signature(schema) or
            static.get("evaluation_holdout_labels_used") is not False or
            parent["bindings"]["static_order_sha256"] != file_hash(args.static_order)):
        raise ValueError("static order is not parent-bound")
    order = static["order"]
    if sorted(order) != list(range(schema.num_groups)):
        raise ValueError("invalid fixed order")
    budgets = (16, schema.num_groups)
    results = {str(k): summarize(score_budget(rows, gold, schema, heads, order,
                                              k, args.batch_size)) for k in budgets}
    for budget, result in results.items():
        parent_budget = parent["metrics"][budget]
        for source in ("automatic", "gold"):
            if (abs(result["by_source"][source]["accuracy"] -
                    parent_budget["correct"][source + "_mean"]) > 1e-8 or
                abs(result["by_source"][source]["ce"] -
                    parent_budget["ce"][source + "_mean"]) > 1e-8):
                raise ValueError("NLL replay differs from matched-head parent")
    report = {"format": "cbmjev-cub-gold-nll-tails-validation-v1",
        "evidence_status": "EXPLORATORY_COMPLETE_CASE_DESCRIPTIVE_NLL_TAILS",
        "seed": args.seed, "split": "validation", "test_evaluated": False,
        "budgets": results,
        "bindings": {"matched_parent_report_sha256": file_hash(Path(args.matched_heads) / "report.json"),
                     "validation_cache_manifest_sha256": file_hash(cache / "manifest.json"),
                     "static_order_sha256": file_hash(args.static_order),
                     "analysis_script_sha256": file_hash(__file__)},
        "caveats": ["Same 118 class-shifted complete CUB validation cases across budgets and seeds.",
                    "Tail and confidence summaries are descriptive; no validation-tuned calibration or decision rule.",
                    "This is neither a population oracle ceiling nor a JEV/controller comparison."]}
    report["report_hash"] = stable_hash(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "seed": args.seed,
                      "summary": {k: {"gold_minus_auto_ce": v["paired"]["gold_minus_auto_mean_nll"],
                                      "gold_correct_auto_wrong": v["paired"]["gold_correct_auto_wrong"],
                                      "auto_correct_gold_wrong": v["paired"]["auto_correct_gold_wrong"],
                                      "top5_positive_delta_fraction":
                                          v["paired"]["top5_positive_delta_fraction_of_positive_mass"]}
                                  for k, v in results.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()

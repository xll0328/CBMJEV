#!/usr/bin/env python3
"""Test whether CUB complete-case auto/gold head differences depend on order.

All candidate orders are defined without validation labels. This is a
development-validation mechanism diagnostic, not a policy search or oracle.
"""
import argparse
import json
from pathlib import Path
import random
import statistics

import torch

from cbmjev.contracts import stable_hash
from cbmjev.crossfit_cache import load_crossfit_cache
from cbmjev.io import file_hash, read_json
from cbmjev.learning import MaskedHead, schema_signature
from scripts.evaluate_cub_matched_gold_heads import evaluate_pair
from scripts.evaluate_cub_paired_gold_intervention import complete_gold_validation


def load_matched_heads(directory, schema):
    directory = Path(directory)
    report = read_json(directory / "report.json")
    unsigned = dict(report)
    if (unsigned.pop("report_hash", None) != stable_hash(unsigned) or
            report.get("format") != "cbmjev-cub-matched-gold-heads-validation-v1" or
            report.get("test_evaluated") is not False):
        raise ValueError("matched-head report integrity/format mismatch")
    heads = {}
    for source in ("automatic", "gold"):
        path = directory / (source + "_head.pt")
        if file_hash(path) != report["checkpoint_sha256"][source]:
            raise ValueError("matched-head checkpoint hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload.get("format") != "cbmjev-cub-matched-complete-head-v1" or
                payload.get("schema_signature") != schema_signature(schema) or
                payload.get("config") != report["config"] or
                payload.get("source") != source):
            raise ValueError("matched-head checkpoint identity mismatch")
        head = MaskedHead(schema, {**payload["config"], "device": "cpu"})
        head.network.load_state_dict(payload["state_dict"], strict=True)
        head.network.eval().requires_grad_(False)
        heads[source] = head
    return heads, report


def order_controls(num_groups, fitted_order, *, random_seed, num_random):
    canonical = list(range(num_groups))
    if sorted(fitted_order) != canonical or num_random < 1:
        raise ValueError("invalid fitted order or random-order count")
    rng = random.Random(random_seed)
    controls = {"auto_oof": list(fitted_order),
                "schema": canonical,
                "reverse_schema": list(reversed(canonical))}
    for index in range(num_random):
        order = canonical.copy()
        rng.shuffle(order)
        controls[f"random_{index:02d}"] = order
    return controls


def summarize_random_orders(results):
    keys = sorted(key for key in results if key.startswith("random_"))
    output = {"num_predeclared_orders": len(keys)}
    for metric in ("correct", "ce"):
        values = [results[key]["16"][metric]["gold_minus_automatic"] for key in keys]
        output[metric] = {"mean": statistics.mean(values),
                          "median": statistics.median(values),
                          "minimum": min(values), "maximum": max(values),
                          "gold_better_count": sum(
                              value > 0 if metric == "correct" else value < 0
                              for value in values),
                          "gold_tie_count": sum(value == 0 for value in values)}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "planned", "responder", "cache", "static-order",
                 "matched-heads", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--random-seed", type=int, default=20260923)
    parser.add_argument("--num-random", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    out = Path(args.out)
    if (out.exists() or args.batch_size < 1 or args.cpu_threads < 1 or
            args.num_random < 1):
        raise ValueError("output exists or invalid settings")
    torch.set_num_threads(args.cpu_threads)
    prepared, planned, responder, cache = (
        Path(getattr(args, name)) for name in
        ("prepared", "planned", "responder", "cache"))
    schema, rows, manifest = load_crossfit_cache(cache, prepared, planned, responder)
    if (manifest["split"] != "validation" or
            manifest["response_source"] != "automatic_model"):
        raise ValueError("requires automatic final validation cache")
    gold = complete_gold_validation(prepared, schema, rows)
    heads, parent = load_matched_heads(args.matched_heads, schema)
    if (parent["seed"] != args.seed or parent["validation_complete_cases"] != len(gold) or
            parent["validation_sample_ids_hash"] != stable_hash(sorted(gold)) or
            parent["bindings"]["validation_cache_manifest_sha256"] != file_hash(cache / "manifest.json") or
            parent["bindings"]["prepared_samples_sha256"] != file_hash(prepared / "samples.jsonl")):
        raise ValueError("matched-head parent differs from validation inputs")
    static = read_json(args.static_order)
    if (static.get("schema_signature") != schema_signature(schema) or
            static.get("evaluation_holdout_labels_used") is not False or
            static.get("fit_scope") != "ALL_OBSERVED_OUTER_OOF_TRAIN_TARGETS" or
            parent["bindings"]["static_order_sha256"] != file_hash(args.static_order)):
        raise ValueError("training-only static order differs from matched-head run")
    orders = order_controls(schema.num_groups, static["order"],
                            random_seed=args.random_seed,
                            num_random=args.num_random)
    results = {}
    for name, order in orders.items():
        budgets = (2, 4, 8, 16, schema.num_groups) if name in (
            "auto_oof", "schema", "reverse_schema") else (16,)
        results[name] = evaluate_pair(rows, gold, schema, heads, order,
                                      budgets, args.batch_size, args.seed)
    # Reloading the saved heads must reproduce the parent fitted-order means.
    for budget, old in parent["metrics"].items():
        new = results["auto_oof"].get(budget)
        if new is None:
            continue
        for metric in ("correct", "ce"):
            for field in ("automatic_mean", "gold_mean", "gold_minus_automatic"):
                if abs(new[metric][field] - old[metric][field]) > 1e-8:
                    raise ValueError("saved-head replay differs from parent metric")
    report = {"format": "cbmjev-cub-gold-order-controls-validation-v1",
        "evidence_status": "EXPLORATORY_COMPLETE_CASE_ORDER_SENSITIVITY_NOT_POLICY_SEARCH",
        "split": "validation", "test_evaluated": False, "seed": args.seed,
        "validation_cases": len(gold), "random_order_seed": args.random_seed,
        "orders_predeclared_without_validation_labels": True,
        "orders": orders, "results": results,
        "random_K16_summary": summarize_random_orders(results),
        "bindings": {"matched_parent_report_sha256": file_hash(Path(args.matched_heads) / "report.json"),
                     "validation_cache_manifest_sha256": file_hash(cache / "manifest.json"),
                     "static_order_sha256": file_hash(args.static_order),
                     "analysis_script_sha256": file_hash(__file__)},
        "caveats": ["All comparisons reuse the same 118 class-shifted CUB validation cases.",
                    "Random orders are controls over masks, not independent image replications.",
                    "No order is chosen using validation performance or promoted to a policy.",
                    "This is head/response sensitivity, not a JEV or adaptive-controller result."]}
    report["report_hash"] = stable_hash(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "seed": args.seed,
                      "K16": {key: {metric: value["16"][metric]["gold_minus_automatic"]
                                       for metric in ("correct", "ce")}
                              for key, value in results.items()
                              if key in ("auto_oof", "schema", "reverse_schema")},
                      "random_K16_summary": report["random_K16_summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()

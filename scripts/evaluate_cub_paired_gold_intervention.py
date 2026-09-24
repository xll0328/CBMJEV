#!/usr/bin/env python3
"""Paired CUB validation sensitivity to replacing automatic answers by gold.

This keeps the OOF-trained task head, cases, and fixed acquisition order frozen.
It is NOT an oracle upper bound: the head was trained on automatic responses,
and fully observed gold is available for only a selected subset of CUB.
"""
import argparse
import json
import math
from pathlib import Path
import random

from cbmjev.contracts import stable_hash
from cbmjev.crossfit_artifacts import load_crossfit_head
from cbmjev.crossfit_cache import load_crossfit_cache
from cbmjev.io import file_hash, read_json
from cbmjev.learning import mask_answers, schema_signature


def complete_gold_validation(prepared, schema, automatic_rows):
    """Select by annotation availability, never by target or prediction."""
    automatic = {row["sample_id"]: row for row in automatic_rows}
    if len(automatic) != len(automatic_rows):
        raise ValueError("duplicate automatic validation sample ID")
    seen, gold = set(), {}
    with (Path(prepared) / "samples.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != "validation":
                continue
            sid = row["sample_id"]
            if sid in seen:
                raise ValueError("duplicate prepared validation sample ID")
            seen.add(sid)
            source = automatic.get(sid)
            if source is None or row["group_id"] != source["group_id"]:
                raise ValueError("prepared/cache validation membership mismatch")
            labels = row["concepts"]
            if len(labels) != schema.num_atoms:
                raise ValueError("concept width mismatch")
            values = []
            complete = True
            for label, concept in zip(labels, schema.concepts):
                if label["concept_id"] != concept.id:
                    raise ValueError("concept order mismatch")
                if label["annotation_status"] == "OBSERVED":
                    value = label["value"]
                    if type(value) is not int or not 0 <= value < len(concept.values):
                        raise ValueError("invalid observed gold value")
                    values.append(value)
                else:
                    if label["value"] is not None:
                        raise ValueError("missing gold status has a value")
                    complete = False
            target = row["target"]
            if (target["status"] != "OBSERVED" or
                    target["value"] != source["y"]):
                raise ValueError("prepared/cache validation target mismatch")
            if complete:
                schema.validate_state(tuple(values), complete=True)
                gold[sid] = tuple(values)
    if seen != set(automatic) or not gold:
        raise ValueError("validation coverage mismatch or no complete gold cases")
    return gold


def paired_scores(rows, gold, schema, head, order, budgets, batch_size):
    """Score both response sources on the same cases, masks, and frozen head."""
    if sorted(order) != list(range(schema.num_groups)):
        raise ValueError("invalid fixed order")
    if any(type(k) is not int or k < 0 or k > schema.num_groups for k in budgets):
        raise ValueError("invalid budget")
    selected = [row for row in rows if row["sample_id"] in gold]
    if len(selected) != len(gold):
        raise ValueError("gold selection differs from automatic cache")
    outcomes = {}
    for budget in budgets:
        mask = tuple(g in set(order[:budget]) for g in range(schema.num_groups))
        pending = []
        for row in selected:
            sid = row["sample_id"]
            schema.validate_state(tuple(row["z"]), complete=True)
            pending.extend(((sid, "automatic", row["y"],
                             mask_answers(row["z"], mask, schema)),
                            (sid, "gold", row["y"],
                             mask_answers(gold[sid], mask, schema))))
        scored = {}
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            probabilities = head.probabilities_many([entry[3] for entry in batch])
            for (sid, source, y, _), p in zip(batch, probabilities):
                scored[(sid, source)] = {
                    "ce": -math.log(max(p[y], 1e-12)),
                    "correct": int(max(range(len(p)), key=p.__getitem__) == y)}
        outcomes[budget] = [{"sample_id": row["sample_id"], "y": row["y"],
            "automatic": scored[(row["sample_id"], "automatic")],
            "gold": scored[(row["sample_id"], "gold")]}
            for row in selected]
    return outcomes


def summarize_paired(entries, *, seed, replicates=2000):
    if not entries or replicates < 1:
        raise ValueError("nonempty entries and positive replicates required")
    n = len(entries)
    scores = {}
    for metric in ("ce", "correct"):
        auto = [row["automatic"][metric] for row in entries]
        gold = [row["gold"][metric] for row in entries]
        differences = [g - a for a, g in zip(auto, gold)]
        rng = random.Random(seed + (0 if metric == "ce" else 1))
        draws = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n
                       for _ in range(replicates))
        scores[metric] = {"automatic_mean": sum(auto) / n,
                          "gold_mean": sum(gold) / n,
                          "gold_minus_automatic": sum(differences) / n,
                          "paired_image_bootstrap_95":
                              [draws[int(.025 * replicates)],
                               draws[min(replicates - 1, int(.975 * replicates))]]}
    return {"n": n, **scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "planned", "merged", "responder", "cache",
                 "static-order", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--budgets", type=int, nargs="+", default=(2, 4, 8, 16, 28))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1 or len(set(args.budgets)) != len(args.budgets):
        raise ValueError("invalid batch size or duplicate budgets")
    out = Path(args.out)
    if out.exists():
        raise ValueError("refuse to overwrite report")
    prepared, planned, merged, responder, cache = (
        Path(getattr(args, name)) for name in
        ("prepared", "planned", "merged", "responder", "cache"))
    schema, rows, manifest = load_crossfit_cache(cache, prepared, planned, responder)
    if (manifest["split"] != "validation" or
            manifest["response_source"] != "automatic_model"):
        raise ValueError("requires final automatic validation cache")
    head, head_report = load_crossfit_head(merged / "head", schema, device="cpu")
    static = read_json(args.static_order)
    if (static.get("schema_signature") != schema_signature(schema) or
            static.get("evaluation_holdout_labels_used") is not False or
            static.get("fit_scope") != "ALL_OBSERVED_OUTER_OOF_TRAIN_TARGETS"):
        raise ValueError("requires training-only schema-bound fixed order")
    gold = complete_gold_validation(prepared, schema, rows)
    outcomes = paired_scores(rows, gold, schema, head, static["order"],
                             args.budgets, args.batch_size)
    report = {"format": "cbmjev-cub-paired-gold-intervention-validation-v1",
        "evidence_status": "EXPLORATORY_COMPLETE_CASE_FROZEN_HEAD_NOT_ORACLE_BOUND",
        "split": "validation", "seed": args.seed,
        "total_validation_cases": len(rows), "complete_gold_cases": len(gold),
        "complete_gold_fraction": len(gold) / len(rows),
        "complete_gold_target_classes": len({row["y"] for row in rows
                                             if row["sample_id"] in gold}),
        "selected_sample_ids_hash": stable_hash(sorted(gold)),
        "selection_rule": "all concept atoms OBSERVED; independent of predictions and target values",
        "comparison": "same cases, masks, fixed OOF-trained automatic-response head",
        "budgets": {str(k): summarize_paired(outcomes[k], seed=args.seed + 1009 * k)
                    for k in args.budgets},
        "bindings": {"prepared_schema_sha256": file_hash(prepared / "schema.json"),
                     "prepared_samples_sha256": file_hash(prepared / "samples.jsonl"),
                     "cache_manifest_sha256": file_hash(cache / "manifest.json"),
                     "merged_head_receipt_sha256": file_hash(merged / "head/receipt.json"),
                     "head_artifact_id": head_report["head_artifact_id"],
                     "static_order_sha256": file_hash(args.static_order),
                     "analysis_script_sha256": file_hash(__file__)},
        "caveats": ["Complete cases are a class-shifted subset, not the full CUB validation population.",
                    "The head was trained on automatic responses; gold substitution induces distribution shift.",
                    "Intervals condition on the selected cases, frozen head and order, and validation development history.",
                    "This diagnostic does not train an adaptive policy or evaluate test data."]}
    report["report_hash"] = stable_hash(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "complete_gold_cases": len(gold),
                      "budgets": {k: {"ce_difference": v["ce"]["gold_minus_automatic"],
                                      "accuracy_difference": v["correct"]["gold_minus_automatic"]}
                                  for k, v in report["budgets"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline E1 diagnosis of frozen CEBaB head transitions on non-test rows.

The complete automatic response and task label are used only to construct and
score counterfactual replay states. This is not a deployable action selector.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, fresh_dir, write_json, write_jsonl
from cbmjev.pipeline import load_model_bundle


ALLOWED_SPLITS = ("validation", "policy_fit", "head_fit")
HIGH_CONFIDENCE = 0.8


def state_for_mask(schema, response, mask):
    state = list(schema.empty_state())
    for group in range(schema.num_groups):
        if mask & (1 << group):
            for atom in schema.groups[group].atoms:
                state[atom] = response[atom]
    return tuple(state)


def _counts(records, *, family_key="group_id"):
    families = {record[family_key] for record in records}
    counts = {key: sum(record["transition"] == key for record in records)
              for key in ("repair", "damage", "unchanged")}
    n = len(records)
    return {"transitions": n, "distinct_families": len(families),
            "counts": counts, "rates": {key: count / n if n else None
                                         for key, count in counts.items()},
            "mean_signed_01_gain": sum(r["signed_01_gain"] for r in records) / n if n else None,
            "mean_gold_probability_gain": sum(r["gold_probability_gain"] for r in records) / n if n else None,
            "mean_signed_ce_gain": sum(r["signed_ce_gain"] for r in records) / n if n else None}


def _confidence_distribution(records):
    values = sorted(record["confidence"] for record in records)
    if not values:
        return {"states": 0, "min": None, "p50": None, "p90": None,
                "p99": None, "max": None}
    def at(q):
        return values[int(q * (len(values) - 1))]
    return {"states": len(values), "min": values[0], "p50": at(0.5),
            "p90": at(0.9), "p99": at(0.99), "max": values[-1]}


def diagnose_rows(rows, schema, head, *, split="validation", limit=None):
    if split not in ALLOWED_SPLITS:
        raise ValueError("E1 diagnosis only accepts validation/policy_fit/head_fit, never test")
    if schema.dataset != "cebab" or schema.num_groups != 4:
        raise ValueError("E1 requires the four-group CEBaB schema")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError("selected split is empty")
    original_count = len(selected)
    if limit is not None:
        selected = selected[:limit]
    transitions = []
    state_records = []
    for row in selected:
        response = tuple(row["z"])
        schema.validate_state(response, complete=True)
        y = row["y"]
        if type(y) is not int or not 0 <= y < schema.num_classes:
            raise ValueError("invalid task label")
        masks = list(range(1 << schema.num_groups))
        states = [state_for_mask(schema, response, mask) for mask in masks]
        probabilities = head.probabilities_many(states)
        if len(probabilities) != len(states):
            raise ValueError("task head returned wrong batch size")
        scored = {}
        for mask, probs in zip(masks, probabilities):
            if (len(probs) != schema.num_classes or
                    any(not math.isfinite(p) or p < 0 or p > 1 for p in probs)):
                raise ValueError("invalid task-head probabilities")
            prediction = max(range(schema.num_classes), key=lambda c: probs[c])
            scored[mask] = (prediction, prediction == y, float(probs[prediction]), float(probs[y]))
        for mask in masks[:-1]:
            before_pred, before_correct, before_conf, before_gold = scored[mask]
            remaining = [group for group in range(4) if not mask & (1 << group)]
            repairable = any(not before_correct and scored[mask | (1 << group)][1]
                             for group in remaining)
            state_records.append({"sample_id": row["sample_id"], "group_id": row["group_id"],
                                  "mask": mask, "prediction": before_pred,
                                  "correct": before_correct, "confidence": before_conf,
                                  "remaining_singleton_repairable": repairable})
            for group in remaining:
                after_mask = mask | (1 << group)
                after_pred, after_correct, after_conf, after_gold = scored[after_mask]
                transition = ("repair" if not before_correct and after_correct else
                              "damage" if before_correct and not after_correct else "unchanged")
                transitions.append({"sample_id": row["sample_id"], "group_id": row["group_id"],
                    "split": split, "mask": mask, "action_group": group,
                    "after_mask": after_mask, "before_prediction": before_pred,
                    "after_prediction": after_pred, "before_correct": before_correct,
                    "after_correct": after_correct, "before_confidence": before_conf,
                    "after_confidence": after_conf,
                    "confidence_gain": after_conf - before_conf,
                    "before_gold_probability": before_gold,
                    "after_gold_probability": after_gold,
                    "gold_probability_gain": after_gold - before_gold,
                    "signed_01_gain": int(after_correct) - int(before_correct),
                    "signed_ce_gain": math.log(max(after_gold, 1e-300)) -
                                      math.log(max(before_gold, 1e-300)),
                    "transition": transition,
                    "before_high_confidence": before_conf >= HIGH_CONFIDENCE})
    high_wrong = [r for r in state_records if not r["correct"] and
                  r["confidence"] >= HIGH_CONFIDENCE]
    wrong_states = [r for r in state_records if not r["correct"]]
    correct_states = [r for r in state_records if r["correct"]]
    high_correct = [r for r in state_records if r["correct"] and
                    r["confidence"] >= HIGH_CONFIDENCE]
    high_transitions = [r for r in transitions if r["before_high_confidence"]]
    high_wrong_transitions = [r for r in high_transitions if not r["before_correct"]]
    high_correct_transitions = [r for r in high_transitions if r["before_correct"]]
    repairable_count = sum(r["remaining_singleton_repairable"] for r in high_wrong)
    summary = {"split": split, "samples": len(selected), "original_split_samples": original_count,
        "sampled_limit": limit, "distinct_families": len({r["group_id"] for r in selected}),
        "partial_states": len(state_records), "all_transitions": _counts(transitions),
        "confidence_distribution": {
            "all": _confidence_distribution(state_records),
            "wrong": _confidence_distribution(wrong_states),
            "correct": _confidence_distribution(correct_states)},
        "high_confidence_threshold": HIGH_CONFIDENCE,
        "high_confidence": {"all": _counts(high_transitions),
                            "wrong": _counts(high_wrong_transitions),
                            "correct": _counts(high_correct_transitions),
                            "wrong_partial_states": len(high_wrong),
                            "wrong_state_distinct_families": len({r["group_id"] for r in high_wrong}),
                            "wrong_repairable_states": repairable_count,
                            "wrong_repairable_fraction": repairable_count / len(high_wrong) if high_wrong else None,
                            "sufficient_family_count_for_subgroup_claim":
                                len({r["group_id"] for r in high_wrong}) >= 30},
        "notes": ["Every partial state with remaining singleton actions is counted; states within one family are dependent.",
                  "Repairability uses at least one remaining singleton, with no action selected from future responses.",
                  "High-confidence subgroup is descriptive unless at least 30 distinct families contribute."]}
    return transitions, summary


def run(models, cache, out, *, split="validation", limit=None, cpu_threads=2):
    if split not in ALLOWED_SPLITS:
        raise ValueError("test and other splits are forbidden")
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("output exists and is not empty")
    torch.set_num_threads(cpu_threads)
    schema, rows, manifest, config, receipt, head, _, _ = load_model_bundle(
        models, cache, device="cpu")
    transitions, summary = diagnose_rows(rows, schema, head, split=split, limit=limit)
    report = {"format": "cbmjev-cebab-repair-damage-v1",
        "evidence_status": "EXPLORATORY_PILOT_NOT_CLAIM_EVIDENCE" if limit is not None else
                           "EXPLORATORY_OFFLINE_DIAGNOSTIC_NOT_POLICY_EVALUATION",
        "test_evaluated": False, "seed": config["seed"], "summary": summary,
        "provenance": {"schema_hash": schema.hash,
                       "models_receipt_sha256": file_hash(Path(models) / "receipt.json"),
                       "model_weights_sha256": receipt["models_sha256"],
                       "head_component_sha256": receipt["head_component_sha256"],
                       "cache_manifest_sha256": file_hash(Path(cache) / "manifest.json"),
                       "cache_responses_sha256": manifest["responses_sha256"],
                       "script_sha256": file_hash(__file__)},
        "interpretation": "Offline enumeration uses frozen automatic responses and gold y solely as diagnostic outcomes; no selector is trained or evaluated here."}
    report["report_hash"] = stable_hash(report)
    out = fresh_dir(out)
    write_jsonl(out / "transitions.jsonl", transitions)
    write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "cache", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split", choices=ALLOWED_SPLITS, default="validation")
    parser.add_argument("--limit", type=int, help="pilot only: first N selected rows")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    report = run(args.models, args.cache, args.out, split=args.split,
                 limit=args.limit, cpu_threads=args.cpu_threads)
    print(json.dumps({"out": args.out, "seed": report["seed"],
                      "summary": report["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Focused four-seed summary of CUB OOF multi-step conditional-risk pilots."""
import argparse
from collections import Counter
from pathlib import Path
import statistics

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, read_json, read_jsonl, write_json
from scripts.diagnose_conditional_second_query import paired_bootstrap


def summarize_seed(root, seed):
    eval_dir = root / f"cub_multistep_risk_seed{seed}_eval_K16_v1"
    receipt = read_json(eval_dir / "receipt.json")
    unsigned = dict(receipt)
    if (unsigned.pop("report_hash", None) != stable_hash(unsigned) or
            receipt.get("format") != "cbmjev-oof-multistep-risk-validation-v1" or
            receipt.get("status") != "COMPLETE" or receipt.get("seed") != seed or
            receipt.get("test_evaluated") is not False or
            receipt.get("validation_labels_used_for_fit") is not False or
            receipt["traces_sha256"] != file_hash(eval_dir / "traces.jsonl")):
        raise ValueError(f"invalid validation result for seed {seed}")
    rows = read_jsonl(eval_dir / "traces.jsonl")
    if (len(rows) != 594 or receipt["metrics"]["samples"] != len(rows) or
            len({r["sample_id"] for r in rows}) != len(rows)):
        raise ValueError("validation traces have unexpected sample identities")
    fold_configs = []
    for fold in range(3):
        directory = root / f"cub_multistep_risk_seed{seed}_fold{fold}_K16_v1"
        report = read_json(directory / "receipt.json")
        core = dict(report)
        if (core.pop("report_hash", None) != stable_hash(core) or
                report["outer_fold"] != fold or report["seed"] != seed + fold or
                report["checkpoint_sha256"] != file_hash(directory / "risk.pt") or
                receipt["fold_receipt_sha256"][fold] != file_hash(directory / "receipt.json") or
                report["static_order_sha256"] != receipt["static_order_sha256"]):
            raise ValueError("fold checkpoint, configuration or receipt changed")
        fold_configs.append(report["config"])
    comparable = [{k: v for k, v in config.items() if k not in ("device", "seed")}
                  for config in fold_configs]
    if comparable[0] != comparable[1] or comparable[0] != comparable[2]:
        raise ValueError("fold training settings differ")
    if any(len(r["dynamic_order"]) != 16 or len(set(r["dynamic_order"])) != 16
           for r in rows):
        raise ValueError("dynamic policy violated exact K16")
    m = receipt["metrics"]
    for key in ("dynamic_accuracy", "fixed_accuracy", "dynamic_ce", "fixed_ce"):
        field = key.replace("accuracy", "correct")
        recomputed = sum(r[field] for r in rows) / len(rows)
        if abs(m[key] - recomputed) > 1e-10:
            raise ValueError("metric/trace mismatch: " + key)
    paired = [{"adaptive_ce": r["dynamic_ce"], "fixed_ce": r["fixed_ce"],
               "adaptive_correct": r["dynamic_correct"],
               "fixed_correct": r["fixed_correct"]} for r in rows]
    fixed = set(receipt["static_order_prefix"])
    first = Counter(r["dynamic_order"][0] for r in rows)
    return {"seed": seed, "samples": len(rows), "metrics": m,
            "paired_bootstrap_95": paired_bootstrap(paired, seed=seed + 10000),
            "paired_correctness": {
                "dynamic_only": sum(r["dynamic_correct"] and not r["fixed_correct"] for r in rows),
                "fixed_only": sum(r["fixed_correct"] and not r["dynamic_correct"] for r in rows)},
            "first_action_counts": {str(k): v for k, v in sorted(first.items())},
            "unique_dynamic_orders": len({tuple(r["dynamic_order"]) for r in rows}),
            "mean_static_set_overlap": sum(len(set(r["dynamic_order"]) & fixed)
                                           for r in rows) / len(rows),
            "fold_config": comparable[0],
            "evaluation_receipt_sha256": file_hash(eval_dir / "receipt.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[60, 61, 62, 63])
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists() or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("output must be new and seeds unique")
    per_seed = [summarize_seed(Path(args.root), seed) for seed in args.seeds]
    deltas = {key: [entry["metrics"][key] for entry in per_seed]
              for key in ("delta_accuracy", "delta_ce")}
    result = {"format": "cbmjev-oof-multistep-risk-four-seed-summary-v1",
              "evidence_status": "DEVELOPMENT_VALIDATION_NOT_LOCKED_TEST",
              "split": "validation", "samples_per_seed": 594,
              "same_validation_images_reused_across_seeds": True,
              "seeds": args.seeds, "per_seed": per_seed,
              "descriptive_seed_mean": {k: statistics.mean(v) for k, v in deltas.items()},
              "descriptive_seed_sample_sd": {k: statistics.stdev(v) for k, v in deltas.items()},
              "accuracy_better_seeds": sum(v > 0 for v in deltas["delta_accuracy"]),
              "ce_better_seeds": sum(v < 0 for v in deltas["delta_ce"]),
              "interpretation_limit": "Paired image intervals condition on fitted policies and a repeatedly used development set; seeds are correlated, not independent datasets."}
    result["report_hash"] = stable_hash(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print(out)


if __name__ == "__main__":
    main()

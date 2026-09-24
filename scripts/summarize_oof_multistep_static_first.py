#!/usr/bin/env python3
"""Paired static-first intervention versus unchanged full multi-step policy."""
import argparse
from pathlib import Path
import statistics

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, read_json, read_jsonl, write_json
from scripts.diagnose_conditional_second_query import paired_bootstrap


def summarize_seed(root, seed):
    intervened = root / f"cub_multistep_staticfirst_seed{seed}_eval_K16_v1"
    original = root / f"cub_multistep_risk_seed{seed}_eval_K16_v1"
    reports = [read_json(path / "receipt.json") for path in (intervened, original)]
    for path, report in zip((intervened, original), reports):
        core = dict(report)
        if (core.pop("report_hash", None) != stable_hash(core) or
                report["seed"] != seed or report["metrics"]["samples"] != 594 or
                report["traces_sha256"] != file_hash(path / "traces.jsonl")):
            raise ValueError("invalid evaluation report")
    intervention, baseline = reports
    if (intervention["static_order_prefix"] != baseline["static_order_prefix"] or
            intervention["fold_receipt_sha256"] != baseline["fold_receipt_sha256"] or
            intervention["metrics"]["fixed_ce"] != baseline["metrics"]["fixed_ce"] or
            intervention["metrics"]["fixed_accuracy"] != baseline["metrics"]["fixed_accuracy"]):
        raise ValueError("intervention and baseline are not matched")
    for fold in range(3):
        path = root / f"cub_multistep_risk_seed{seed}_fold{fold}_K16_v1"
        receipt = read_json(path / "receipt.json")
        if (receipt["checkpoint_sha256"] != file_hash(path / "risk.pt") or
                intervention["fold_checkpoint_sha256"][fold] != receipt["checkpoint_sha256"] or
                intervention["fold_receipt_sha256"][fold] != file_hash(path / "receipt.json")):
            raise ValueError("fold model changed or mismatched")
    static_rows = read_jsonl(intervened / "traces.jsonl")
    learned_rows = read_jsonl(original / "traces.jsonl")
    if [r["sample_id"] for r in static_rows] != [r["sample_id"] for r in learned_rows]:
        raise ValueError("not paired validation images")
    pairs = [{"adaptive_ce": a["dynamic_ce"], "fixed_ce": b["dynamic_ce"],
              "adaptive_correct": a["dynamic_correct"],
              "fixed_correct": b["dynamic_correct"]}
             for a, b in zip(static_rows, learned_rows)]
    delta_accuracy = sum(p["adaptive_correct"] - p["fixed_correct"]
                         for p in pairs) / len(pairs)
    delta_ce = sum(p["adaptive_ce"] - p["fixed_ce"] for p in pairs) / len(pairs)
    return {"seed": seed, "samples": len(pairs), "static_first_group":
            intervention["static_first_group"],
            "static_first_vs_fixed": intervention["metrics"],
            "learned_first_vs_fixed": baseline["metrics"],
            "static_first_minus_learned_first_accuracy": delta_accuracy,
            "static_first_minus_learned_first_ce": delta_ce,
            "paired_bootstrap_95": paired_bootstrap(pairs, seed=seed + 30000),
            "changed_full_paths": sum(a["dynamic_order"] != b["dynamic_order"]
                                      for a, b in zip(static_rows, learned_rows)),
            "intervention_receipt_sha256": file_hash(intervened / "receipt.json"),
            "original_receipt_sha256": file_hash(original / "receipt.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise ValueError("output must be new")
    per_seed = [summarize_seed(Path(args.root), seed) for seed in (60, 61, 62, 63)]
    result = {"format": "cbmjev-oof-multistep-static-first-four-seed-v1",
              "evidence_status": "DEVELOPMENT_VALIDATION_INTERVENTION_NOT_JEV_GAIN",
              "same_validation_images_reused_across_seeds": True,
              "per_seed": per_seed,
              "descriptive_seed_mean": {key: statistics.mean(row[key] for row in per_seed)
                 for key in ("static_first_minus_learned_first_accuracy",
                             "static_first_minus_learned_first_ce")},
              "ce_better_than_fixed_seeds": sum(
                  row["static_first_vs_fixed"]["delta_ce"] < 0 for row in per_seed),
              "interpretation_limit": "Intervening on first action changes later states; it does not isolate a causal first-step effect. All intervals are development-only."}
    result["report_hash"] = stable_hash(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print(out)


if __name__ == "__main__":
    main()

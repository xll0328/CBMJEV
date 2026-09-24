#!/usr/bin/env python3
"""Audit the frozen CUB BRiG/RLE exact-K16 validation decision gate."""

import argparse
from collections import Counter
from pathlib import Path
import random
import statistics

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, read_json, read_jsonl, write_json
from scripts.train_evaluate_cub_brig_rle_crossfit import source_hashes, verify_rle_fold


SEEDS = (60, 61, 62, 63)
N_VALIDATION = 594
BINDING_KEYS = (
    "merged_receipt_sha256", "responder_receipt_sha256",
    "cache_manifest_sha256", "outer_sources", "plan_binding",
    "head_component_sha256", "controller_component_sha256",
    "responder_checkpoint_sha256",
)


def checked_receipt(directory, expected_format):
    receipt = read_json(directory / "receipt.json")
    unsigned = dict(receipt)
    if (unsigned.pop("receipt_hash", None) != stable_hash(unsigned)
            or receipt.get("status") != "COMPLETE"
            or receipt.get("format") != expected_format):
        raise ValueError(f"invalid receipt: {directory}")
    return receipt


def paired_interval(differences, seed, repetitions=2000):
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(repetitions))
    return [means[int(.025 * repetitions)], means[int(.975 * repetitions)]]


def checked_seed(root, seed):
    brig = root / f"cub_brig_rle_seed{seed}_eval_K16_v1"
    static = root / f"cub_eval_value_budget_grid_seed{seed}_v1"
    for fold in range(3):
        verify_rle_fold(root / f"cub_brig_rle_seed{seed}_fold{fold}_K16_v1")
    receipt = checked_receipt(brig, "cbmjev-crossfit-brig-validation-v1")
    sidecar = read_json(brig / "rle_variant_receipt.json")
    unsigned = dict(sidecar)
    if (unsigned.pop("receipt_hash", None) != stable_hash(unsigned)
            or sidecar.get("format") != "cbmjev-brig-rle-eval-sidecar-v1"
            or sidecar.get("status") != "COMPLETE"
            or sidecar.get("sources_sha256") != source_hashes()
            or sidecar.get("base_receipt_sha256") != file_hash(brig / "receipt.json")
            or sidecar.get("fast_sidecar_sha256") !=
            file_hash(brig / "fast_variant_receipt.json")
            or sidecar.get("fold_sidecars_sha256") != [
                file_hash(root / f"cub_brig_rle_seed{seed}_fold{fold}_K16_v1"
                          / "rle_variant_receipt.json") for fold in range(3)]):
        raise ValueError(f"invalid RLE evaluation sidecar: seed {seed}")
    for name, digest in receipt["files_sha256"].items():
        if file_hash(brig / name) != digest:
            raise ValueError(f"changed BRiG evaluation file: {name}")
    base_receipt = checked_receipt(static, "cbmjev-crossfit-budget-grid-v1")
    a, b = read_json(brig / "metrics.json"), read_json(static / "metrics.json")
    if (a.get("split") != b.get("split") or a.get("split") != "validation"
            or a.get("mode") != b.get("mode") or a.get("mode") != "offline_replay"
            or a.get("seed") != b.get("seed") or a.get("seed") != seed
            or a.get("budget_groups") != 16
            or a.get("paper_evidence") is not False
            or b.get("paper_evidence") is not False
            or a.get("source_code_hash") != b.get("source_code_hash")
            or receipt.get("samples") != N_VALIDATION
            or base_receipt.get("samples_per_policy") != N_VALIDATION):
        raise ValueError(f"mismatched CUB validation protocol: seed {seed}")
    if any(a["source_binding"][key] != b["source_binding"][key]
           for key in BINDING_KEYS):
        raise ValueError(f"responder/head/cache/source mismatch: seed {seed}")
    ancestry_a = a["source_binding"]["ancestry_exclusion"]
    ancestry_b = b["source_binding"]["ancestry_exclusion"]
    # Static-order fitting adds a legitimate extra artifact/ancestor to its
    # exclusion receipt; the target identities and supervised ancestors must
    # nevertheless match, and both checks must pass.
    if (ancestry_a.get("status") != "PASS" or ancestry_b.get("status") != "PASS"
            or any(ancestry_a[key] != ancestry_b[key] for key in (
                "target_group_ids_hash", "ancestor_supervised_group_ids_hash",
                "scope", "external_pretraining_membership_verified",
                "group_disjointness_implies_iid"))):
        raise ValueError(f"incompatible ancestry exclusion: seed {seed}")

    p, q = a["policies"]["brig_mean_q_K16"], b["policies"]["static_K16"]
    if (p["num_samples"] != q["num_samples"] or p["num_samples"] != N_VALIDATION
            or p["budget_groups"] != q["budget_groups"] or p["budget_groups"] != 16):
        raise ValueError(f"incompatible policy metrics: seed {seed}")
    brig_rows = [r for r in read_jsonl(brig / "traces.jsonl")
                 if r["policy_id"] == "brig_mean_q_K16"]
    static_rows = [r for r in read_jsonl(static / "traces.jsonl")
                   if r["policy_id"] == "static_K16"]
    by_id = {r["sample_id"]: r for r in static_rows}
    if (len(brig_rows) != len(static_rows) or len(by_id) != N_VALIDATION
            or len({r["sample_id"] for r in brig_rows}) != N_VALIDATION):
        raise ValueError(f"missing/duplicate paired validation samples: seed {seed}")
    paired, differences = Counter(), []
    for row in brig_rows:
        control = by_id[row["sample_id"]]
        if (row["group_id"] != control["group_id"] or row["y"] != control["y"]
                or row["seed"] != control["seed"] or row["seed"] != seed
                or row["split"] != control["split"] or row["split"] != "validation"
                or len(row["queried_groups"]) != len(control["queried_groups"])
                or len(row["queried_groups"]) != 16):
            raise ValueError(f"invalid paired trace: seed {seed}")
        adaptive = int(row["prediction"] == row["y"])
        fixed = int(control["prediction"] == control["y"])
        paired[(adaptive, fixed)] += 1
        differences.append(adaptive - fixed)
    if (abs(sum(row["prediction"] == row["y"] for row in brig_rows)
            / N_VALIDATION - p["accuracy"]) > 1e-10
            or abs(sum(row["prediction"] == row["y"] for row in static_rows)
                   / N_VALIDATION - q["accuracy"]) > 1e-10):
        raise ValueError(f"trace/metric accuracy mismatch: seed {seed}")
    return {
        "seed": seed, "samples": N_VALIDATION,
        "brig_accuracy": p["accuracy"], "static_accuracy": q["accuracy"],
        "accuracy_delta": p["accuracy"] - q["accuracy"],
        "brig_macro_f1": p["macro_f1"], "static_macro_f1": q["macro_f1"],
        "macro_f1_delta": p["macro_f1"] - q["macro_f1"],
        "paired_correctness": {"brig_only": paired[(1, 0)],
                               "static_only": paired[(0, 1)],
                               "both": paired[(1, 1)], "neither": paired[(0, 0)]},
        "paired_image_accuracy_delta_95_interval": paired_interval(
            differences, 8100 + seed),
        "brig_metrics_sha256": file_hash(brig / "metrics.json"),
        "static_metrics_sha256": file_hash(static / "metrics.json"),
        "brig_eval_receipt_sha256": file_hash(brig / "receipt.json"),
        "static_eval_receipt_sha256": file_hash(static / "receipt.json"),
        "brig_traces_sha256": file_hash(brig / "traces.jsonl"),
        "static_traces_sha256": file_hash(static / "traces.jsonl"),
    }


def summarize(root, seeds):
    rows = [checked_seed(root, seed) for seed in seeds]
    accuracy = [row["accuracy_delta"] for row in rows]
    f1 = [row["macro_f1_delta"] for row in rows]
    result = {
        "format": "cbmjev-cub-brig-rle-k16-validation-gate-v1",
        "evidence_status": "DEVELOPMENT_VALIDATION_NOT_TEST_OR_LATENCY_EVIDENCE",
        "split": "validation", "budget_groups": 16,
        "seeds": list(seeds), "samples_per_seed": N_VALIDATION,
        "same_validation_images_reused_across_seeds": True,
        "per_seed": rows,
        "mean_accuracy_delta": statistics.mean(accuracy),
        "mean_macro_f1_delta": statistics.mean(f1),
        "positive_accuracy_seeds": sum(value > 0 for value in accuracy),
        "positive_macro_f1_seeds": sum(value > 0 for value in f1),
        "sample_sd_accuracy_delta": statistics.stdev(accuracy) if len(rows) > 1 else None,
        "sample_sd_macro_f1_delta": statistics.stdev(f1) if len(rows) > 1 else None,
        "scope_frozen_gate": "after exploratory seed 60 but before seeds 61--63: four specified seeds; accuracy mean >= +1.0 percentage point, >=3/4 positive seeds, and nonnegative mean macro-F1 delta",
        "gate_pass": (tuple(seeds) == SEEDS and statistics.mean(accuracy) >= .01
                      and sum(value > 0 for value in accuracy) >= 3
                      and statistics.mean(f1) >= 0),
        "interpretation_limit": "Paired image intervals condition on fitted policies and repeatedly inspected development validation images; seeds are not independent datasets, and this is not JEV-specific evidence.",
    }
    result["report_hash"] = stable_hash(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    if args.out.exists() or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("output must be new and seeds distinct")
    result = summarize(args.root, tuple(args.seeds))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(args.out)


if __name__ == "__main__":
    main()

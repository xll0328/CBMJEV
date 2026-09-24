#!/usr/bin/env python3
"""Source-matched CUB Choice scalar/attention/fixed K16 validation summary."""

import argparse
from collections import defaultdict
import math
from pathlib import Path
import random
import statistics

from cbmjev.contracts import stable_hash
from cbmjev.evaluation import paired_group_bootstrap
from cbmjev.io import file_hash, read_json, read_jsonl, write_json


def _verified_eval(directory, expected_format):
    directory = Path(directory)
    receipt = read_json(directory / "receipt.json")
    unsigned = dict(receipt)
    if (unsigned.pop("receipt_hash", None) != stable_hash(unsigned)
            or receipt.get("format") != expected_format
            or receipt.get("status") != "COMPLETE"
            or receipt.get("paper_evidence") is not False):
        raise ValueError("invalid development-validation evaluation receipt")
    for name in ("metrics.json", "settings.json", "traces.jsonl"):
        if receipt["files_sha256"].get(name) != file_hash(directory / name):
            raise ValueError("evaluation file hash mismatch: " + name)
    metrics = read_json(directory / "metrics.json")
    traces = read_jsonl(directory / "traces.jsonl")
    if metrics.get("split") != "validation" or metrics.get("mode") != "offline_replay":
        raise ValueError("only offline validation may be summarized")
    return receipt, metrics, traces


def _one_policy(rows, *, policy_id=None, method=None):
    chosen = [row for row in rows if (row.get("policy_id") == policy_id if policy_id
                                     else row.get("method") == method)]
    if len(chosen) != 594 or len({row["sample_id"] for row in chosen}) != 594:
        raise ValueError("expected exactly 594 distinct CUB validation samples")
    if any(row["split"] != "validation" or row["mode"] != "offline_replay"
           for row in chosen):
        raise ValueError("unexpected policy split/mode")
    return chosen


def _nll(row):
    probabilities, label = row["probabilities"], row["y"]
    if (type(label) is not int or not 0 <= label < len(probabilities)
            or len(probabilities) != 200
            or any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 or p > 1
                   for p in probabilities)
            or abs(sum(probabilities) - 1) > 1e-4):
        raise ValueError("invalid CUB class distribution")
    return -math.log(max(probabilities[label], 1e-12))


def _basic(rows):
    return {"accuracy": statistics.mean(row["prediction"] == row["y"] for row in rows),
            "ce": statistics.mean(_nll(row) for row in rows),
            "mean_queried_groups": statistics.mean(len(row["queried_groups"]) for row in rows),
            "mean_declared_cost": statistics.mean(row["declared_cost"] for row in rows),
            "zero_query_fraction": statistics.mean(len(row["queried_groups"]) == 0 for row in rows),
            "exact_16_query_fraction": statistics.mean(len(row["queried_groups"]) == 16 for row in rows)}


def _paired_ce(a, b, *, seed, draws):
    by_group = defaultdict(list)
    for left, right in zip(a, b):
        by_group[left["group_id"]].append(_nll(left) - _nll(right))
    values = [statistics.mean(parts) for _, parts in sorted(by_group.items())]
    rng = random.Random(seed)
    sampled = sorted(statistics.mean(values[rng.randrange(len(values))]
                                      for _ in values) for _ in range(draws))
    lower = sampled[int(.025 * (draws - 1))]
    upper = sampled[int(.975 * (draws - 1))]
    return {"equal_group_mean_ce_delta_a_minus_b": statistics.mean(values),
            "paired_group_bootstrap_95": [lower, upper],
            "uncertainty_kind": "group_sampling_only_fixed_trained_models"}


def _compare(a, b, *, seed, draws):
    index_a = {row["sample_id"]: row for row in a}
    index_b = {row["sample_id"]: row for row in b}
    if set(index_a) != set(index_b):
        raise ValueError("policy validation sample sets differ")
    ordered = sorted(index_a)
    left, right = [index_a[key] for key in ordered], [index_b[key] for key in ordered]
    for x, y in zip(left, right):
        if (x["group_id"], x["y"], x["split"], x["mode"]) != (y["group_id"], y["y"], y["split"], y["mode"]):
            raise ValueError("paired validation identities/labels differ")
    metrics_a, metrics_b = _basic(left), _basic(right)
    return {"accuracy_delta_a_minus_b": metrics_a["accuracy"] - metrics_b["accuracy"],
            "sample_mean_ce_delta_a_minus_b": metrics_a["ce"] - metrics_b["ce"],
            "mean_queries_delta_a_minus_b": metrics_a["mean_queried_groups"] - metrics_b["mean_queried_groups"],
            "group_bootstrap_error_and_cost": paired_group_bootstrap(left, right,
                seed=seed, n_resamples=draws),
            "group_bootstrap_ce": _paired_ce(left, right, seed=seed + 1, draws=draws)}


def summarize(choice_dir, fixed_dir, *, draws=2000, seed=60):
    if type(draws) is not int or draws < 100 or type(seed) is not int or seed < 0:
        raise ValueError("invalid bootstrap settings")
    choice_receipt, choice_metrics, choice_rows = _verified_eval(
        choice_dir, "cbmjev-choice-crossfit-validation-v1")
    fixed_receipt, fixed_metrics, fixed_rows = _verified_eval(
        fixed_dir, "cbmjev-crossfit-validation-v1")
    source = choice_receipt["source_binding"]["nested_system"]
    if (source != fixed_receipt["source_binding"]
            or choice_metrics["source_binding"] != choice_receipt["source_binding"]
            or fixed_metrics["source_binding"] != fixed_receipt["source_binding"]
            or choice_metrics.get("seed") != fixed_metrics.get("seed")
            or choice_metrics.get("seed") != seed):
        raise ValueError("Choice and fixed evaluations do not share an identical nested system/seed")
    scalar = _one_policy(choice_rows, policy_id="structured_choice_scalar")
    attention = _one_policy(choice_rows, policy_id="structured_choice_attention")
    fixed = _one_policy(fixed_rows, method="fixed")
    if (fixed_metrics["policies"]["fixed"]["mean_queried_groups"] != 16
            or any(len(row["queried_groups"]) != 16 for row in fixed)):
        raise ValueError("fixed comparator must use exact K16")
    raw = {"scalar": scalar, "attention": attention, "fixed_K16": fixed}
    result = {"format": "cbmjev-choice-paired-k16-summary-v1",
        "evidence_status": "SINGLE_SEED_DEVELOPMENT_VALIDATION_NOT_LOCKED_TEST",
        "seed": seed, "split": "validation", "samples": 594,
        "same_nested_system_verified": True,
        "choice_receipt_sha256": file_hash(Path(choice_dir) / "receipt.json"),
        "fixed_receipt_sha256": file_hash(Path(fixed_dir) / "receipt.json"),
        "metrics": {name: _basic(rows) for name, rows in raw.items()},
        "comparisons": {
            "attention_minus_scalar": _compare(attention, scalar, seed=seed + 1000, draws=draws),
            "attention_minus_fixed_K16": _compare(attention, fixed, seed=seed + 2000, draws=draws),
            "scalar_minus_fixed_K16": _compare(scalar, fixed, seed=seed + 3000, draws=draws)},
        "interpretation_limit": "Choice has a learned STOP and at most 16 queries, whereas fixed uses exactly 16; accuracy must be read jointly with query/cost means. Group bootstrap conditions on fitted policies and reused development data; shared GPU timing is not paper evidence."}
    result["report_hash"] = stable_hash(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choice-eval", required=True)
    parser.add_argument("--fixed-eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=60)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise ValueError("summary output must be new")
    result = summarize(args.choice_eval, args.fixed_eval, draws=args.draws, seed=args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print({"status": "COMPLETE", "out": str(out), "metrics": result["metrics"]}, flush=True)


if __name__ == "__main__":
    main()

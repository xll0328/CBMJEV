#!/usr/bin/env python3
"""Diagnose instance-conditional acquisition against a matched static mixture.

This is a validation-only development diagnostic.  The static reference is the
convex mixture of two fixed-budget static policies whose expected
query count matches the adaptive policy.  It is an expected randomized-policy
comparison, not a measured additional training run or a test-set claim.
Optionally select the best validation-accuracy mixture across all provided
static budgets; this is an optimistic diagnostic, not a train-selected policy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from itertools import combinations


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_traces(path: Path, method: str):
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get("method") != method:
                continue
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in result:
                raise ValueError(f"duplicate/invalid sample_id at {path}:{line_number}")
            if row.get("split") != "validation" or row.get("mode") != "offline_replay":
                raise ValueError("branching diagnostic accepts validation offline replay only")
            if not isinstance(row.get("group_id"), str) or type(row.get("y")) is not int:
                raise ValueError("trace needs group_id and integer target")
            result[sample_id] = row
    if not result:
        raise ValueError(f"no traces for method {method}")
    return result


def load_run(directory: Path, method: str):
    metrics_path, settings_path, traces_path = (directory / name for name in
                                                 ("metrics.json", "settings.json", "traces.jsonl"))
    metrics, settings = read_json(metrics_path), read_json(settings_path)
    if (metrics.get("split") != "validation" or metrics.get("mode") != "offline_replay"
            or metrics.get("paper_evidence") is not False):
        raise ValueError("requires explicit validation-only non-paper metrics")
    if method not in metrics.get("policies", {}) or settings.get(method, {}).get("method") != method:
        raise ValueError(f"missing {method} policy metrics/settings")
    report = metrics["policies"][method]
    traces = read_traces(traces_path, method)
    if report.get("num_samples") != len(traces):
        raise ValueError("metrics/trace sample count mismatch")
    return {
        "directory": str(directory.resolve()), "metrics": metrics, "settings": settings,
        "report": report, "traces": traces,
        "hashes": {name: sha256(path) for name, path in
                   (("metrics", metrics_path), ("settings", settings_path), ("traces", traces_path))},
    }


def macro_f1_from_confusion(matrix):
    scores = []
    for label in range(len(matrix)):
        tp = matrix[label][label]
        fp = sum(matrix[truth][label] for truth in range(len(matrix)) if truth != label)
        fn = sum(matrix[label][pred] for pred in range(len(matrix)) if pred != label)
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return statistics.mean(scores)


def mix_confusions(low, high, probability_high):
    if len(low) != len(high) or any(len(a) != len(b) for a, b in zip(low, high)):
        raise ValueError("static confusion matrices differ in shape")
    return [[(1 - probability_high) * a + probability_high * b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(low, high)]


def percentile(sorted_values, probability):
    index = min(len(sorted_values) - 1, max(0, math.ceil(probability * len(sorted_values)) - 1))
    return sorted_values[index]


def validate_comparison_identity(runs):
    """Bind a matched-budget comparison to identical models and response data.

    Matching seeds/source alone is insufficient: two tuned heads can share both.
    Missing historical provenance is rejected, never treated as equal evidence.
    Policy budgets may differ; training settings and declared cost units may not.
    """
    fields = ("schema_hash", "models_receipt_sha256", "models_sha256",
              "cache_sha256", "static_order_sha256", "responder_checkpoint_sha256",
              "head_component_sha256", "controller_component_sha256",
              "training_source_code_hash")
    identity = {}
    for field in fields:
        values = [run["metrics"].get(field) for run in runs]
        if any(not isinstance(value, str) or len(value) != 64
               or any(c not in "0123456789abcdef" for c in value) for value in values):
            raise ValueError("missing/invalid comparison provenance: " + field)
        if len(set(values)) != 1:
            raise ValueError("comparison artifacts differ: " + field)
        identity[field] = values[0]
    configs = [run["settings"][run["report"]["method"]]["config"] for run in runs]
    for field in ("learning", "cost"):
        if any(field not in config for config in configs):
            raise ValueError("missing comparison configuration: " + field)
        if any(config[field] != configs[0][field] for config in configs[1:]):
            raise ValueError("comparison configuration differs: " + field)
    return identity


def select_static_frontier_pair(candidates, query_count):
    """Best validation-accuracy static mixture at an observed expected budget.

    This is an optimistic development comparator, not a train-selected policy.
    All distinct-cost pairs are considered, not merely neighboring budgets.
    """
    if not math.isfinite(query_count):
        raise ValueError("query count must be finite")
    candidates = sorted(candidates, key=lambda r: (r["report"]["mean_queried_groups"],
                                                   r["directory"]))
    best = None
    for low, high in combinations(candidates, 2):
        ql, qh = (float(r["report"]["mean_queried_groups"]) for r in (low, high))
        al, ah = (float(r["report"]["accuracy"]) for r in (low, high))
        if not all(math.isfinite(x) for x in (ql, qh, al, ah)):
            raise ValueError("static frontier metrics must be finite")
        if not ql < qh or not ql <= query_count <= qh:
            continue
        p = (query_count - ql) / (qh - ql)
        score = (1 - p) * al + p * ah
        if best is None or score > best[0]:
            best = (score, low, high)
    if best is None:
        raise ValueError("static candidates do not bracket requested budget")
    return best[1], best[2]


def validate_matched_population(runs):
    identities = {(run["metrics"]["seed"], run["metrics"]["source_revision"],
                   run["metrics"]["source_code_hash"]) for run in runs}
    if len(identities) != 1:
        raise ValueError("runs differ in seed/source revision/code hash")
    sample_sets = [set(run["traces"]) for run in runs]
    if sample_sets[1:] != sample_sets[:-1]:
        raise ValueError("runs do not contain the same validation samples")
    for sample_id in sample_sets[0]:
        keys = {(run["traces"][sample_id]["group_id"], run["traces"][sample_id]["y"]) for run in runs}
        if len(keys) != 1:
            raise ValueError("group/target differs across matched traces")


def analyze(dynamic, static_low, static_high, *, bootstrap_resamples=10000, bootstrap_seed=20260922):
    runs = (dynamic, static_low, static_high)
    comparison_identity = validate_comparison_identity(runs)
    validate_matched_population(runs)
    q_dynamic = float(dynamic["report"]["mean_queried_groups"])
    q_low = float(static_low["report"]["mean_queried_groups"])
    q_high = float(static_high["report"]["mean_queried_groups"])
    if not q_low < q_high or not q_low <= q_dynamic <= q_high:
        raise ValueError("adaptive query count is not bracketed by static budgets")
    probability_high = (q_dynamic - q_low) / (q_high - q_low)
    low_accuracy, high_accuracy = (float(run["report"]["accuracy"])
                                   for run in (static_low, static_high))
    mixture_accuracy = (1 - probability_high) * low_accuracy + probability_high * high_accuracy
    mixture_group_risk = ((1 - probability_high) * float(static_low["report"]["group_mean_risk"])
                          + probability_high * float(static_high["report"]["group_mean_risk"]))
    mixed_confusion = mix_confusions(static_low["report"]["confusion_matrix_true_rows"],
                                     static_high["report"]["confusion_matrix_true_rows"], probability_high)
    mixture_macro_f1 = macro_f1_from_confusion(mixed_confusion)

    branch_rows = defaultdict(list)
    first_values = defaultdict(Counter)
    group_effects = defaultdict(list)
    sample_effects = []
    for sample_id, row in dynamic["traces"].items():
        low, high = static_low["traces"][sample_id], static_high["traces"][sample_id]
        query_set = tuple(row["queried_groups"])
        branch_rows[query_set].append((sample_id, row, low, high))
        values = tuple(row.get("steps", [{}])[0].get("values", ()))
        first_values[values][query_set] += 1
        dynamic_correct = float(row["prediction"] == row["y"])
        expected_static_correct = ((1 - probability_high) * float(low["prediction"] == row["y"])
                                   + probability_high * float(high["prediction"] == row["y"]))
        effect = dynamic_correct - expected_static_correct
        sample_effects.append(effect)
        group_effects[row["group_id"]].append(effect)

    branch_summary = []
    for query_set, entries in sorted(branch_rows.items()):
        def accuracy(position):
            return statistics.mean(float(item[position]["prediction"] == item[1]["y"]) for item in entries)
        sequences = Counter(tuple(tuple(step["action"]) for step in item[1]["steps"]) for item in entries)
        branch_summary.append({
            "queried_groups": list(query_set), "num_samples": len(entries),
            "fraction_samples": len(entries) / len(dynamic["traces"]),
            "dynamic_accuracy": accuracy(1), "static_low_accuracy_on_same_cases": accuracy(2),
            "static_high_accuracy_on_same_cases": accuracy(3),
            "action_sequences": [{"actions": [list(action) for action in sequence], "count": count}
                                 for sequence, count in sorted(sequences.items())],
        })

    unit_effects = [statistics.mean(values) for values in group_effects.values()]
    if type(bootstrap_resamples) is not int or bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    rng = random.Random(bootstrap_seed)
    bootstrap = sorted(statistics.mean(rng.choices(unit_effects, k=len(unit_effects)))
                       for _ in range(bootstrap_resamples))
    group_gain = statistics.mean(unit_effects)
    return {
        "format": "cbmjev-adaptive-branching-diagnostic-v1",
        "evidence_status": "VALIDATION_SINGLE_SEED_DEVELOPMENT_ONLY_NOT_PAPER_CLAIM",
        "paper_claim": False, "test_evaluated": False,
        "seed": dynamic["metrics"]["seed"], "source_revision": dynamic["metrics"]["source_revision"],
        "num_samples": len(dynamic["traces"]), "num_independent_groups": len(group_effects),
        "adaptive": {"method": dynamic["report"]["method"], "mean_queried_groups": q_dynamic,
                     "accuracy": dynamic["report"]["accuracy"], "macro_f1": dynamic["report"]["macro_f1"],
                     "group_mean_risk": dynamic["report"]["group_mean_risk"]},
        "matched_static_convex_mixture": {
            "low_budget_mean_groups": q_low, "high_budget_mean_groups": q_high,
            "probability_high_budget": probability_high, "expected_mean_groups": q_dynamic,
            "expected_accuracy": mixture_accuracy, "expected_macro_f1_from_mixed_confusion": mixture_macro_f1,
            "expected_group_mean_risk": mixture_group_risk,
            "interpretation": "expected randomized static policy, not an additional measured run",
        },
        "adaptive_minus_mixture": {
            "sample_weighted_accuracy": float(dynamic["report"]["accuracy"]) - mixture_accuracy,
            "group_equal_accuracy": group_gain,
            "group_bootstrap_95_percentile_interval": [percentile(bootstrap, .025), percentile(bootstrap, .975)],
            "bootstrap_resamples": bootstrap_resamples, "bootstrap_seed": bootstrap_seed,
            "unit": "independent validation group; within-group samples averaged before resampling",
        },
        "conditional_branches": branch_summary,
        "first_observed_values_to_branch": [
            {"values": list(values), "branches": [{"queried_groups": list(query_set), "count": count}
                                                   for query_set, count in sorted(counts.items())]}
            for values, counts in sorted(first_values.items())
        ],
        "diagnostic_checks": {
            "distinct_adaptive_query_sets": len(branch_rows),
            "branch_is_deterministic_given_recorded_first_values": all(len(counts) == 1 for counts in first_values.values()),
            "matched_sample_ids": True, "matched_seed_revision_code": True,
        },
        "input_hashes": {label: run["hashes"] for label, run in
                         (("dynamic", dynamic), ("static_low", static_low), ("static_high", static_high))},
        "comparison_artifact_identity": comparison_identity,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic", required=True, help="evaluation dir containing the adaptive policy")
    parser.add_argument("--static-low")
    parser.add_argument("--static-high")
    parser.add_argument("--static-runs", nargs="+", help="use optimistic validation static-mixture envelope")
    parser.add_argument("--adaptive-method", default="value")
    parser.add_argument("--reference-method", choices=("static", "static_value"), default="static",
                        help="fixed-budget static order, or fixed order with learned stopping")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260922)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    dynamic = load_run(Path(args.dynamic), args.adaptive_method)
    candidates = None
    if args.static_runs:
        if args.static_low or args.static_high:
            parser.error("static-runs cannot be combined with explicit low/high")
        candidates = [load_run(Path(path), args.reference_method) for path in args.static_runs]
        validate_comparison_identity([dynamic] + candidates)
        validate_matched_population([dynamic] + candidates)
        low, high = select_static_frontier_pair(candidates, dynamic["report"]["mean_queried_groups"])
    else:
        if not args.static_low or not args.static_high:
            parser.error("provide static-low and static-high, or static-runs")
        low, high = load_run(Path(args.static_low), args.reference_method), load_run(Path(args.static_high), args.reference_method)
    report = analyze(dynamic, low, high,
                     bootstrap_resamples=args.bootstrap_resamples, bootstrap_seed=args.bootstrap_seed)
    report["reference_method"] = args.reference_method
    if candidates is not None:
        report["static_reference_selection"] = {
            "criterion": "maximum validation sample-weighted accuracy at matched expected query count",
            "status": "OPTIMISTIC_DEVELOPMENT_COMPARATOR_NOT_TRAIN_SELECTED_POLICY",
            "selected_low": low["directory"], "selected_high": high["directory"],
            "candidates": [{"directory": c["directory"], "hashes": c["hashes"]} for c in candidates],
            "uncertainty_caveat": "bootstrap conditions on selected pair and mixture weight; does not adjust for selection or tuning"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"out": str(out.resolve()), "gain": report["adaptive_minus_mixture"],
                      "branches": report["diagnostic_checks"]}, sort_keys=True))


if __name__ == "__main__":
    main()

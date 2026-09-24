#!/usr/bin/env python3
"""Development-only paired K16 error transitions from saved group metrics.

The original per-case traces are not local. A group with multiple samples
cannot identify which sample each policy got right, so transitions are counted
only for groups with exactly one sample. Aggregate sample means still use all
groups with their original sample weights. ``fixed_only`` names the comparator
side of a transition even when --fixed selects a forced-stop policy rather
than a static fixed-order policy; the report records both exact policy names.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, write_json


def _bucket(queries):
    if queries == 0:
        return "0"
    for ceiling, name in ((4, "1-4"), (8, "5-8"), (12, "9-12"),
                          (15, "13-15"), (16, "16")):
        if queries <= ceiling:
            return name
    raise ValueError("query count exceeds K16")


def _groups(metrics, name):
    policy = metrics["policies"][name]
    groups = policy["group_metrics"]
    total = 0
    errors = 0.
    queries = 0.
    for group in groups.values():
        count, error, acquired = (group[key] for key in
                                  ("num_samples", "error", "queried_groups"))
        if (type(count) is not int or count < 1 or not math.isfinite(error)
                or not 0 <= error <= 1 or not math.isfinite(acquired)
                or not 0 <= acquired <= 16):
            raise ValueError("invalid group metric")
        total += count
        errors += count * error
        queries += count * acquired
    if (total != policy["num_samples"]
            or not math.isclose(1 - errors / total, policy["accuracy"], abs_tol=1e-9)
            or not math.isclose(queries / total, policy["mean_queried_groups"], abs_tol=1e-9)):
        raise ValueError("group metrics disagree with policy sample aggregates")
    return groups


_MATCHED_SYSTEM_KEYS = ("merged_receipt_sha256", "responder_receipt_sha256",
    "cache_manifest_sha256", "head_component_sha256", "controller_component_sha256",
    "responder_checkpoint_sha256", "outer_sources", "plan_binding")


def analyze(paths, *, comparator_paths=None, adaptive="value_K16", fixed="fixed_K16"):
    if not paths or len({str(Path(path).resolve()) for path in paths}) != len(paths):
        raise ValueError("distinct nonempty metric paths required")
    if comparator_paths is not None and len(comparator_paths) != len(paths):
        raise ValueError("one comparator metric path required per adaptive metric path")
    comparator_paths = paths if comparator_paths is None else comparator_paths
    by_seed = {}
    transitions_by_image = {}
    singleton_keys = {}
    for path, comparator_path in zip(paths, comparator_paths):
        path, comparator_path = Path(path), Path(comparator_path)
        metrics = json.loads(path.read_text(encoding="utf-8"))
        comparator = (metrics if path.resolve() == comparator_path.resolve() else
                      json.loads(comparator_path.read_text(encoding="utf-8")))
        seed = metrics["seed"]
        if (type(seed) is not int or seed in by_seed or metrics["split"] != "validation"
                or metrics["mode"] != "offline_replay"
                or (comparator["seed"], comparator["split"], comparator["mode"])
                   != (seed, "validation", "offline_replay")):
            raise ValueError("duplicate seed or non-validation/offline metrics")
        if comparator is not metrics:
            first, second = metrics["source_binding"], comparator["source_binding"]
            if any(key not in first or key not in second or first[key] != second[key]
                   for key in _MATCHED_SYSTEM_KEYS):
                raise ValueError("cross-file policies do not share the same nested system")
        dynamic, static = _groups(metrics, adaptive), _groups(comparator, fixed)
        if dynamic.keys() != static.keys():
            raise ValueError("paired policies lack identical image groups")
        if any(group["queried_groups"] != 16 for group in static.values()):
            raise ValueError("fixed comparator is not exact K16")
        transitions = Counter()
        loss_queries = Counter()
        all_queries = Counter()
        excluded = {}
        for image_id, current in dynamic.items():
            fixed_group = static[image_id]
            if current["num_samples"] != fixed_group["num_samples"]:
                raise ValueError("paired image-group sample counts differ")
            if current["num_samples"] != 1:
                excluded[image_id] = current["num_samples"]
                continue
            a_error, f_error = current["error"], fixed_group["error"]
            if a_error not in (0, 1) or f_error not in (0, 1):
                raise ValueError("singleton error must be binary")
            status = ("both_correct" if a_error == f_error == 0 else
                      "adaptive_only" if a_error == 0 else
                      "fixed_only" if f_error == 0 else "both_wrong")
            transitions[status] += 1
            bucket = _bucket(current["queried_groups"])
            all_queries[bucket] += 1
            if status == "fixed_only":
                loss_queries[bucket] += 1
            transitions_by_image.setdefault(image_id, {})[seed] = status
        by_seed[seed] = {"metric_path": str(path.resolve()),
            "metric_sha256": file_hash(path),
            "comparator_metric_path": str(comparator_path.resolve()),
            "comparator_metric_sha256": file_hash(comparator_path),
            "cross_file_nested_system_match_verified": comparator is not metrics,
            "samples": metrics["policies"][adaptive]["num_samples"],
            "paired_singleton_images": sum(transitions.values()),
            "excluded_multisample_images": excluded,
            "adaptive_accuracy": metrics["policies"][adaptive]["accuracy"],
            "fixed_accuracy": comparator["policies"][fixed]["accuracy"],
            "adaptive_minus_fixed_accuracy": (metrics["policies"][adaptive]["accuracy"]
                                              - comparator["policies"][fixed]["accuracy"]),
            "adaptive_mean_queried_groups": metrics["policies"][adaptive]["mean_queried_groups"],
            "fixed_mean_queried_groups": comparator["policies"][fixed]["mean_queried_groups"],
            "singleton_transitions": dict(sorted(transitions.items())),
            "adaptive_query_buckets_singletons": dict(sorted(all_queries.items())),
            "fixed_only_by_adaptive_query_bucket": dict(sorted(loss_queries.items()))}
        singleton_keys[seed] = set(dynamic) - set(excluded)
    seeds = sorted(by_seed)
    if any(singleton_keys[seed] != singleton_keys[seeds[0]] for seed in seeds[1:]):
        raise ValueError("single-sample image groups differ across seeds")
    complete = {image_id: statuses for image_id, statuses in transitions_by_image.items()
                if set(statuses) == set(seeds)}
    if len(complete) != len(singleton_keys[seeds[0]]):
        raise ValueError("incomplete singleton image transitions")
    fixed_only_frequency = Counter(sum(status == "fixed_only" for status in statuses.values())
                                   for statuses in complete.values())
    report = {"format": "cbmjev-cub-k16-group-transition-diagnostic-v1",
        "evidence_status": "LOCAL_VALIDATION_METRICS_ONLY_NO_ORIGINAL_RECEIPTS_OR_TRACES",
        "policies": {"adaptive": adaptive, "fixed": fixed},
        "limitations": ["Only single-sample image groups identify paired correctness transitions.",
            "The four seeds reuse images; seed-by-image cells are not independent trials.",
            "These group summaries cannot recover actions, selection order, logits or causal stopping effects.",
            "Local metric JSON files lack the original completion receipts; this is a diagnostic, not a paper claim."],
        "cross_file_match_scope": "Eight nested-system binding fields must match; static-order ancestry may differ because the policies do not share that artifact.",
        "seeds": seeds, "by_seed": {str(seed): by_seed[seed] for seed in seeds},
        "complete_singleton_images_across_seeds": len(complete),
        "fixed_only_frequency_across_seeds": dict(sorted(fixed_only_frequency.items()))}
    report["report_hash"] = stable_hash(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--comparator-metrics", nargs="+",
                        help="paired comparator files; defaults to --metrics")
    parser.add_argument("--out", required=True)
    parser.add_argument("--adaptive", default="value_K16")
    parser.add_argument("--fixed", default="fixed_K16")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise ValueError("analysis output must be new")
    report = analyze(args.metrics, comparator_paths=args.comparator_metrics,
                     adaptive=args.adaptive, fixed=args.fixed)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    print({"out": str(out), "seeds": report["seeds"],
           "singletons": report["complete_singleton_images_across_seeds"]}, flush=True)


if __name__ == "__main__":
    main()

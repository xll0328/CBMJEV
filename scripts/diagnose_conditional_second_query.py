#!/usr/bin/env python3
"""Exploratory two-query adaptive-headroom diagnostic on held-out validation halves.

This uses validation labels to FIT a simple outcome-conditional second-query
table, so it is model-development evidence only, never a locked test result or
an unbiased estimate of the best adaptive policy. The final head and automatic
response cache are fixed. Labels never enter the simulated inference state.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from cbmjev.contracts import Schema, stable_hash
from cbmjev.crossfit_artifacts import load_crossfit_head
from cbmjev.io import file_hash, read_json, read_jsonl
from cbmjev.learning import mask_answers, schema_signature


def split_fold(group_id, seed):
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2


def outcome_key(answers, schema, first_group):
    return tuple(answers[atom] for atom in schema.groups[first_group].atoms)


def choose_second(train, candidates, *, smoothing):
    """Fit global and outcome-specific CE means from training-fold cases only."""
    if not train or smoothing < 0:
        raise ValueError("nonempty training data and nonnegative smoothing required")
    global_loss = {g: sum(row["ce"][g] for row in train) / len(train)
                   for g in candidates}
    fixed = min(candidates, key=lambda g: (global_loss[g], g))
    by_outcome = defaultdict(list)
    for row in train:
        by_outcome[row["outcome"]].append(row)
    policy = {}
    for outcome, subset in by_outcome.items():
        estimates = {g: (sum(row["ce"][g] for row in subset)
                         + smoothing * global_loss[g]) / (len(subset) + smoothing)
                     for g in candidates}
        policy[outcome] = min(candidates, key=lambda g: (estimates[g], g))
    return fixed, policy, dict(global_loss), {str(k): len(v) for k, v in by_outcome.items()}


def summarize(evaluated):
    n = len(evaluated)
    if not n:
        raise ValueError("empty evaluation")
    result = {"n": n, "adaptive_ce": sum(x["adaptive_ce"] for x in evaluated) / n,
              "fixed_ce": sum(x["fixed_ce"] for x in evaluated) / n,
              "adaptive_accuracy": sum(x["adaptive_correct"] for x in evaluated) / n,
              "fixed_accuracy": sum(x["fixed_correct"] for x in evaluated) / n,
              "adaptive_minus_fixed_ce": sum(x["adaptive_ce"] - x["fixed_ce"]
                                              for x in evaluated) / n,
              "adaptive_minus_fixed_accuracy": sum(x["adaptive_correct"] - x["fixed_correct"]
                                                    for x in evaluated) / n,
              "changed_action_fraction": sum(x["adaptive_action"] != x["fixed_action"]
                                             for x in evaluated) / n}
    pairs = Counter((x["adaptive_correct"], x["fixed_correct"]) for x in evaluated)
    result["paired_correctness"] = {"both": pairs[(1, 1)],
                                    "adaptive_only": pairs[(1, 0)],
                                    "fixed_only": pairs[(0, 1)],
                                    "neither": pairs[(0, 0)]}
    return result


def paired_bootstrap(evaluated, *, seed, repetitions=2000):
    """Conditional-on-fitted-policies uncertainty across original image groups."""
    rng = random.Random(seed)
    n = len(evaluated)
    intervals = {}
    for key, field in (("accuracy", "correct"), ("cross_entropy", "ce")):
        differences = [row["adaptive_" + field] - row["fixed_" + field]
                       for row in evaluated]
        means = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n
                       for _ in range(repetitions))
        intervals[key] = {"lower_95": means[int(.025 * repetitions)],
                          "upper_95": means[int(.975 * repetitions)],
                          "interpretation": "paired image bootstrap, fitted policies fixed"}
    return intervals


def outcome_shuffle_control(heldout_parts, *, seed, repetitions=1000):
    """Shuffle first-query answers within each held-out fold, retaining menus.

    This checks whether the pairing between an observed answer and its chosen
    continuation matters, while preserving each fold's action frequencies.
    It is exploratory because the analysis itself was designed on validation.
    """
    rng = random.Random(seed + 8123)
    n = sum(len(test) for test, _, _ in heldout_parts)
    observed = sum(row["ce"][policy.get(row["outcome"], fixed)] - row["ce"][fixed]
                   for test, policy, fixed in heldout_parts for row in test) / n
    shuffled = []
    for _ in range(repetitions):
        total = 0.0
        for test, policy, fixed in heldout_parts:
            outcomes = [row["outcome"] for row in test]
            rng.shuffle(outcomes)
            for row, permuted_outcome in zip(test, outcomes):
                second = policy.get(permuted_outcome, fixed)
                total += row["ce"][second] - row["ce"][fixed]
        shuffled.append(total / n)
    shuffled.sort()
    return {"observed_adaptive_minus_fixed_ce": observed,
            "shuffled_mean": sum(shuffled) / repetitions,
            "shuffled_lower_95": shuffled[int(.025 * repetitions)],
            "shuffled_upper_95": shuffled[int(.975 * repetitions)],
            "fraction_shuffled_at_least_as_good": sum(x <= observed for x in shuffled) / repetitions,
            "repetitions": repetitions,
            "interpretation": "within-fold first-response permutation; exploratory development control"}


def analyze(rows, candidates, *, smoothing, seed):
    folds = {f: [row for row in rows if split_fold(row["group_id"], seed) == f]
             for f in (0, 1)}
    if any(not part for part in folds.values()):
        raise ValueError("both held-out folds must be nonempty")
    evaluated, reports, heldout_parts = [], [], []
    for holdout in (0, 1):
        train, test = folds[1 - holdout], folds[holdout]
        fixed, policy, global_loss, counts = choose_second(
            train, candidates, smoothing=smoothing)
        fold_eval = []
        for row in test:
            dynamic = policy.get(row["outcome"], fixed)
            entry = {"adaptive_action": dynamic, "fixed_action": fixed,
                     "adaptive_ce": row["ce"][dynamic],
                     "fixed_ce": row["ce"][fixed],
                     "adaptive_correct": row["correct"][dynamic],
                     "fixed_correct": row["correct"][fixed]}
            fold_eval.append(entry)
        evaluated.extend(fold_eval)
        heldout_parts.append((test, policy, fixed))
        reports.append({"holdout_fold": holdout, "fit_n": len(train),
                        "test_n": len(test), "fixed_second_group": fixed,
                        "learned_outcome_actions": {str(k): v for k, v in policy.items()},
                        "fit_outcome_counts": counts,
                        "fit_outcomes_with_at_most_3_cases": sum(v <= 3 for v in counts.values()),
                        "unseen_outcome_count": sum(row["outcome"] not in policy for row in test),
                        "global_fit_ce_by_group": global_loss,
                        "heldout": summarize(fold_eval)})
    return {"aggregate": summarize(evaluated),
            "paired_bootstrap_95": paired_bootstrap(evaluated, seed=seed),
            "outcome_shuffle_control": outcome_shuffle_control(heldout_parts, seed=seed),
            "folds": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--static-order", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--smoothing", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.smoothing < 0 or not math.isfinite(args.smoothing) or args.batch_size < 1:
        raise ValueError("invalid smoothing or batch size")
    out = Path(args.out)
    if out.exists():
        raise ValueError("refuse to overwrite existing report")
    cache = Path(args.cache)
    manifest = read_json(cache / "manifest.json")
    unsigned = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if (manifest.get("manifest_hash") != stable_hash(unsigned)
            or manifest.get("split") != "validation"
            or manifest.get("response_source") != "automatic_model"
            or manifest.get("status") != "COMPLETE"):
        raise ValueError("requires completed automatic validation cache")
    for name in ("schema.json", "responses.jsonl", "exclusions.jsonl"):
        if manifest["files_sha256"].get(name) != file_hash(cache / name):
            raise ValueError("cache file hash mismatch: " + name)
    schema = Schema.from_dict(read_json(cache / "schema.json"))
    head, head_report = load_crossfit_head(args.head, schema, device="cpu")
    order = read_json(args.static_order)
    if (order.get("schema_signature") != schema_signature(schema)
            or order.get("evaluation_holdout_labels_used") is not False
            or sorted(order["order"]) != list(range(schema.num_groups))):
        raise ValueError("requires training-only, schema-bound static order")
    first = order["order"][0]
    candidates = tuple(g for g in range(schema.num_groups) if g != first)
    source_rows = read_jsonl(cache / "responses.jsonl")
    if len(source_rows) != manifest["num_responses"] or not source_rows:
        raise ValueError("cache row count mismatch")
    if len({row["sample_id"] for row in source_rows}) != len(source_rows):
        raise ValueError("duplicate sample ID")
    rows = []
    for source in source_rows:
        if source["split"] != "validation" or source["response_source"] != "automatic_model":
            raise ValueError("wrong cache row split/source")
        answers, y = tuple(source["z"]), source["y"]
        schema.validate_state(answers, complete=True)
        if type(y) is not int or not 0 <= y < schema.num_classes:
            raise ValueError("invalid label")
        rows.append({"group_id": source["group_id"],
                     "outcome": outcome_key(answers, schema, first),
                     "answers": answers, "y": y, "ce": {}, "correct": {}})
    # No hidden answer enters the controller: all counterfactual responses and
    # labels are used only to fit and score policies across held-out halves.
    pending = []
    for index, row in enumerate(rows):
        for second in candidates:
            mask = tuple(g == first or g == second for g in range(schema.num_groups))
            pending.append((index, second, mask_answers(row["answers"], mask, schema)))
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset + args.batch_size]
        probs = head.probabilities_many([entry[2] for entry in batch])
        for (index, second, _), p in zip(batch, probs):
            y = rows[index]["y"]
            rows[index]["ce"][second] = -math.log(max(p[y], 1e-12))
            rows[index]["correct"][second] = int(max(range(len(p)), key=p.__getitem__) == y)
    report = analyze(rows, candidates, smoothing=args.smoothing, seed=args.seed)
    report.update({"format": "cbmjev-conditional-second-query-diagnostic-v1",
                   "evidence_status": "EXPLORATORY_VALIDATION_SPLIT_FIT_NOT_PAPER_OR_TEST_EVIDENCE",
                   "regime": "two_singleton_queries_automatic_responses_fixed_final_head",
                   "first_group": first, "first_group_id": schema.groups[first].id,
                   "seed": args.seed, "smoothing": args.smoothing,
                   "selection_loss": "unweighted_cross_entropy",
                   "samples": len(rows), "cache_manifest_sha256": file_hash(cache / "manifest.json"),
                   "head_receipt_sha256": file_hash(Path(args.head) / "receipt.json"),
                   "head_artifact_id": head_report["head_artifact_id"],
                   "static_order_sha256": file_hash(args.static_order),
                   "caveats": ["Validation labels fit conditional tables; not locked evidence.",
                               "Only the second query after a fixed first query is adaptive.",
                               "A fixed learned head and automatic concept outputs define this measurement regime."]})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "aggregate": report["aggregate"]}, sort_keys=True))


if __name__ == "__main__":
    main()

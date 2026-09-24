#!/usr/bin/env python3
"""Bounded training-only diagnostic of the one-step Choice teacher at empty states."""

import argparse
import json
import math
from pathlib import Path

from cbmjev.contracts import Schema, stable_hash
from cbmjev.io import file_hash, read_json, write_json


def diagnose(schema_path, shard_path, *, max_histories, cost_weight, temperature):
    if max_histories < 1 or cost_weight < 0 or temperature <= 0:
        raise ValueError("invalid diagnostic settings")
    schema = Schema.from_dict(read_json(schema_path))
    seen = empty = stop_correct = any_singleton_correct = stop_argmin = 0
    teacher_sum = [0.] * (schema.num_groups + 1)
    risk_sum = [0.] * (schema.num_groups + 1)
    current = []

    def finish(rows):
        nonlocal seen, empty, stop_correct, any_singleton_correct, stop_argmin
        seen += 1
        observed = tuple(rows[0]["observed"])
        schema.validate_state(observed)
        if any(tuple(row["observed"]) != observed for row in rows):
            raise ValueError("decision state changed within occurrence")
        if observed != schema.empty_state():
            return
        empty += 1
        singleton = {tuple(row["action"]): row["target"] for row in rows
                     if len(row["action"]) <= 1}
        required = [()] + [(group,) for group in range(schema.num_groups)]
        if len(singleton) != schema.num_groups + 1 or any(key not in singleton for key in required):
            raise ValueError("incomplete empty-state singleton Choice set")
        errors = [singleton[key] for key in required]
        if any(type(error) not in (int, float) or error not in (0, 1) for error in errors):
            raise ValueError("teacher requires realized Boolean errors")
        stop_correct += errors[0] == 0
        any_singleton_correct += any(error == 0 for error in errors[1:])
        stop_argmin += errors[0] == 0 or all(error == 1 for error in errors[1:])
        utilities = [errors[0]] + [error + cost_weight for error in errors[1:]]
        minimum = min(utilities)
        weights = [math.exp(-(utility - minimum) / temperature) for utility in utilities]
        total = sum(weights)
        for index, (error, weight) in enumerate(zip(errors, weights)):
            risk_sum[index] += error
            teacher_sum[index] += weight / total

    with Path(shard_path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["action"] == [] and current:
                finish(current)
                if seen >= max_histories:
                    break
                current = []
            current.append(row)
        else:
            if current and seen < max_histories:
                finish(current)
    if not empty:
        raise ValueError("no empty-state occurrences in bounded sample")
    ranking_q = sorted(range(schema.num_groups + 1), key=lambda i: (-teacher_sum[i], i))
    ranking_risk = sorted(range(schema.num_groups + 1), key=lambda i: (risk_sum[i], i))
    def name(index):
        return "STOP" if index == 0 else str(index - 1)
    result = {"format": "cbmjev-choice-empty-teacher-bounded-diagnostic-v1",
        "evidence_status": "TRAINING_ONLY_ORDERED_SHARD_SAMPLE_NOT_PAPER_EVIDENCE",
        "paper_evidence": False, "schema_hash": schema.hash,
        "shard": str(Path(shard_path).resolve()), "shard_sha256": file_hash(shard_path),
        "max_histories": max_histories, "histories_scanned": seen,
        "empty_histories": empty, "cost_weight": cost_weight,
        "temperature": temperature, "stop_correct_fraction": stop_correct / empty,
        "any_singleton_correct_fraction": any_singleton_correct / empty,
        "stop_argmin_fraction": stop_argmin / empty,
        "mean_soft_teacher_stop_probability": teacher_sum[0] / empty,
        "teacher_top5": [{"action": name(i), "mean_probability": teacher_sum[i] / empty}
                         for i in ranking_q[:5]],
        "realized_error_best5": [{"action": name(i), "mean_error": risk_sum[i] / empty}
                                 for i in ranking_risk[:5]],
        "teacher_top_action_is_lowest_mean_error_action": ranking_q[0] == ranking_risk[0],
        "interpretation_limit": "First ordered histories in one training OOF shard, not a random sample or validation result. Per-example STOP argmin frequency is not the same as mean soft teacher STOP probability; this does not prove learned-policy stopping behavior."}
    result["report_hash"] = stable_hash(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-histories", type=int, default=1000)
    parser.add_argument("--cost-weight", type=float, default=.03)
    parser.add_argument("--temperature", type=float, default=.5)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise ValueError("output must be new")
    result = diagnose(args.schema, args.shard, max_histories=args.max_histories,
        cost_weight=args.cost_weight, temperature=args.temperature)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print({key: result[key] for key in ("histories_scanned", "empty_histories",
        "stop_argmin_fraction", "mean_soft_teacher_stop_probability",
        "teacher_top_action_is_lowest_mean_error_action")}, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Training-OOF static first query, then frozen multi-step risk policy.

Diagnostic intervention only: no retraining and no validation-label policy fit.
"""
import argparse
import json
import math
from pathlib import Path

from cbmjev.contracts import stable_hash
from cbmjev.crossfit_artifacts import load_crossfit_head
from cbmjev.crossfit_evaluation import _load_sources, _verify_binding_files_unchanged
from cbmjev.io import file_hash, fresh_dir, write_json, write_jsonl
from cbmjev.learning import mask_answers
from cbmjev.pipeline import code_fingerprint
from scripts.evaluate_oof_multistep_conditional_risk import choose_next, load_model, load_order


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prepared", "planned", "merged", "responder", "cache",
                "static-order", "out"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--fold-models", nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    args.budget = 16
    if args.batch_size < 1 or Path(args.static_order).name != "static_order.json":
        raise ValueError("invalid batch size or persisted static order")
    out = Path(args.out)
    if out.exists():
        raise ValueError("refuse to overwrite output")
    prepared, planned, merged, responder, cache = map(Path,
        (args.prepared, args.planned, args.merged, args.responder, args.cache))
    source_hash = code_fingerprint()
    script_hash = file_hash(__file__)
    schema, rows, _, head, _, binding = _load_sources(
        prepared, planned, merged, responder, cache, args.device,
        static_order_dir=Path(args.static_order).parent)
    _, head_report = load_crossfit_head(merged / "head", schema, device=args.device)
    order = load_order(args.static_order, schema)
    loaded = [load_model(path, args, schema) for path in args.fold_models]
    models, reports = zip(*loaded)
    if (len(reports) != reports[0]["outer_folds"] or
            sorted(r["outer_fold"] for r in reports) != list(range(len(reports))) or
            any(r["seed"] != args.seed + r["outer_fold"] for r in reports) or
            len({json.dumps({k: v for k, v in r["config"].items()
                             if k != "device"}, sort_keys=True) for r in reports}) != 1):
        raise ValueError("fold model configuration mismatch")
    if set().union(*(set(r["binding"]["group_ids"]) for r in reports)) & {
            row["group_id"] for row in rows}:
        raise ValueError("validation group seen in policy fitting")
    model_hashes = [file_hash(Path(p) / "risk.pt") for p in args.fold_models]
    receipt_hashes = [file_hash(Path(p) / "receipt.json") for p in args.fold_models]
    fixed_mask = tuple(g in set(order[:16]) for g in range(schema.num_groups))
    records, dynamic_states, fixed_states = [], [], []
    for row in rows:
        answers = tuple(row["z"])
        schema.validate_state(answers, complete=True)
        state = schema.empty_state()
        queried = []
        for step in range(16):
            if step == 0:
                action = order[0]
            else:
                index, available = choose_next(state, models, schema, args.device)
                action = available[index]
            next_state = list(state)
            for atom in schema.groups[action].atoms:
                next_state[atom] = answers[atom]
            state = tuple(next_state)
            queried.append(action)
        dynamic_states.append(state)
        fixed_states.append(mask_answers(answers, fixed_mask, schema))
        records.append({"sample_id": row["sample_id"], "group_id": row["group_id"],
                        "y": row["y"], "dynamic_order": queried})
    for start in range(0, len(rows), args.batch_size):
        batch = records[start:start + args.batch_size]
        dynamic_p = head.probabilities_many(dynamic_states[start:start + args.batch_size])
        fixed_p = head.probabilities_many(fixed_states[start:start + args.batch_size])
        for item, dp, fp in zip(batch, dynamic_p, fixed_p):
            y = item["y"]
            item.update(dynamic_ce=-math.log(max(dp[y], 1e-12)),
                        fixed_ce=-math.log(max(fp[y], 1e-12)),
                        dynamic_correct=int(max(range(len(dp)), key=dp.__getitem__) == y),
                        fixed_correct=int(max(range(len(fp)), key=fp.__getitem__) == y))
    n = len(records)
    metrics = {"samples": n, "budget_groups": 16,
               "dynamic_accuracy": sum(r["dynamic_correct"] for r in records) / n,
               "fixed_accuracy": sum(r["fixed_correct"] for r in records) / n,
               "dynamic_ce": math.fsum(r["dynamic_ce"] for r in records) / n,
               "fixed_ce": math.fsum(r["fixed_ce"] for r in records) / n,
               "changed_set_rate": sum(set(r["dynamic_order"]) != set(order[:16])
                                       for r in records) / n}
    metrics["delta_accuracy"] = metrics["dynamic_accuracy"] - metrics["fixed_accuracy"]
    metrics["delta_ce"] = metrics["dynamic_ce"] - metrics["fixed_ce"]
    if (code_fingerprint() != source_hash or file_hash(__file__) != script_hash or
            any(file_hash(Path(p) / "receipt.json") != h
                for p, h in zip(args.fold_models, receipt_hashes)) or
            any(file_hash(Path(p) / "risk.pt") != h
                for p, h in zip(args.fold_models, model_hashes))):
        raise ValueError("model or source changed during inference")
    _verify_binding_files_unchanged(binding, merged, responder, cache,
                                    Path(args.static_order).parent, planned_dir=planned)
    out = fresh_dir(out)
    write_jsonl(out / "traces.jsonl", records)
    report = {"format": "cbmjev-oof-multistep-static-first-validation-v1",
              "status": "COMPLETE", "split": "validation", "test_evaluated": False,
              "method": "training_oof_static_first_then_frozen_conditional_risk",
              "selection_uses_only_acquired_automatic_concepts": True,
              "validation_labels_used_for_fit": False, "seed": args.seed,
              "static_first_group": order[0], "static_order_prefix": list(order[:16]),
              "metrics": metrics, "fold_receipt_sha256": receipt_hashes,
              "fold_checkpoint_sha256": model_hashes,
              "final_head_artifact_id": head_report["head_artifact_id"],
              "source_binding": binding, "source_code_hash": source_hash,
              "script_sha256": script_hash, "traces_sha256": file_hash(out / "traces.jsonl"),
              "evidence_status": "DEVELOPMENT_VALIDATION_FIRST_ACTION_INTERVENTION",
              "caveat": "Fixing first action changes subsequent observed states; the delta is not an isolated causal estimate of only first-action error."}
    report["report_hash"] = stable_hash(report)
    write_json(out / "receipt.json", report)
    print(json.dumps({"out": str(out), "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()

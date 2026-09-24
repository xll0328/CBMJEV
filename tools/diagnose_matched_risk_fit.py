#!/usr/bin/env python3
"""Source-bound, validation-only fit diagnostic for paired scalar risk heads.

Empirical state/action probabilities are optimistic in-sample fit references,
NOT a Bayes noise floor. No model is trained and no test events are scored.
"""
import argparse
from collections import Counter, defaultdict
from itertools import islice
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbmjev.config import learning_config
from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, read_json, write_json
from cbmjev.learning import (action_targets, iter_risk_training_examples, normalize_config,
                            policy_examples, training_rows)
from cbmjev.matched_risk import matched_event_manifest
from cbmjev.nanojev import RiskTrainingExample, load_nano_head, risk_prompts
from cbmjev.pipeline import code_fingerprint, load_model_bundle, load_matched_mlp_controller
from cbmjev.provenance import validate_target_exclusion


def reconstruct_training_events(rows, head, schema, config, receipt):
    """Replay the exact training reservoir, not a new prefix/subsample."""
    if receipt.get("sampling") != "seeded_reservoir":
        raise ValueError("unsupported training sampling protocol")
    cap = receipt["retained_examples"]
    if type(cap) is not int or cap < 1 or receipt["seed"] != config["seed"]:
        raise ValueError("invalid retained size/seed")
    rng = random.Random(config["seed"])
    events, seen = [], 0
    for seen, row in enumerate(iter_risk_training_examples(rows, head, schema, learning_config(config)), 1):
        event = RiskTrainingExample(tuple(row["observed"]), tuple(row["action"]), float(row["error"]))
        if len(events) < cap:
            events.append(event)
        else:
            index = rng.randrange(seen)
            if index < cap:
                events[index] = event
    events = tuple(events)
    if seen != receipt["available_examples"] or matched_event_manifest(events, schema)["event_sha256"] != receipt["event_sha256"]:
        raise ValueError("reconstructed training events differ from recorded digest/count")
    return events


def validation_events(rows, head, schema, config, cap):
    cfg = normalize_config({**learning_config(config), "objective": "risk"}, schema)
    examples = policy_examples(training_rows(rows, "validation", schema), schema, cfg, epoch=0)
    iterator = islice(examples, cap)
    result = []
    while True:
        batch = list(islice(iterator, cfg["batch_size"]))
        if not batch:
            break
        targets = action_targets(head, batch, "risk").cpu().tolist()
        result.extend(RiskTrainingExample(tuple(before), tuple(action), float(y), "validation")
                      for (before, action, _, _), y in zip(batch, targets))
    return tuple(result)


def probabilities(events):
    buckets = defaultdict(list)
    for e in events:
        buckets[(e.observed, e.action)].append(e.error)
    return {key: sum(values) / len(values) for key, values in buckets.items()}


def losses(events, scores):
    if len(events) != len(scores):
        raise ValueError("prediction/event count mismatch")
    if not events:
        return {"n": 0, "bce": None, "brier": None}
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ValueError("risk probabilities must be finite and in [0,1]")
    if any(not math.isfinite(e.error) or not 0 <= e.error <= 1 for e in events):
        raise ValueError("event targets must be finite and in [0,1]")
    eps = 1e-7
    bce = sum(-e.error * math.log(max(eps, min(1 - eps, p)))
              -(1-e.error) * math.log(max(eps, min(1-eps, 1-p))) for e, p in zip(events, scores))
    return {"n": len(events), "bce": bce / len(events),
            "brier": sum((p-e.error)**2 for e, p in zip(events, scores)) / len(events)}


def diagnose_events(events, predictions, training_events):
    """Evaluate all/STOP/nonSTOP; reference fits never consume model predictions."""
    if not events or not training_events:
        raise ValueError("nonempty diagnostic and training events required")
    losses(training_events, [0.5] * len(training_events))
    for name, scores in predictions.items():
        losses(events, scores)
    results = {}
    for section, accept in (("all", lambda e: True), ("stop", lambda e: not e.action),
                            ("nonstop", lambda e: bool(e.action))):
        ids = [i for i, e in enumerate(events) if accept(e)]
        selected = tuple(events[i] for i in ids)
        train = tuple(e for e in training_events if accept(e))
        buckets = Counter((e.observed, e.action) for e in selected)
        record = {"n": len(selected), "state_action_buckets": len(buckets),
                  "singleton_buckets": sum(n == 1 for n in buckets.values()),
                  "singleton_events": sum(n for n in buckets.values() if n == 1),
                  "bucket_size_histogram": dict(sorted(Counter(buckets.values()).items())),
                  "models": {name: losses(selected, [scores[i] for i in ids])
                             for name, scores in predictions.items()}, "references": {}}
        if selected:
            local = probabilities(selected)
            global_mean = sum(e.error for e in selected) / len(selected)
            record["mean_target"] = global_mean
            record["references"]["global_insample"] = losses(selected, [global_mean] * len(selected))
            record["references"]["state_action_insample"] = losses(selected, [local[(e.observed,e.action)] for e in selected])
        if selected and train:
            fitted = probabilities(train)
            mean_train = sum(e.error for e in train) / len(train)
            record["references"]["train_global"] = losses(selected, [mean_train] * len(selected))
            record["references"]["train_state_action_fallback_global"] = losses(selected,
                                [fitted.get((e.observed,e.action), mean_train) for e in selected])
            record["unseen_in_training_bucket_events"] = sum((e.observed,e.action) not in fitted for e in selected)
        else:
            record["unseen_in_training_bucket_events"] = len(selected)
        results[section] = record
    return results


def score_events(events, schema, mlp, scorer, batch_size):
    import torch
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive inference batch size required")
    keys = list(dict.fromkeys((e.observed, e.action) for e in events))
    scored = {"matched_mlp": {}, "nano_scalar_adapter": {}}
    for start in range(0, len(keys), batch_size):
        batch = keys[start:start+batch_size]
        # Targets, IDs, raw inputs and unobserved measurements never enter forward.
        by_state = defaultdict(list)
        for state, action in batch:
            by_state[state].append(action)
        for state, actions in by_state.items():
            values = mlp.predict(state, actions)
            if len(values) != len(actions):
                raise ValueError("MLP output count mismatch")
            scored["matched_mlp"].update({(state,a): p for a,p in zip(actions,values)})
        prompts = [risk_prompts(schema, state, [action])[0] for state,action in batch]
        with torch.inference_mode():
            values = scorer.score_prompts(prompts).sigmoid().cpu().tolist()
        if len(values) != len(batch):
            raise ValueError("Nano output count mismatch")
        scored["nano_scalar_adapter"].update(dict(zip(batch,values)))
    return {name: [lookup[(e.observed,e.action)] for e in events] for name,lookup in scored.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "cache", "nano-controller", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-validation-examples", type=int, default=20000)
    args = p.parse_args(argv)
    if Path(args.out).exists() or min(args.batch_size, args.max_validation_examples) < 1:
        raise ValueError("new output and positive sizes required")
    paired = Path(args.nano_controller)
    def identity():
        return {"core_source": code_fingerprint(), "diagnostic_source": file_hash(__file__),
                "paired": {p.name: file_hash(p) for p in paired.iterdir() if p.is_file()},
                "parent": {p.name: file_hash(p) for p in Path(args.models).iterdir() if p.is_file()},
                "cache": {p.name: file_hash(p) for p in Path(args.cache).iterdir() if p.is_file()}}
    before = identity()
    schema, rows, _, cfg, parent, head, _, _ = load_model_bundle(args.models, args.cache, args.device)
    mlp, _ = load_matched_mlp_controller(paired, args.models, args.cache, device=args.device)
    receipt = read_json(paired / "training.json")
    for split in ("policy_fit", "validation"):
        validate_target_exclusion(parent["provenance"], artifact_ids=[parent["head_artifact_id"]],
                                  target_group_ids=sorted({r["group_id"] for r in rows if r["split"] == split}))
    validate_target_exclusion(parent["provenance"], artifact_ids=[parent["controller_artifact_id"]],
                              target_group_ids=sorted({r["group_id"] for r in rows if r["split"] == "validation"}))
    train = reconstruct_training_events(rows, head, schema, cfg, receipt)
    validation = validation_events(rows, head, schema, cfg, args.max_validation_examples)
    scorer = load_nano_head(paired / "nano_head.pt", schema, device=args.device, expected_task="risk")
    scorer.eval()
    results = {split: diagnose_events(events, score_events(events, schema, mlp, scorer, args.batch_size), train)
               for split, events in (("policy_fit", train), ("validation", validation))}
    if identity() != before:
        raise ValueError("diagnostic source or inputs changed during scoring")
    report = {"scope": "DEVELOPMENT_FIT_DIAGNOSTIC_NOT_TEST_OR_PAPER_EVIDENCE", "results": results,
              "training_event_sha256_verified": receipt["event_sha256"], "source_identity": before,
              "validation_event_sha256": stable_hash([vars(e) for e in validation]),
              "validation_sampling": "deterministic epoch-zero event prefix, not iid cases or deployment trajectories",
              "max_validation_examples": args.max_validation_examples, "batch_size": args.batch_size,
              "forward_scope": "unique observed-state/action inputs scored in eval mode then expanded to original event multiplicities",
              "bce_probability_clip_epsilon": 1e-7, "aggregation": "equal event weight; masks/actions repeat cases",
              "reference_scope": "in-sample empirical means are optimistic references, NOT Bayes noise floors; train references fit separately per STOP/nonSTOP slice",
              "test_evaluated": False, "paper_evidence": False}
    write_json(args.out, report)
    print({"out": args.out, "training_events": len(train), "validation_events": len(validation)})


if __name__ == "__main__":
    main()

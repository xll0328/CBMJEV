#!/usr/bin/env python3
"""Fold-safe, non-JEV multi-step conditional-risk baseline for CUB.

Fit one state/action -> post-query terminal CE regressor per outer fold, using
only that fold's automatic held-out responses and excluded task head. At
validation, average fold predictions and greedily acquire exactly K groups.
This is a development diagnostic, not a calibrated Bayes-risk or JEV model.
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from cbmjev.contracts import stable_hash
from cbmjev.crossfit_artifacts import load_crossfit_head
from cbmjev.crossfit_cache import load_crossfit_cache
from cbmjev.crossfit_evaluation import _load_sources, _verify_binding_files_unchanged
from cbmjev.crossfit_training import _merge_provenance
from cbmjev.io import file_hash, fresh_dir, read_json, write_json, write_jsonl
from cbmjev.learning import encode_actions, encode_states, mask_answers, schema_signature
from cbmjev.pipeline import code_fingerprint
from cbmjev.provenance import validate_target_exclusion


class ConditionalRisk(nn.Module):
    def __init__(self, schema, hidden):
        super().__init__()
        width = sum(schema.num_categories) + schema.num_atoms + schema.num_groups + 1
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, features):
        return self.net(features).flatten()


def state_action_features(states, actions, schema, device):
    if len(states) != len(actions):
        raise ValueError("states/actions length mismatch")
    if any(schema.group_mask(s)[a] for s, a in zip(states, actions)):
        raise ValueError("candidate already observed")
    return torch.cat((encode_states(states, schema, device),
                      encode_actions([(a,) for a in actions], schema, device)), dim=1)


def sampled_events(rows, schema, static_order, seed, candidates_per_state):
    """Sample masks/actions without looking at hidden responses or targets."""
    rng = random.Random(seed)
    events = []
    for row in rows:
        answers = tuple(row["z"])
        schema.validate_state(answers, complete=True)
        if type(row["y"]) is not int or not 0 <= row["y"] < schema.num_classes:
            raise ValueError("invalid target")
        cardinality = rng.randrange(16)
        masks = [set(static_order[:cardinality]),
                 set(rng.sample(range(schema.num_groups), cardinality))]
        for visible in masks:
            before = mask_answers(answers,
                tuple(g in visible for g in range(schema.num_groups)), schema)
            remaining = [g for g in range(schema.num_groups) if g not in visible]
            chosen = rng.sample(remaining, min(candidates_per_state, len(remaining)))
            next_static = next(g for g in static_order if g not in visible)
            if next_static not in chosen:
                chosen.append(next_static)
            for action in chosen:
                after = list(before)
                for atom in schema.groups[action].atoms:
                    after[atom] = answers[atom]
                events.append((before, action, tuple(after), row["y"]))
    return events


def fit_fold(rows, head, schema, order, *, seed, hidden, epochs, batch_size,
             candidates_per_state, device):
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate training sample ID")
    torch.manual_seed(seed)
    events = sampled_events(rows, schema, order, seed, candidates_per_state)
    if not events:
        raise ValueError("empty training events")
    inputs, targets = [], []
    head.network.eval()
    for start in range(0, len(events), batch_size):
        batch = events[start:start + batch_size]
        before, actions, after, labels = zip(*batch)
        inputs.append(state_action_features(before, actions, schema, "cpu"))
        with torch.no_grad():
            logits = head.network(encode_states(after, schema, head.device))
            y = torch.tensor(labels, dtype=torch.long, device=head.device)
            targets.append(F.cross_entropy(logits, y, reduction="none").cpu())
    x, y = torch.cat(inputs), torch.cat(targets)
    if not torch.isfinite(y).all():
        raise ValueError("nonfinite terminal-risk targets")
    model = ConditionalRisk(schema, hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(seed + 1000)
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(len(y), generator=generator)
        total = 0.0
        model.train()
        for ids in perm.split(batch_size):
            pred = model(x[ids].to(device))
            loss = F.mse_loss(pred, y[ids].to(device))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite conditional-risk loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        history.append({"epoch": epoch, "mse": total / len(y)})
    model.eval().requires_grad_(False)
    return model, {"training_rows": len(rows), "events": len(events),
                   "target_ce_mean": float(y.mean()), "target_ce_max": float(y.max()),
                   "history": history}


def fold_sources(prepared, planned, merged, fold, device):
    receipt = read_json(merged / "receipt.json")
    sources = sorted(receipt["sources"], key=lambda s: s["outer_fold"])
    if [s["outer_fold"] for s in sources] != list(range(len(sources))):
        raise ValueError("incomplete outer folds")
    source = sources[fold]
    outer = Path(source["directory"])
    schema, rows, manifest = load_crossfit_cache(
        outer / "outer/cache", prepared, planned, outer / "outer/responder")
    head, head_report = load_crossfit_head(outer / "head", schema, device=device)
    groups = sorted({row["group_id"] for row in rows})
    if (not rows or manifest["response_source"] != "automatic_model" or
            manifest["crossfit"]["outer_fold"] != fold or
            head_report["head_artifact_id"] != read_json(outer / "receipt.json")["head_artifact_id"] or
            set(groups) & set(head_report["fit_group_ids"])):
        raise ValueError("invalid fold-excluded automatic response/head pair")
    validate_target_exclusion(
        _merge_provenance(head_report["provenance"], manifest["provenance"]),
        artifact_ids=[head_report["head_artifact_id"],
                      manifest["responder_artifact_id"]], target_group_ids=groups)
    binding = {"fold": fold, "rows": len(rows), "group_ids": groups,
               "outer_receipt_sha256": file_hash(outer / "receipt.json"),
               "cache_manifest_sha256": file_hash(outer / "outer/cache/manifest.json"),
               "head_receipt_sha256": file_hash(outer / "head/receipt.json"),
               "head_artifact_id": head_report["head_artifact_id"],
               "merged_receipt_sha256": file_hash(merged / "receipt.json")}
    return schema, rows, head, binding, len(sources)


def load_order(path, schema):
    static = read_json(path)
    if (static.get("schema_signature") != schema_signature(schema) or
            static.get("fit_scope") != "ALL_OBSERVED_OUTER_OOF_TRAIN_TARGETS" or
            static.get("evaluation_holdout_labels_used") is not False or
            sorted(static["order"]) != list(range(schema.num_groups))):
        raise ValueError("invalid training-only static order")
    return tuple(static["order"])


def train(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError("refuse to overwrite fold output")
    prepared, planned, merged = map(Path, (args.prepared, args.planned, args.merged))
    source_hash = code_fingerprint()
    schema, rows, head, binding, n_folds = fold_sources(
        prepared, planned, merged, args.outer_fold, args.device)
    if not 1 <= args.budget <= schema.num_groups or args.budget != 16:
        raise ValueError("this diagnostic currently requires exact K16")
    order = load_order(args.static_order, schema)
    model, metrics = fit_fold(rows, head, schema, order,
        seed=args.seed, hidden=args.hidden, epochs=args.epochs,
        batch_size=args.batch_size, candidates_per_state=args.candidates_per_state,
        device=args.device)
    refreshed = fold_sources(prepared, planned, merged, args.outer_fold, args.device)[3]
    if refreshed != binding or code_fingerprint() != source_hash:
        raise ValueError("sources changed during training")
    out = fresh_dir(out)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "risk.pt")
    report = {"format": "cbmjev-oof-multistep-risk-fold-v1", "status": "COMPLETE",
              "method": "non_jev_conditional_terminal_ce_mlp", "outer_fold": args.outer_fold,
              "outer_folds": n_folds, "seed": args.seed, "schema_hash": schema.hash,
              "config": {"budget": args.budget, "hidden": args.hidden, "epochs": args.epochs,
                         "batch_size": args.batch_size,
                         "candidates_per_state": args.candidates_per_state,
                         "device": args.device, "learning_rate": .001},
              "metrics": metrics, "binding": binding,
              "static_order_sha256": file_hash(args.static_order),
              "source_code_hash": source_hash, "script_sha256": file_hash(__file__),
              "checkpoint_sha256": file_hash(out / "risk.pt"),
              "evidence_status": "FOLD_TRAINING_NOT_VALIDATION_RESULT"}
    report["report_hash"] = stable_hash(report)
    write_json(out / "receipt.json", report)
    print(json.dumps({"out": str(out), "events": metrics["events"],
                      "final_mse": metrics["history"][-1]["mse"]}), flush=True)


def load_model(path, args, schema):
    path = Path(path)
    report = read_json(path / "receipt.json")
    core = dict(report)
    if (core.pop("report_hash", None) != stable_hash(core) or
            report.get("format") != "cbmjev-oof-multistep-risk-fold-v1" or
            report.get("status") != "COMPLETE" or
            report["schema_hash"] != schema.hash or
            report["source_code_hash"] != code_fingerprint() or
            report["script_sha256"] != file_hash(__file__) or
            report["checkpoint_sha256"] != file_hash(path / "risk.pt") or
            report["static_order_sha256"] != file_hash(args.static_order) or
            report["config"]["budget"] != args.budget):
        raise ValueError("invalid fold model receipt or source drift")
    refreshed = fold_sources(Path(args.prepared), Path(args.planned),
                             Path(args.merged), report["outer_fold"], "cpu")[3]
    if report["binding"] != refreshed:
        raise ValueError("fold training source changed")
    model = ConditionalRisk(schema, report["config"]["hidden"]).to(args.device)
    model.load_state_dict(torch.load(path / "risk.pt", map_location="cpu",
                                     weights_only=True), strict=True)
    model.eval().requires_grad_(False)
    return model, report


@torch.no_grad()
def choose_next(state, models, schema, device):
    available = [g for g, seen in enumerate(schema.group_mask(state)) if not seen]
    if not available:
        raise ValueError("no available query")
    features = state_action_features([state] * len(available), available, schema, device)
    scores = torch.stack([model(features) for model in models]).mean(0)
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite risk score")
    return min(range(len(available)), key=lambda i: (float(scores[i]), available[i])), available


def evaluate(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError("refuse to overwrite validation output")
    prepared, planned, merged = map(Path, (args.prepared, args.planned, args.merged))
    if Path(args.static_order).name != "static_order.json":
        raise ValueError("static order must be the persisted crossfit artifact")
    schema, rows, _, head, _, source_binding = _load_sources(
        prepared, planned, merged, Path(args.responder), Path(args.cache),
        args.device, static_order_dir=Path(args.static_order).parent)
    _, head_report = load_crossfit_head(merged / "head", schema, device=args.device)
    order = load_order(args.static_order, schema)
    loaded = [load_model(path, args, schema) for path in args.fold_models]
    models, reports = zip(*loaded)
    if (sorted(r["outer_fold"] for r in reports) != list(range(len(reports))) or
            len(reports) != reports[0]["outer_folds"] or
            len({r["config"]["hidden"] for r in reports}) != 1):
        raise ValueError("missing, duplicate or incompatible fold models")
    train_groups = set().union(*(set(r["binding"]["group_ids"]) for r in reports))
    val_groups = {row["group_id"] for row in rows}
    if train_groups & val_groups:
        raise ValueError("validation groups seen in policy training")
    source_hash = code_fingerprint()
    fold_receipt_hashes = [file_hash(Path(p) / "receipt.json") for p in args.fold_models]
    fixed_mask = tuple(g in set(order[:args.budget]) for g in range(schema.num_groups))
    records, static_states, dynamic_states = [], [], []
    for row in rows:
        answers = tuple(row["z"])
        schema.validate_state(answers, complete=True)
        state, queried = schema.empty_state(), []
        for _ in range(args.budget):
            index, available = choose_next(state, models, schema, args.device)
            action = available[index]
            next_state = list(state)
            for atom in schema.groups[action].atoms:
                # The automatic answer is exposed only after this action.
                next_state[atom] = answers[atom]
            state = tuple(next_state)
            queried.append(action)
        dynamic_states.append(state)
        static_states.append(mask_answers(answers, fixed_mask, schema))
        records.append({"sample_id": row["sample_id"], "group_id": row["group_id"],
                        "y": row["y"], "dynamic_order": queried})
    for start in range(0, len(rows), args.batch_size):
        batch = records[start:start + args.batch_size]
        dynamic_p = head.probabilities_many(dynamic_states[start:start + args.batch_size])
        fixed_p = head.probabilities_many(static_states[start:start + args.batch_size])
        for item, dp, fp in zip(batch, dynamic_p, fixed_p):
            y = item["y"]
            item.update(dynamic_ce=-math.log(max(dp[y], 1e-12)),
                        fixed_ce=-math.log(max(fp[y], 1e-12)),
                        dynamic_correct=int(max(range(len(dp)), key=dp.__getitem__) == y),
                        fixed_correct=int(max(range(len(fp)), key=fp.__getitem__) == y))
    n = len(records)
    metrics = {"samples": n, "budget_groups": args.budget,
               "dynamic_accuracy": sum(r["dynamic_correct"] for r in records) / n,
               "fixed_accuracy": sum(r["fixed_correct"] for r in records) / n,
               "dynamic_ce": math.fsum(r["dynamic_ce"] for r in records) / n,
               "fixed_ce": math.fsum(r["fixed_ce"] for r in records) / n,
               "changed_set_rate": sum(set(r["dynamic_order"]) != set(order[:args.budget])
                                       for r in records) / n}
    metrics["delta_accuracy"] = metrics["dynamic_accuracy"] - metrics["fixed_accuracy"]
    metrics["delta_ce"] = metrics["dynamic_ce"] - metrics["fixed_ce"]
    if code_fingerprint() != source_hash or any(
        file_hash(Path(path) / "receipt.json") != digest
        for path, digest in zip(args.fold_models, fold_receipt_hashes)):
        raise ValueError("source changed during evaluation")
    _verify_binding_files_unchanged(source_binding, merged, Path(args.responder),
                                    Path(args.cache), Path(args.static_order).parent,
                                    planned_dir=planned)
    out = fresh_dir(out)
    write_jsonl(out / "traces.jsonl", records)
    report = {"format": "cbmjev-oof-multistep-risk-validation-v1",
              "status": "COMPLETE", "split": "validation", "test_evaluated": False,
              "method": "non_jev_greedy_conditional_terminal_ce_mlp",
              "selection_uses_only_acquired_automatic_concepts": True,
              "validation_labels_used_for_fit": False, "seed": args.seed,
              "budget": args.budget, "static_order_prefix": list(order[:args.budget]),
              "metrics": metrics, "fold_receipt_sha256": fold_receipt_hashes,
              "final_head_artifact_id": head_report["head_artifact_id"],
              "source_binding": source_binding,
              "final_head_receipt_sha256": file_hash(merged / "head/receipt.json"),
              "validation_cache_manifest_sha256": file_hash(Path(args.cache) / "manifest.json"),
              "static_order_sha256": file_hash(args.static_order),
              "source_code_hash": source_hash, "script_sha256": file_hash(__file__),
              "traces_sha256": file_hash(out / "traces.jsonl"),
              "evidence_status": "DEVELOPMENT_VALIDATION_NOT_LOCKED_TEST_OR_JEV_GAIN",
              "caveats": ["One-step terminal CE target trained on offline complete responses; greedy replanning is not a Bellman-optimal policy.",
                          "Fold training heads differ from the final validation head.",
                          "Validation reused during research; no confirmatory p-value or speed claim."]}
    report["report_hash"] = stable_hash(report)
    write_json(out / "receipt.json", report)
    print(json.dumps({"out": str(out), "metrics": metrics}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train-fold", "evaluate"):
        p = sub.add_parser(command)
        for key in ("prepared", "planned", "merged", "responder", "cache",
                    "static-order", "out"):
            p.add_argument("--" + key, required=True)
        p.add_argument("--budget", type=int, default=16)
        p.add_argument("--seed", type=int, default=60)
        p.add_argument("--batch-size", type=int, default=256)
        p.add_argument("--device", default="cpu")
    train_parser = sub.choices["train-fold"]
    train_parser.add_argument("--outer-fold", type=int, required=True)
    train_parser.add_argument("--hidden", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--candidates-per-state", type=int, default=8)
    sub.choices["evaluate"].add_argument("--fold-models", nargs="+", required=True)
    args = parser.parse_args(argv)
    if (args.budget != 16 or args.batch_size < 1 or
            (args.command == "train-fold" and
             (args.hidden < 1 or args.epochs < 1 or args.candidates_per_state < 1))):
        raise ValueError("invalid K16 configuration")
    train(args) if args.command == "train-fold" else evaluate(args)


if __name__ == "__main__":
    main()

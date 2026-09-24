#!/usr/bin/env python3
"""Fold-safe BRiG-AFA adaptation on the CUB nested-OOF validation protocol.

Train one Q_1..Q_K collection per outer fold using only that fold's held-out
automatic responses and its own inner-OOF head. At evaluation, average the
fold Q predictions and use the existing final validation task head. This is a
validation diagnostic, not an official BRiG reproduction or a test result.
"""
import argparse
import math
from pathlib import Path

import torch

from cbmjev.brig import BRiGPolicy, GroupQ, fit_brig
from cbmjev.contracts import DeclaredCost, stable_hash
from cbmjev.crossfit_artifacts import load_crossfit_head
from cbmjev.crossfit_cache import load_crossfit_cache
from cbmjev.crossfit_evaluation import _load_sources, _verify_binding_files_unchanged
from cbmjev.crossfit_merge import load_outer_folds
from cbmjev.crossfit_training import _merge_provenance
from cbmjev.evaluation import summarize_traces
from cbmjev.io import file_hash, fresh_dir, read_json, write_json, write_jsonl
from cbmjev.pipeline import code_fingerprint
from cbmjev.provenance import make_fit_record, validate_target_exclusion
from cbmjev.runtime import ReplayEnvironment, run_episode


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train-fold", "evaluate"):
        p = sub.add_parser(command)
        for key in ("prepared", "planned", "merged", "responder", "cache", "out"):
            p.add_argument("--" + key, required=True)
        p.add_argument("--device", default="cpu")
        p.add_argument("--responder-source-dir")
    train = sub.choices["train-fold"]
    train.add_argument("--outer-fold", type=int, required=True)
    train.add_argument("--max-budget", type=int, default=16)
    train.add_argument("--seed", type=int, default=60)
    train.add_argument("--hidden", type=int, default=128)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=.001)
    train.add_argument("--no-empty-rollout", action="store_true")
    evaluate = sub.choices["evaluate"]
    evaluate.add_argument("--fold-models", nargs="+", required=True)
    evaluate.add_argument("--budget", type=int, default=16)
    return parser.parse_args(argv)


def _source_paths(args):
    return tuple(Path(getattr(args, name)) for name in
                 ("prepared", "planned", "merged", "responder", "cache"))


def _fold_sources(args, fold):
    prepared, planned, merged, _, _ = _source_paths(args)
    receipt = read_json(merged / "receipt.json")
    sources = receipt["sources"]
    schema, data = load_outer_folds(prepared, planned,
        [source["directory"] for source in sources],
        responder_source_dir=args.responder_source_dir, validate_target_records=False)
    if data["sources"] != sources or data["plan_binding"] != receipt["plan_binding"]:
        raise ValueError("merged outer source binding changed")
    selected = [source for source in sources if source["outer_fold"] == fold]
    if len(selected) != 1:
        raise ValueError("outer fold is absent or duplicated")
    outer = Path(selected[0]["directory"])
    head, head_report = load_crossfit_head(outer / "head", schema, device=args.device)
    _, rows, manifest = load_crossfit_cache(outer / "outer/cache", prepared, planned,
        outer / "outer/responder", responder_source_dir=args.responder_source_dir)
    groups = sorted({row["group_id"] for row in rows})
    if (not rows or manifest["response_source"] != "automatic_model"
            or manifest["target_group_ids"] != groups
            or head_report["head_artifact_id"] != read_json(outer / "receipt.json")["head_artifact_id"]):
        raise ValueError("fold responses or task head do not match the outer receipt")
    exclusion = validate_target_exclusion(
        _merge_provenance(head_report["provenance"], manifest["provenance"]),
        artifact_ids=[head_report["head_artifact_id"], manifest["responder_artifact_id"]],
        target_group_ids=groups)
    if set(head_report["fit_group_ids"]) & set(groups):
        raise ValueError("fold task head saw BRiG training groups")
    binding = {"outer_fold": fold, "outer_receipt_sha256": file_hash(outer / "receipt.json"),
               "outer_cache_manifest_sha256": file_hash(outer / "outer/cache/manifest.json"),
               "outer_head_receipt_sha256": file_hash(outer / "head/receipt.json"),
               "outer_head_artifact_id": head_report["head_artifact_id"],
               "outer_responder_artifact_id": manifest["responder_artifact_id"],
               "training_group_ids": groups, "training_sample_count": len(rows),
               "exclusion": exclusion, "merged_receipt_sha256": file_hash(merged / "receipt.json")}
    return schema, rows, head, binding, head_report, manifest


def train_fold(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError("fold output must be new")
    source_before = code_fingerprint()
    schema, rows, head, binding, head_report, manifest = _fold_sources(args, args.outer_fold)
    if not 1 <= args.max_budget <= schema.num_groups:
        raise ValueError("max budget exceeds query groups")
    config = dict(seed=args.seed, device=args.device, hidden=args.hidden,
                  max_budget=args.max_budget, epochs=args.epochs,
                  batch_size=args.batch_size, learning_rate=args.learning_rate,
                  empty_rollout=not args.no_empty_rollout)
    # The core fit accepts only policy_fit. The verified outer rows alone are
    # relabeled here; neither final validation rows nor other folds enter it.
    fit_rows = [{**row, "split": "policy_fit"} for row in rows]
    policy, report = fit_brig(fit_rows, head, schema, config,
                             excluded_head_group_ids=binding["training_group_ids"])
    if report["sample_count"] != len(rows) or report["schema_hash"] != schema.hash:
        raise ValueError("BRiG report differs from verified fold rows")
    _, _, _, refreshed, _, _ = _fold_sources(args, args.outer_fold)
    if refreshed != binding or code_fingerprint() != source_before:
        raise ValueError("training sources changed during fitting")
    out = fresh_dir(out)
    torch.save({str(b): {k: v.detach().cpu() for k, v in model.state_dict().items()}
                for b, model in policy.models.items()}, out / "brig.pt")
    checkpoint = file_hash(out / "brig.pt")
    policy_id = "brig-crossfit:" + checkpoint
    provenance = _merge_provenance(head_report["provenance"], manifest["provenance"], [make_fit_record(
        policy_id, supervised_group_ids=binding["training_group_ids"],
        parent_ids=[binding["outer_head_artifact_id"], binding["outer_responder_artifact_id"]],
        metadata={"role": "outer_fold_brig", "outer_fold": args.outer_fold})])
    write_json(out / "training.json", {"report": report, "binding": binding,
        "source_code_hash": source_before, "schema_hash": schema.hash,
        "adapter_sha256": file_hash(__file__),
        "policy_artifact_id": policy_id, "provenance": provenance,
        "evidence_status": "FOLD_TRAINING_NOT_PAPER_RESULT"})
    receipt = {"format": "cbmjev-crossfit-brig-fold-v1", "status": "COMPLETE",
        "outer_fold": args.outer_fold, "checkpoint_sha256": checkpoint,
        "training_sha256": file_hash(out / "training.json"),
        "adapter_sha256": file_hash(__file__),
        "source_code_hash": source_before, "binding": binding,
        "policy_artifact_id": policy_id, "schema_hash": schema.hash}
    receipt["receipt_hash"] = stable_hash(receipt)
    write_json(out / "receipt.json", receipt)
    return receipt


def _load_fold_policy(directory, args, schema):
    directory = Path(directory)
    receipt = read_json(directory / "receipt.json")
    core = dict(receipt)
    if (core.pop("receipt_hash", None) != stable_hash(core)
            or receipt.get("format") != "cbmjev-crossfit-brig-fold-v1"
            or receipt.get("status") != "COMPLETE"):
        raise ValueError("invalid BRiG fold receipt")
    if (receipt["checkpoint_sha256"] != file_hash(directory / "brig.pt")
            or receipt["training_sha256"] != file_hash(directory / "training.json")
            or receipt["source_code_hash"] != code_fingerprint()
            or receipt.get("adapter_sha256") != file_hash(__file__)
            or receipt["schema_hash"] != schema.hash):
        raise ValueError("BRiG fold model, report, schema or source changed")
    training = read_json(directory / "training.json")
    _, _, _, binding, _, _ = _fold_sources(args, receipt["outer_fold"])
    if (receipt["binding"] != binding or training["binding"] != binding
            or training["source_code_hash"] != receipt["source_code_hash"]
            or training.get("adapter_sha256") != receipt["adapter_sha256"]
            or training["policy_artifact_id"] != receipt["policy_artifact_id"]
            or training["schema_hash"] != schema.hash):
        raise ValueError("BRiG fold provenance differs from verified sources")
    report = training["report"]
    if (report["schema_hash"] != schema.hash
            or report["sample_count"] != binding["training_sample_count"]
            or report["config"]["max_budget"] < args.budget):
        raise ValueError("BRiG fold was not trained for requested budget")
    payload = torch.load(directory / "brig.pt", map_location="cpu", weights_only=True)
    if set(payload) != {str(i) for i in range(1, report["config"]["max_budget"] + 1)}:
        raise ValueError("BRiG fold checkpoint budget keys differ")
    models = {}
    for budget in range(1, report["config"]["max_budget"] + 1):
        model = GroupQ(schema, report["config"]["hidden"]).to(args.device)
        model.load_state_dict(payload[str(budget)], strict=True)
        if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
            raise ValueError("nonfinite BRiG model")
        models[budget] = model.eval().requires_grad_(False)
    return BRiGPolicy(schema, models), receipt, training


class MeanQPolicy:
    """Average fold predictions for the same feasible observed state and budget."""
    def __init__(self, policies):
        if not policies or any(p.schema.hash != policies[0].schema.hash for p in policies):
            raise ValueError("nonempty same-schema BRiG policies required")
        self.policies = tuple(policies)

    def choose(self, observed, remaining_budget):
        if remaining_budget == 0:
            return ()
        predictions = [p.predict_budget(observed, remaining_budget) for p in self.policies]
        candidates = [tuple(g for g, _ in values) for values in predictions]
        if any(values != candidates[0] for values in candidates):
            raise ValueError("fold policies disagree on available actions")
        return (min(candidates[0], key=lambda g: (
            sum(dict(values)[g] for values in predictions) / len(predictions), g)),)


def evaluate(args):
    prepared, planned, merged, responder, cache = _source_paths(args)
    out = Path(args.out)
    if out.exists():
        raise ValueError("evaluation output must be new")
    source_before = code_fingerprint()
    schema, rows, config, head, _, binding = _load_sources(
        prepared, planned, merged, responder, cache, args.device,
        responder_source_dir=args.responder_source_dir)
    if not 1 <= args.budget <= schema.num_groups or config["policy"]["max_cost"] is not None:
        raise ValueError("BRiG requires a feasible fixed group budget without max_cost")
    if len(args.fold_models) != len(binding["outer_sources"]):
        raise ValueError("exactly one trained BRiG model per outer fold is required")
    loaded = [_load_fold_policy(path, args, schema) for path in args.fold_models]
    folds = [receipt["outer_fold"] for _, receipt, _ in loaded]
    if sorted(folds) != list(range(len(folds))):
        raise ValueError("BRiG outer folds are missing or duplicated")
    if len({training["report"]["config"]["max_budget"] for _, _, training in loaded}) != 1:
        raise ValueError("fold BRiG budgets differ")
    policies = [entry[0] for entry in sorted(loaded, key=lambda x: x[1]["outer_fold"])]
    ensemble = MeanQPolicy(policies)
    provenance = _merge_provenance(*(training["provenance"] for _, _, training in loaded))
    exclusion = validate_target_exclusion(provenance,
        artifact_ids=[receipt["policy_artifact_id"] for _, receipt, _ in loaded],
        target_group_ids=sorted({row["group_id"] for row in rows}))
    model_bindings = [{"outer_fold": receipt["outer_fold"],
        "receipt_sha256": file_hash(Path(path) / "receipt.json"),
        "checkpoint_sha256": receipt["checkpoint_sha256"]}
        for path, (_, receipt, _) in zip(args.fold_models, loaded)]
    spec = {"method": "brig_mean_q", "budget": args.budget,
        "split": "validation", "mode": "offline_replay", "source_binding": binding,
        "fold_models": sorted(model_bindings, key=lambda x: x["outer_fold"]),
        "validation_exclusion": exclusion,
        "cost": config["cost"], "budget_type": "fixed_exact_groups"}
    spec["system_hash"] = stable_hash(spec)
    cost = DeclaredCost(**config["cost"])
    traces = []
    for index, row in enumerate(rows, 1):
        trace = run_episode(ReplayEnvironment(row["z"], schema), schema, head,
            method="brig", controller=ensemble, cost=cost, max_groups=args.budget,
            max_cost=None, include_pairs=False, include_all=False,
            seed=config["seed"], device=args.device)
        if len(trace["queried_groups"]) != args.budget:
            raise ValueError("BRiG did not use its exact group budget")
        trace.update(sample_id=row["sample_id"], group_id=row["group_id"], y=row["y"],
                     split="validation", policy_id="brig_mean_q_K" + str(args.budget),
                     system_hash=spec["system_hash"], seed=config["seed"],
                     num_query_groups=schema.num_groups, num_atoms=schema.num_atoms,
                     budget_groups=args.budget)
        traces.append(trace)
        if index % 100 == 0 or index == len(rows):
            print(f"[brig-crossfit] validation {index}/{len(rows)}", flush=True)
    metrics = summarize_traces(traces, num_classes=schema.num_classes)
    metrics["budget_groups"] = args.budget
    _verify_binding_files_unchanged(binding, merged, responder, cache,
                                    planned_dir=Path(args.planned))
    if code_fingerprint() != source_before or any(
        file_hash(Path(path) / "receipt.json") != item["receipt_sha256"] or
        file_hash(Path(path) / "brig.pt") != item["checkpoint_sha256"]
        for path, item in zip(args.fold_models, model_bindings)):
        raise ValueError("BRiG or source changed during evaluation")
    out = fresh_dir(out)
    write_jsonl(out / "traces.jsonl", traces)
    write_json(out / "settings.json", spec)
    write_json(out / "metrics.json", {"split": "validation", "mode": "offline_replay",
        "seed": config["seed"], "budget_groups": args.budget,
        "policies": {"brig_mean_q_K" + str(args.budget): metrics},
        "paper_evidence": False,
        "evidence_status": "OFFLINE_VALIDATION_DEVELOPMENT_NOT_TEST_OR_LATENCY_EVIDENCE",
        "source_binding": binding, "fold_models": model_bindings,
        "source_code_hash": source_before})
    receipt = {"format": "cbmjev-crossfit-brig-validation-v1", "status": "COMPLETE",
        "paper_evidence": False, "samples": len(rows), "budget_groups": args.budget,
        "source_code_hash": source_before, "system_hash": spec["system_hash"],
        "files_sha256": {name: file_hash(out / name) for name in
                         ("traces.jsonl", "settings.json", "metrics.json")}}
    receipt["receipt_hash"] = stable_hash(receipt)
    write_json(out / "receipt.json", receipt)
    return receipt


def main(argv=None):
    args = parse_args(argv)
    result = train_fold(args) if args.command == "train-fold" else evaluate(args)
    print(result, flush=True)


if __name__ == "__main__":
    main()

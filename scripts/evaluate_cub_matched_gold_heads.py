#!/usr/bin/env python3
"""Train matched auto/gold CUB heads on identical complete cases, evaluate val.

This is a complete-case validation diagnostic, not a population oracle bound.
OOF automatic train responses and public gold train annotations share every
training sample. Both heads use identical initialization, masks and optimizer.
"""
import argparse
import json
import math
from pathlib import Path
import random

import torch

from cbmjev.contracts import stable_hash
from cbmjev.crossfit_cache import load_crossfit_cache
from cbmjev.io import file_hash, fresh_dir, read_json, write_json
from cbmjev.learning import (MaskedHead, encode_states, mask_answers,
                             normalize_config, sample_group_masks, schema_signature)
from scripts.evaluate_cub_paired_gold_intervention import (
    complete_gold_validation, summarize_paired)


def complete_train_pairs(prepared, schema, automatic):
    by_id = {row["sample_id"]: row for row in automatic}
    if len(by_id) != len(automatic):
        raise ValueError("duplicate OOF automatic sample IDs")
    seen, pairs = set(), []
    with (Path(prepared) / "samples.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            source = json.loads(line)
            if source["split"] != "train":
                continue
            sid = source["sample_id"]
            if sid in seen:
                raise ValueError("duplicate prepared train sample ID")
            seen.add(sid)
            row = by_id.get(sid)
            if row is None or row["group_id"] != source["group_id"]:
                raise ValueError("OOF response does not match train membership")
            target = source["target"]
            if target["status"] != "OBSERVED" or target["value"] != row["y"]:
                raise ValueError("OOF/train target mismatch")
            labels = source["concepts"]
            if len(labels) != schema.num_atoms:
                raise ValueError("train gold width mismatch")
            gold = []
            complete = True
            for label, concept in zip(labels, schema.concepts):
                if label["concept_id"] != concept.id:
                    raise ValueError("train gold concept order mismatch")
                if label["annotation_status"] == "OBSERVED":
                    value = label["value"]
                    if type(value) is not int or not 0 <= value < len(concept.values):
                        raise ValueError("invalid observed train gold value")
                    gold.append(value)
                else:
                    if label["value"] is not None:
                        raise ValueError("missing train gold status has value")
                    complete = False
            if complete:
                schema.validate_state(tuple(gold), complete=True)
                schema.validate_state(tuple(row["z"]), complete=True)
                pairs.append({"sample_id": sid, "group_id": row["group_id"],
                              "y": row["y"], "automatic": tuple(row["z"]),
                              "gold": tuple(gold)})
    if seen != set(by_id) or not pairs:
        raise ValueError("OOF train coverage mismatch or empty complete subset")
    return sorted(pairs, key=lambda row: row["sample_id"])


def fit_pair(pairs, schema, base_config, *, seed, epochs, cpu_threads):
    config = normalize_config({**base_config, "seed": seed, "device": "cpu",
                               "head_epochs": epochs, "cpu_threads": cpu_threads,
                               "class_weighting": "none"}, schema)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(cpu_threads)
    heads = {}
    optimizers = {}
    for source in ("automatic", "gold"):
        torch.manual_seed(seed)
        heads[source] = MaskedHead(schema, config)
        optimizers[source] = torch.optim.AdamW(
            heads[source].network.parameters(), lr=config["learning_rate"],
            weight_decay=config["weight_decay"])
    first, second = (heads[source].network.state_dict() for source in
                     ("automatic", "gold"))
    if any(not torch.equal(first[k], second[k]) for k in first):
        raise ValueError("matched heads did not initialize identically")
    rng = random.Random(seed + 31)
    losses = {source: [] for source in heads}
    for _ in range(epochs):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        totals = {source: 0.0 for source in heads}
        for offset in range(0, len(order), config["batch_size"]):
            indices = order[offset:offset + config["batch_size"]]
            masks = sample_group_masks(len(indices), schema.num_groups, rng)
            targets = torch.tensor([pairs[i]["y"] for i in indices], dtype=torch.long)
            for source in ("automatic", "gold"):
                head = heads[source]
                head.network.train()
                states = [mask_answers(pairs[i][source], mask, schema)
                          for i, mask in zip(indices, masks)]
                logits = head.network(encode_states(states, schema, head.device))
                loss = head.cross_entropy(logits, targets)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite matched-head loss")
                optimizers[source].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[source].step()
                totals[source] += float(loss.detach()) * len(indices)
        for source, head in heads.items():
            losses[source].append(totals[source] / len(pairs))
            head.network.eval()
    return heads, config, losses


def evaluate_pair(rows, val_gold, schema, heads, order, budgets, batch_size, seed):
    selected = [row for row in rows if row["sample_id"] in val_gold]
    if len(selected) != len(val_gold) or sorted(order) != list(range(schema.num_groups)):
        raise ValueError("validation pairing/order mismatch")
    output = {}
    for budget in budgets:
        if type(budget) is not int or not 0 <= budget <= schema.num_groups:
            raise ValueError("invalid budget")
        visible = set(order[:budget])
        mask = tuple(g in visible for g in range(schema.num_groups))
        scores = {}
        for source in ("automatic", "gold"):
            head = heads[source]
            for start in range(0, len(selected), batch_size):
                batch = selected[start:start + batch_size]
                states = [mask_answers(
                    row["z"] if source == "automatic" else val_gold[row["sample_id"]],
                    mask, schema) for row in batch]
                probs = head.probabilities_many(states)
                for row, p in zip(batch, probs):
                    y = row["y"]
                    scores[(row["sample_id"], source)] = {
                        "ce": -math.log(max(p[y], 1e-12)),
                        "correct": int(max(range(len(p)), key=p.__getitem__) == y)}
        entries = [{"automatic": scores[(row["sample_id"], "automatic")],
                    "gold": scores[(row["sample_id"], "gold")]} for row in selected]
        output[str(budget)] = summarize_paired(entries, seed=seed + 1009 * budget)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "planned", "merged", "responder", "cache",
                 "static-order", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--budgets", type=int, nargs="+", default=(2, 4, 8, 16, 28))
    args = parser.parse_args()
    out = Path(args.out)
    if (out.exists() or args.epochs < 1 or args.cpu_threads < 1 or
            args.batch_size < 1 or len(set(args.budgets)) != len(args.budgets)):
        raise ValueError("output exists or invalid training/evaluation settings")
    prepared, planned, merged, responder, cache = (
        Path(getattr(args, name)) for name in
        ("prepared", "planned", "merged", "responder", "cache"))
    receipt = read_json(merged / "receipt.json")
    sources = sorted(receipt["sources"], key=lambda source: source["outer_fold"])
    if [source["outer_fold"] for source in sources] != list(range(len(sources))):
        raise ValueError("incomplete ordered OOF folds")
    training, fold_bindings, schema = [], [], None
    for source in sources:
        outer = Path(source["directory"])
        fold_schema, rows, manifest = load_crossfit_cache(
            outer / "outer/cache", prepared, planned, outer / "outer/responder")
        if (schema is not None and fold_schema != schema) or manifest["response_source"] != "automatic_model":
            raise ValueError("OOF schema/source mismatch")
        schema = fold_schema
        training.extend(rows)
        fold_bindings.append({"fold": source["outer_fold"], "rows": len(rows),
            "cache_manifest_sha256": file_hash(outer / "outer/cache/manifest.json")})
    train_pairs = complete_train_pairs(prepared, schema, training)
    val_schema, validation, val_manifest = load_crossfit_cache(
        cache, prepared, planned, responder)
    if (val_schema != schema or val_manifest["split"] != "validation" or
            val_manifest["response_source"] != "automatic_model"):
        raise ValueError("validation cache mismatch")
    val_gold = complete_gold_validation(prepared, schema, validation)
    static = read_json(args.static_order)
    if (static.get("schema_signature") != schema_signature(schema) or
            static.get("evaluation_holdout_labels_used") is not False or
            static.get("fit_scope") != "ALL_OBSERVED_OUTER_OOF_TRAIN_TARGETS"):
        raise ValueError("static order is not training-only and schema-bound")
    base_config = read_json(merged / "head/head_report.json")["config"]
    if base_config["class_weighting"] != "none":
        raise ValueError("matched complete-case diagnostic requires unweighted head")
    heads, config, losses = fit_pair(train_pairs, schema, base_config,
                                    seed=args.seed, epochs=args.epochs,
                                    cpu_threads=args.cpu_threads)
    metrics = evaluate_pair(validation, val_gold, schema, heads, static["order"],
                            args.budgets, args.batch_size, args.seed)
    out = fresh_dir(out)
    checkpoints = {}
    for source, head in heads.items():
        path = out / (source + "_head.pt")
        with path.open("xb") as stream:
            torch.save({"format": "cbmjev-cub-matched-complete-head-v1",
                        "schema_signature": schema_signature(schema),
                        "config": config, "source": source,
                        "state_dict": {key: value.detach().cpu()
                                       for key, value in head.network.state_dict().items()}}, stream)
        checkpoints[source] = file_hash(path)
    report = {"format": "cbmjev-cub-matched-gold-heads-validation-v1",
        "evidence_status": "EXPLORATORY_COMPLETE_CASE_VALIDATION_NOT_POPULATION_ORACLE",
        "seed": args.seed, "split": "validation", "test_evaluated": False,
        "train_complete_cases": len(train_pairs),
        "train_target_classes": len({row["y"] for row in train_pairs}),
        "validation_complete_cases": len(val_gold),
        "validation_target_classes": len({row["y"] for row in validation
                                          if row["sample_id"] in val_gold}),
        "training_sample_ids_hash": stable_hash([row["sample_id"] for row in train_pairs]),
        "validation_sample_ids_hash": stable_hash(sorted(val_gold)),
        "training_protocol": "same cases, labels, random initialization, sampled masks, epochs and optimizer",
        "selection_rule": "all public concept atoms OBSERVED; no target/prediction-based selection",
        "config": config, "losses": losses, "metrics": metrics,
        "fold_bindings": fold_bindings, "checkpoint_sha256": checkpoints,
        "bindings": {"prepared_samples_sha256": file_hash(prepared / "samples.jsonl"),
                     "merged_receipt_sha256": file_hash(merged / "receipt.json"),
                     "validation_cache_manifest_sha256": file_hash(cache / "manifest.json"),
                     "static_order_sha256": file_hash(args.static_order),
                     "script_sha256": file_hash(__file__)},
        "caveats": ["Complete-case selection shifts CUB class mix substantially.",
                    "Each source-specific head sees only complete-case train samples, not all 4807 train cases.",
                    "This compares response regimes, not JEV/controller quality or an all-validation oracle bound.",
                    "Development validation has informed earlier design; intervals are descriptive."]}
    report["report_hash"] = stable_hash(report)
    write_json(out / "report.json", report)
    print(json.dumps({"out": str(out), "train_complete": len(train_pairs),
                      "validation_complete": len(val_gold),
                      "K16": metrics.get("16")}, sort_keys=True))


if __name__ == "__main__":
    main()

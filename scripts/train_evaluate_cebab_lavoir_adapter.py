#!/usr/bin/env python3
"""CEBaB concept-only LAVOIR-style Eq. (3)/(4) adaptation, CPU replay.

Two equal-capacity MLPs see the same partial concept state and singleton action,
trained on the same realized frozen-head gold-probability gain. The uncapped
output is h; the capped output is G(p_before)*sigmoid(h). This is not the
original text/slot encoder, answer simulator, or exact-posterior LAVOIR.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn

from cbmjev.contracts import DeclaredCost, stable_hash
from cbmjev.evaluation import summarize_traces
from cbmjev.io import file_hash, fresh_dir, read_jsonl, write_json, write_jsonl
from cbmjev.learning import (ActionController, encode_actions, encode_states,
                             normalize_config, policy_examples, training_rows)
from cbmjev.pipeline import load_model_bundle
from cbmjev.runtime import ReplayEnvironment


LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.40)
THRESHOLDS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
VARIANTS = ("uncapped", "gini_capped")


def gini(probabilities):
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("invalid probability")
    if abs(sum(probabilities) - 1) > 1e-5:
        raise ValueError("probabilities must sum to one")
    return max(0.0, 1.0 - sum(p * p for p in probabilities))


def targets(head, examples, schema):
    """Gold y and future answers are offline supervision only, never policy input."""
    before = [e[0] for e in examples]
    after = [e[2] for e in examples]
    p0 = head.probabilities_many(before)
    p1 = head.probabilities_many(after)
    values, caps = [], []
    for example, b, a in zip(examples, p0, p1):
        y = example[3]
        if type(y) is not int or not 0 <= y < schema.num_classes:
            raise ValueError("invalid target label")
        value = float(a[y] - b[y])
        if not -1 - 1e-6 <= value <= 1 + 1e-6:
            raise ValueError("probability-gain target out of [-1,1]")
        values.append(value)
        caps.append(gini(b))
    return values, caps


def fit_pair(rows, schema, head, learning, seed, *, epochs=None, limit=None):
    cfg = normalize_config({**learning, "seed": seed, "device": "cpu",
        "objective": "value", "include_pairs": False, "include_all": False}, schema)
    if epochs is not None:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        cfg["policy_epochs"] = epochs
    selected = [r for r in rows if r["split"] == "policy_fit"]
    if not selected:
        raise ValueError("policy_fit required")
    if limit is not None:
        selected = selected[:limit]
    training = training_rows(selected, "policy_fit", schema)
    torch.manual_seed(seed + 1009)
    uncapped = ActionController(schema, cfg)
    capped = ActionController(schema, cfg)
    capped.network.load_state_dict(copy.deepcopy(uncapped.network.state_dict()))
    optimizers = {"uncapped": torch.optim.AdamW(uncapped.network.parameters(),
                      lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]),
                  "gini_capped": torch.optim.AdamW(capped.network.parameters(),
                      lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])}
    models = {"uncapped": uncapped, "gini_capped": capped}
    losses, target_counts = {v: [] for v in VARIANTS}, {"negative": 0, "zero": 0,
        "positive": 0, "above_gini": 0, "examples_first_epoch": 0}
    for epoch in range(cfg["policy_epochs"]):
        for model in models.values():
            model.network.train()
        examples = (e for e in policy_examples(training, schema, cfg, epoch)
                    if len(e[1]) == 1)
        batch, sums, count = [], {v: 0.0 for v in VARIANTS}, 0
        def step(items):
            nonlocal count
            target, cap = targets(head, items, schema)
            if epoch == 0:
                target_counts["negative"] += sum(x < -1e-8 for x in target)
                target_counts["zero"] += sum(abs(x) <= 1e-8 for x in target)
                target_counts["positive"] += sum(x > 1e-8 for x in target)
                target_counts["above_gini"] += sum(x > c + 1e-8 for x, c in zip(target, cap))
                target_counts["examples_first_epoch"] += len(items)
            x = torch.cat((encode_states([e[0] for e in items], schema, "cpu"),
                           encode_actions([e[1] for e in items], schema, "cpu")), dim=-1)
            y = torch.tensor(target, dtype=torch.float32)
            g = torch.tensor(cap, dtype=torch.float32)
            for name, model in models.items():
                raw = model.network(x).flatten()
                output = raw if name == "uncapped" else g * torch.sigmoid(raw)
                loss = nn.functional.mse_loss(output, y)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite LAVOIR-adapter loss")
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[name].step()
                sums[name] += float(loss.detach()) * len(items)
            count += len(items)
        for example in examples:
            batch.append(example)
            if len(batch) == cfg["batch_size"]:
                step(batch)
                batch = []
        if batch:
            step(batch)
        if not count:
            raise ValueError("no singleton acquisition targets")
        for name in VARIANTS:
            losses[name].append(sums[name] / count)
    for model in models.values():
        model.network.eval().requires_grad_(False)
    return models, {"training_role": "policy_fit", "policy_fit_rows": len(training),
        "epochs": cfg["policy_epochs"], "examples_per_epoch": count,
        "same_initialization_and_batches": True, "configuration": cfg,
        "target_signs_first_epoch": target_counts, "training_mse": losses,
        "threshold_selection_overlap_caveat": "policy_fit is also used for threshold selection; exploratory development estimate only"}


class AdapterScores:
    def __init__(self, schema, head, models):
        self.schema, self.head, self.models = schema, head, models
        self.probabilities_cache, self.values_cache = {}, {}

    def probabilities(self, state):
        key = tuple(state)
        if key not in self.probabilities_cache:
            self.probabilities_cache[key] = tuple(float(p) for p in self.head.probabilities(key))
        return self.probabilities_cache[key]

    def values(self, state, actions, variant):
        key = (tuple(state), tuple(actions), variant)
        if key not in self.values_cache:
            raw = self.models[variant].predict(state, actions)
            self.values_cache[key] = (tuple(raw) if variant == "uncapped" else
                tuple((gini(self.probabilities(state)) *
                       torch.sigmoid(torch.tensor(raw))).tolist()))
        return self.values_cache[key]


def choose(state, schema, scores, cost, variant, threshold):
    if variant not in VARIANTS or threshold < 0:
        raise ValueError("unknown variant or negative threshold")
    remaining = tuple(g for g, used in enumerate(schema.group_mask(state)) if not used)
    if not remaining:
        return (), {}
    actions = tuple((g,) for g in remaining)
    values = scores.values(state, actions, variant)
    utilities = tuple(v - threshold * cost(state, action)
                      for v, action in zip(values, actions))
    best = min(range(len(actions)), key=lambda i: (-utilities[i], actions[i]))
    return (actions[best] if utilities[best] > 0 else ()), {
        str(g): value for g, value in zip(remaining, values)}


def replay(row, schema, scores, cost, variant, threshold):
    env = ReplayEnvironment(row["z"], schema)
    declared, steps = 0.0, []
    for _ in range(schema.num_groups + 1):
        state = env.state()
        action, detail = choose(state, schema, scores, cost, variant, threshold)
        step = {"before": list(state), "action": list(action), "scores": detail,
                "decision": "ACQUIRE" if action else "STOP",
                "declared_cost": cost(state, action),
                "legal_candidate_count": 1 + schema.num_groups - sum(schema.group_mask(state))}
        if not action:
            steps.append(step)
            break
        values = env.query(action)
        declared += step["declared_cost"]
        step.update(atom_ids=list(schema.expand(action)), values=list(values))
        steps.append(step)
    else:
        raise RuntimeError("adapter episode failed to stop")
    probs = scores.probabilities(env.state())
    prediction = max(range(schema.num_classes), key=lambda i: probs[i])
    return {"sample_id": row["sample_id"], "group_id": row["group_id"],
        "split": row["split"], "y": row["y"], "method": variant,
        "prediction": prediction, "probabilities": list(probs),
        "queried_groups": [g for g, used in enumerate(schema.group_mask(env.state())) if used],
        "queried_atoms": [a for a, v in enumerate(env.state()) if v != -1],
        "calls": env.calls, "declared_cost": declared, "cost_units": cost.units,
        "mode": "offline_replay", "steps": steps, "final_state": list(env.state())}


def external_j(report, lam, groups):
    return report["error"] + lam * report["mean_queried_groups"] / groups


def select(fit_reports, variant, lam, groups):
    candidates = [(t, fit_reports[(variant, t)]) for t in THRESHOLDS]
    return min(candidates, key=lambda item: (external_j(item[1], lam, groups),
        item[1]["mean_queried_groups"], item[0]))[0]


def paired(a, b, lam, groups):
    first, second = {r["sample_id"]: r for r in a}, {r["sample_id"]: r for r in b}
    if not first or first.keys() != second.keys():
        raise ValueError("paired methods require identical samples")
    output = {"num_samples": len(first), "path_different": 0,
        "query_set_different": 0, "prediction_different": 0,
        "a_wrong_b_correct": 0, "a_correct_b_wrong": 0, "mean_b_minus_a_j": 0.0}
    for key in first:
        x, y = first[key], second[key]
        if (x["group_id"], x["y"], x["split"]) != (y["group_id"], y["y"], y["split"]):
            raise ValueError("paired metadata mismatch")
        path = lambda row: tuple(tuple(s["action"]) for s in row["steps"] if s["action"])
        output["path_different"] += path(x) != path(y)
        output["query_set_different"] += x["queried_groups"] != y["queried_groups"]
        output["prediction_different"] += x["prediction"] != y["prediction"]
        output["a_wrong_b_correct"] += x["prediction"] != x["y"] and y["prediction"] == y["y"]
        output["a_correct_b_wrong"] += x["prediction"] == x["y"] and y["prediction"] != y["y"]
        output["mean_b_minus_a_j"] += int(y["prediction"] != y["y"]) - int(x["prediction"] != x["y"]) + lam * (
            len(y["queried_groups"]) - len(x["queried_groups"])) / groups
    output["mean_b_minus_a_j"] /= len(first)
    return output


def run(models, cache, out, *, limit=None, epochs=None, cpu_threads=2,
        signed_traces=None):
    if cpu_threads < 1 or limit is not None and limit < 1:
        raise ValueError("cpu_threads and limit must be positive")
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("output must be new or empty")
    torch.set_num_threads(cpu_threads)
    schema, rows, manifest, config, receipt, head, _, _ = load_model_bundle(
        models, cache, device="cpu")
    if schema.dataset != "cebab" or schema.num_groups != 4:
        raise ValueError("adapter requires four-group CEBaB")
    if config["policy"]["max_cost"] is not None:
        raise ValueError("hard max_cost not implemented for adapter")
    fit = [r for r in rows if r["split"] == "policy_fit"]
    val = [r for r in rows if r["split"] == "validation"]
    if not fit or not val:
        raise ValueError("policy_fit and validation required")
    original = {"policy_fit": len(fit), "validation": len(val)}
    if limit is not None:
        fit, val = fit[:limit], val[:limit]
    pair, training_report = fit_pair(fit, schema, head, config["learning"],
        config["seed"], epochs=epochs, limit=None)
    scores = AdapterScores(schema, head, pair)
    cost = DeclaredCost(**config["cost"])
    fit_reports = {}
    for variant in VARIANTS:
        for threshold in THRESHOLDS:
            traces = [replay(r, schema, scores, cost, variant, threshold) for r in fit]
            fit_reports[(variant, threshold)] = summarize_traces(traces, num_classes=schema.num_classes)
    selected, val_traces, comparisons = {}, [], {}
    for lam in LAMBDAS:
        selected[str(lam)] = {}
        selected_traces = {}
        for variant in VARIANTS:
            threshold = select(fit_reports, variant, lam, schema.num_groups)
            traces = [replay(r, schema, scores, cost, variant, threshold) for r in val]
            summary = summarize_traces(traces, num_classes=schema.num_classes)
            selected[str(lam)][variant] = {"threshold": threshold,
                "policy_fit_j": external_j(fit_reports[(variant, threshold)], lam, schema.num_groups),
                "validation_j": external_j(summary, lam, schema.num_groups),
                "validation": summary}
            selected_traces[variant] = traces
            val_traces.extend({**trace, "external_lambda": lam, "selected_threshold": threshold}
                              for trace in traces)
        comparisons[str(lam)] = paired(selected_traces["uncapped"],
                                      selected_traces["gini_capped"], lam, schema.num_groups)
    signed_comparisons = None
    if signed_traces is not None:
        if limit is not None:
            raise ValueError("signed full-validation traces cannot be paired to a pilot")
        reference = read_jsonl(signed_traces)
        signed_comparisons = {}
        for lam in LAMBDAS:
            selected_signed = [r for r in reference if r.get("external_lambda") == lam
                               and r.get("method") == "signed_value"]
            if len(selected_signed) != len(val) or any(r.get("split") != "validation"
                                                       for r in selected_signed):
                raise ValueError("signed reference is not the same complete validation role")
            signed_comparisons[str(lam)] = {variant: paired(selected_signed,
                [r for r in val_traces if r["external_lambda"] == lam and
                 r["method"] == variant], lam, schema.num_groups) for variant in VARIANTS}
    out = fresh_dir(out)
    weights = out / "adapter_weights.pt"
    torch.save({variant: model.network.state_dict() for variant, model in pair.items()}, weights)
    report = {"format": "cbmjev-cebab-lavoir-style-adapter-v1",
        "evidence_status": "PILOT_SUBSAMPLE_NOT_CLAIM_EVIDENCE" if limit is not None or epochs is not None else
                           "EXPLORATORY_VALIDATION_NOT_TEST",
        "test_evaluated": False, "seed": config["seed"],
        "population": {"policy_fit_samples": len(fit), "validation_samples": len(val),
            "original_role_counts": original, "limit": limit},
        "training": training_report,
        "protocol": {"target": "frozen_head_p_after_gold_minus_p_before_gold",
            "uncapped": "raw scalar h", "gini_capped": "G(p_before)*sigmoid(h)",
            "gini": "1-sum_c p_before(c)^2", "same_target_capacity_optimizer_batches": True,
            "cap_status": "heuristic for imperfect frozen head; original bound needs calibrated true posterior",
            "realized_target_note": "Gini bounds conditional expected VOI under a calibrated posterior, not each signed realized target; negative/above-cap examples are not counterexamples",
            "adaptation_scope": "concept-only frozen head; not original LAVOIR encoder or simulator",
            "external_j": "error_rate + lambda*mean_queried_groups/4",
            "lambdas": LAMBDAS, "thresholds": THRESHOLDS,
            "selection_role": "policy_fit (in-sample to training, exploratory)",
            "evaluation_role": "validation", "cost": cost.to_dict()},
        "policy_fit_grid": [{"variant": v, "threshold": t, "report": r}
                            for (v, t), r in fit_reports.items()],
        "selected_validation": selected, "paired_capped_vs_uncapped": comparisons,
        "paired_signed_vs_adapter": signed_comparisons,
        "bindings": {"models_receipt_sha256": file_hash(Path(models) / "receipt.json"),
            "model_weights_sha256": receipt["models_sha256"],
            "cache_manifest_sha256": file_hash(Path(cache) / "manifest.json"),
            "cache_responses_sha256": manifest["responses_sha256"],
            "adapter_weights_sha256": file_hash(weights), "script_sha256": file_hash(__file__),
            "signed_validation_traces_sha256": file_hash(signed_traces) if signed_traces else None}}
    report["report_hash"] = stable_hash(report)
    write_jsonl(out / "validation_traces.jsonl", val_traces)
    write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("models", "cache", "out"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--limit", type=int, help="pilot only: first N rows per role")
    parser.add_argument("--epochs", type=int, help="pilot only: shortened training")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--signed-traces", help="optional same-seed frozen signed-value validation traces")
    args = parser.parse_args()
    report = run(args.models, args.cache, args.out, limit=args.limit,
                 epochs=args.epochs, cpu_threads=args.cpu_threads,
                 signed_traces=args.signed_traces)
    print(json.dumps({"out": args.out, "seed": report["seed"],
        "policy_fit": report["population"]["policy_fit_samples"],
        "validation": report["population"]["validation_samples"]}))


if __name__ == "__main__":
    main()

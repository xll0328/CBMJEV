"""Held-out hard-answer fusion diagnostic for two CEBaB typed sources.

Fits channel models and a common-capacity task lookup only on head_fit,
selects finite smoothing grids on policy_fit, and evaluates on validation.
Validation is repeatedly inspected development data, never a blind test.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

ASPECTS = ("food", "noise", "ambiance", "service")
CHANNEL_GRID = (1.0, 5.0, 20.0)
TASK_GRID = (1.0, 10.0, 50.0)
METHODS = ("a", "b", "reliability", "independent", "joint")


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = {r["sample_id"] for r in rows}
    if len(ids) != len(rows):
        raise ValueError("duplicate sample_id")
    roles = defaultdict(set)
    for r in rows:
        if r["split"] not in ("head_fit", "policy_fit", "validation"):
            raise ValueError("test or unknown role forbidden")
        if len(r["gold"]) != 4 or len(r["source_a"]) != 4 or len(r["source_b"]) != 4:
            raise ValueError("expected four aspects")
        if any(v not in (0, 1, 2) for v in r["source_a"] + r["source_b"]):
            raise ValueError("source emitted invalid hard category")
        if any(v not in (None, 0, 1, 2) for v in r["gold"]):
            raise ValueError("invalid gold category")
        if r["y"] not in range(5):
            raise ValueError("invalid native five-class task label")
        roles[r["split"]].add(r["group_id"])
    for x in roles:
        for y in roles:
            if x != y and roles[x] & roles[y]:
                raise ValueError("family crosses roles")
    if set(roles) != {"head_fit", "policy_fit", "validation"}:
        raise ValueError("all three independent roles required")
    return rows


def fit_channel(rows, alpha):
    result = []
    for j in range(4):
        prior = [0] * 3
        ca = [[0] * 3 for _ in range(3)]
        cb = [[0] * 3 for _ in range(3)]
        cab = [[0] * 9 for _ in range(3)]
        for row in rows:
            gold = row["gold"][j]
            if gold is None:
                continue
            a, b = row["source_a"][j], row["source_b"][j]
            prior[gold] += 1
            ca[gold][a] += 1
            cb[gold][b] += 1
            cab[gold][3*a+b] += 1
        if sum(prior) == 0:
            raise ValueError("no labeled concept for aspect")
        reliability_a = sum(prior[c] and ca[c][c] for c in range(3))
        reliability_b = sum(prior[c] and cb[c][c] for c in range(3))
        result.append({"prior": prior, "a": ca, "b": cb, "joint": cab,
                       "preferred": "a" if reliability_a >= reliability_b else "b",
                       "labeled": sum(prior)})
    return result


def posterior(row, j, channel, method, alpha):
    record = channel[j]
    a, b = row["source_a"][j], row["source_b"][j]
    prior = record["prior"]
    n = sum(prior)
    if method == "reliability":
        method = record["preferred"]
    logs = []
    for c in range(3):
        count = prior[c]
        value = math.log((count+1)/(n+3))
        if method in ("a", "independent"):
            value += math.log((record["a"][c][a] + alpha/3)/(count + alpha))
        if method in ("b", "independent"):
            value += math.log((record["b"][c][b] + alpha/3)/(count + alpha))
        if method == "joint":
            value += math.log((record["joint"][c][3*a+b] + alpha/9)/(count + alpha))
        logs.append(value)
    peak = max(logs)
    weights = [math.exp(x-peak) for x in logs]
    return [x/sum(weights) for x in weights]


def concept_scores(rows, channel, method, alpha):
    nll = brier = correct = n = 0.0
    for row in rows:
        for j, gold in enumerate(row["gold"]):
            if gold is None:
                continue
            probabilities = posterior(row, j, channel, method, alpha)
            nll -= math.log(max(probabilities[gold], 1e-300))
            brier += sum((probabilities[k] - (k == gold))**2 for k in range(3))
            correct += max(range(3), key=probabilities.__getitem__) == gold
            n += 1
    return {"labeled": int(n), "nll": nll/n, "brier": brier/n, "accuracy": correct/n}


def hard_answers(row, channel, method, alpha):
    if method == "a":
        return tuple(row["source_a"])
    if method == "b":
        return tuple(row["source_b"])
    if method == "reliability":
        return tuple(row["source_a"][j] if channel[j]["preferred"] == "a"
                     else row["source_b"][j] for j in range(4))
    return tuple(max(range(3), key=posterior(row, j, channel, method, alpha).__getitem__)
                 for j in range(4))


def _fold(group_id):
    return int.from_bytes(hashlib.sha256(("second-source-oof-v1|" + group_id).encode()).digest()[:4], "big") % 5


def train_task(rows, features, alpha):
    counts = defaultdict(lambda: [0]*5)
    global_counts = [0]*5
    for row, state in zip(rows, features):
        counts[state][row["y"]] += 1
        global_counts[row["y"]] += 1
    n = sum(global_counts)
    prior = [(v+1)/(n+5) for v in global_counts]
    return {"counts": counts, "prior": prior, "alpha": alpha}


def task_probabilities(model, state):
    counts = model["counts"].get(state, [0]*5)
    denominator = sum(counts) + model["alpha"]
    return [(counts[k] + model["alpha"]*model["prior"][k])/denominator for k in range(5)]


def task_scores(rows, channel, method, channel_alpha, task_model, *, include_details=False):
    nll = error = 0.0
    details = []
    for row in rows:
        state = hard_answers(row, channel, method, channel_alpha)
        probabilities = task_probabilities(task_model, state)
        prediction = max(range(5), key=probabilities.__getitem__)
        wrong = int(prediction != row["y"])
        error += wrong
        nll -= math.log(max(probabilities[row["y"]], 1e-300))
        if include_details:
            details.append({"sample_id": row["sample_id"], "group_id": row["group_id"],
                            "error": wrong, "nll": -math.log(max(probabilities[row["y"]], 1e-300))})
    result = {"samples": len(rows), "families": len({r["group_id"] for r in rows}),
              "error": error/len(rows), "nll": nll/len(rows)}
    return (result, details) if include_details else result


def bootstrap_delta(baseline, candidate, *, iterations=2000, seed=20260929):
    b = {r["sample_id"]: r for r in baseline}
    c = {r["sample_id"]: r for r in candidate}
    if b.keys() != c.keys():
        raise ValueError("unpaired bootstrap")
    groups = defaultdict(list)
    for sid in b:
        if b[sid]["group_id"] != c[sid]["group_id"]:
            raise ValueError("family mismatch")
        groups[b[sid]["group_id"]].append(c[sid]["error"]-b[sid]["error"])
    values = list(groups.values())
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        pick = [values[rng.randrange(len(values))] for _ in values]
        draws.append(sum(map(sum, pick))/sum(map(len, pick)))
    draws.sort()
    return {"candidate_minus_a_error": sum(sum(v) for v in values)/sum(len(v) for v in values),
            "family_bootstrap_95_percentile": [draws[int(.025*iterations)], draws[int(.975*iterations)]]}


def confusion_by_role(roles):
    tables = {}
    for role, rows in roles.items():
        aspects = []
        for j in range(4):
            # Axis order is gold semantic class, first hard response, second hard response.
            table = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(3)]
            for row in rows:
                gold = row["gold"][j]
                if gold is not None:
                    table[gold][row["source_a"][j]][row["source_b"][j]] += 1
            aspects.append(table)
        tables[role] = dict(zip(ASPECTS, aspects))
    return tables


def run(rows):
    roles = {role: [r for r in rows if r["split"] == role]
             for role in ("head_fit", "policy_fit", "validation")}
    fit, tune, dev = roles["head_fit"], roles["policy_fit"], roles["validation"]
    channels = {alpha: fit_channel(fit, alpha) for alpha in CHANNEL_GRID}
    selected = {}
    validation_details = {}
    for method in METHODS:
        # The concept channel and the task lookup each get a finite, declared
        # smoothing grid. Both are selected on policy_fit, never validation.
        channel_alpha = min(CHANNEL_GRID, key=lambda alpha:
            (concept_scores(tune, channels[alpha], method, alpha)["nll"], alpha))
        channel = channels[channel_alpha]
        fold_channels = {f: fit_channel([r for r in fit if _fold(r["group_id"]) != f], channel_alpha)
                         for f in range(5)}
        oof_features = [hard_answers(r, fold_channels[_fold(r["group_id"])], method, channel_alpha)
                        for r in fit]
        task_models = {alpha: train_task(fit, oof_features, alpha) for alpha in TASK_GRID}
        task_alpha = min(TASK_GRID, key=lambda alpha:
            (task_scores(tune, channel, method, channel_alpha, task_models[alpha])["nll"], alpha))
        val_task, details = task_scores(dev, channel, method, channel_alpha,
                                        task_models[task_alpha], include_details=True)
        validation_details[method] = details
        selected[method] = {"channel_alpha": channel_alpha, "task_alpha": task_alpha,
            "preferred_source_per_aspect": [record["preferred"] for record in channel],
            "policy_fit_concept": concept_scores(tune, channel, method, channel_alpha),
            "validation_concept": concept_scores(dev, channel, method, channel_alpha),
            "policy_fit_task": task_scores(tune, channel, method, channel_alpha, task_models[task_alpha]),
            "validation_task": val_task}
    for method in METHODS:
        if method != "a":
            selected[method]["paired_task_vs_a"] = bootstrap_delta(
                validation_details["a"], validation_details[method])
    return {"format": "cbmjev-cebab-second-source-hard-fusion-v1",
        "evidence_status": "EXPLORATORY_VALIDATION_NOT_TEST", "test_evaluated": False,
        "roles": {role: {"samples": len(records), "families": len({r["group_id"] for r in records})}
                  for role, records in roles.items()},
        "joint_confusion": {"axes": ["gold", "source_a", "source_b"],
                            "categories": ["Negative", "Positive", "unknown"],
                            "by_role_aspect": confusion_by_role(roles)},
        "protocol": {"channel_grid": CHANNEL_GRID, "task_grid": TASK_GRID,
           "channel_supervision": "observed gold concept labels on head_fit only",
           "channel_tune": "mean concept NLL on policy_fit",
           "task_fit": "five-fold family OOF fused hard features on head_fit",
           "task_tune": "task NLL on policy_fit",
           "task_head": "81-state categorical lookup, smoothed toward head_fit class prior; same capacity each arm",
           "evaluation": "validation already inspected historically, development evidence only",
           "interface": "hard answers only; no Qwen logit or unqueried answer in fusion",
           "family_bootstrap": "2000 paired family resamples; conditional on fitted models and this dev set"},
        "results": selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise ValueError("output report already exists")
    report = run(read_rows(args.responses))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "roles": report["roles"],
                      "results": {m: r["validation_task"] for m, r in report["results"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

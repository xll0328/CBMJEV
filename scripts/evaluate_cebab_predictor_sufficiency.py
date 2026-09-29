#!/usr/bin/env python3
"""CEBaB same-information predictor sufficiency, development roles only.

Four family-held-out configurations are selected inside head_fit. The MLP has
the legacy architecture and the original cardinality-uniform mask objective;
its only new training information is exhaustive coverage of all 16 masks.
The discrete head uses the same masked hard states with shrinkage selected on
the same inner holdout. Policies fit exclusively on policy_fit and are evaluated
on the already-inspected validation role. No test records are read.
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random

import torch
from torch.nn import functional as F

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, fresh_dir, write_json
from cbmjev.learning import MaskedHead, encode_states, mask_answers, training_rows
from cbmjev.pipeline import load_model_bundle


COSTS = (0.0, 0.0125, 0.025, 0.03, 0.05, 0.1)
MLP_CONFIGS = ((0.001, 0.0), (0.0003, 0.0))
EPOCHS = (10, 30, 60, 100)
TABULAR_ALPHAS = (1.0, 5.0, 20.0, 100.0)
MASKS = tuple(itertools.product((False, True), repeat=4))
MASK_WEIGHTS = tuple(1.0 / (5 * math.comb(4, sum(mask))) for mask in MASKS)


def family_split(rows):
    families = sorted({row["group_id"] for row in rows},
                      key=lambda group: stable_hash(["cebab-head-inner-v1", group]))
    n_tune = max(1, round(len(families) * 0.2))
    tune_ids = set(families[:n_tune])
    fit = [row for row in rows if row["group_id"] not in tune_ids]
    tune = [row for row in rows if row["group_id"] in tune_ids]
    if not fit or not tune or {r["group_id"] for r in fit} & {r["group_id"] for r in tune}:
        raise ValueError("invalid family-disjoint inner split")
    return fit, tune


def all_mask_tensors(rows, schema):
    states, targets = [], []
    for row in rows:
        for mask in MASKS:
            states.append(mask_answers(row["z"], mask, schema))
            targets.append(row["y"])
    return encode_states(states, schema), torch.tensor(targets, dtype=torch.long)


def weighted_mask_ce(head, x, y):
    losses = F.cross_entropy(head.network(x), y, reduction="none").reshape(-1, 16)
    return (losses * x.new_tensor(MASK_WEIGHTS)).sum(dim=1).mean()


def fit_mlp(rows, schema, original, *, seed, lr, weight_decay, epochs,
            tune_rows=None, checkpoints=()):
    cfg = {**original.config, "device": "cpu", "learning_rate": lr,
           "weight_decay": weight_decay, "dropout": original.config["dropout"]}
    torch.manual_seed(seed)
    head = MaskedHead(schema, cfg)
    x, y = all_mask_tensors(rows, schema)
    xt, yt = all_mask_tensors(tune_rows, schema) if tune_rows is not None else (None, None)
    optimizer = torch.optim.AdamW(head.network.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed + 31)
    checkpoints_out = {}
    for epoch in range(1, epochs + 1):
        head.network.train()
        for order in torch.randperm(len(rows), generator=generator).split(128):
            indices = (order[:, None] * 16 + torch.arange(16)).reshape(-1)
            loss = weighted_mask_ce(head, x[indices], y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch in checkpoints:
            head.network.eval()
            with torch.no_grad():
                checkpoints_out[epoch] = float(weighted_mask_ce(head, xt, yt))
    head.network.eval().requires_grad_(False)
    return head, checkpoints_out


class TabularHead:
    """Smoothed state-to-label counts; root prior is learned on head_fit only."""

    def __init__(self, rows, schema, alpha):
        self.schema, self.alpha = schema, alpha
        self.counts = defaultdict(lambda: [0] * schema.num_classes)
        self.prior = [1] * schema.num_classes
        for row in rows:
            y = row["y"]
            self.prior[y] += 1
            for mask in MASKS:
                self.counts[mask_answers(row["z"], mask, schema)][y] += 1
        total = sum(self.prior)
        self.prior = tuple(n / total for n in self.prior)

    def probabilities_many(self, states):
        return tuple(self.probabilities(state) for state in states)

    def probabilities(self, state):
        counts = self.counts.get(tuple(state), [0] * self.schema.num_classes)
        denom = sum(counts) + self.alpha
        return tuple((counts[i] + self.alpha * self.prior[i]) / denom
                     for i in range(self.schema.num_classes))

    def predict(self, state):
        p = self.probabilities(state)
        return max(range(len(p)), key=p.__getitem__)


def mask_nll(rows, head, schema):
    total = 0.0
    for mask in MASKS:
        states = [mask_answers(row["z"], mask, schema) for row in rows]
        total += MASK_WEIGHTS[MASKS.index(mask)] * sum(
            -math.log(max(1e-12, p[row["y"]]))
            for row, p in zip(rows, head.probabilities_many(states))) / len(rows)
    return total


def select_heads(head_rows, schema, legacy, seed):
    if len(MLP_CONFIGS) * len(EPOCHS) > 8 or len(TABULAR_ALPHAS) > 8:
        raise ValueError("new head tuning budget exceeds eight candidates")
    fit, tune = family_split(head_rows)
    searches = []
    for lr, wd in MLP_CONFIGS:
        _, scores = fit_mlp(fit, schema, legacy, seed=seed, lr=lr,
                            weight_decay=wd, epochs=max(EPOCHS),
                            tune_rows=tune, checkpoints=EPOCHS)
        searches.extend({"learning_rate": lr, "weight_decay": wd,
                         "epochs": ep, "inner_mask_nll": score}
                        for ep, score in scores.items())
    selected = min(searches, key=lambda row: (row["inner_mask_nll"],
                    row["epochs"], row["learning_rate"], row["weight_decay"]))
    mlp, _ = fit_mlp(head_rows, schema, legacy, seed=seed,
                     lr=selected["learning_rate"], weight_decay=selected["weight_decay"],
                     epochs=selected["epochs"])
    tab_scores = [{"alpha": alpha,
                   "inner_mask_nll": mask_nll(tune, TabularHead(fit, schema, alpha), schema)}
                  for alpha in TABULAR_ALPHAS]
    tab_selected = min(tab_scores, key=lambda row: (row["inner_mask_nll"], row["alpha"]))
    tab = TabularHead(head_rows, schema, tab_selected["alpha"])
    return {"legacy": legacy, "exhaustive_mlp": mlp, "shrunk_tabular": tab}, {
        "fit_rows": len(fit), "tune_rows": len(tune),
        "fit_families": len({r["group_id"] for r in fit}),
        "tune_families": len({r["group_id"] for r in tune}),
        "inner_fit_group_hash": stable_hash(sorted({r["group_id"] for r in fit})),
        "inner_tune_group_hash": stable_hash(sorted({r["group_id"] for r in tune})),
        "mask_objective": "uniform cardinality, uniform subset: all 16 exact weighted masks",
        "mlp_candidates": searches, "mlp_selected": selected,
        "tabular_candidates": tab_scores, "tabular_selected": tab_selected}


def metrics(rows, states, head, schema):
    probs = head.probabilities_many(states)
    predictions = [max(range(schema.num_classes), key=p.__getitem__) for p in probs]
    correct = sum(p == row["y"] for p, row in zip(predictions, rows))
    f1s = []
    for cls in range(schema.num_classes):
        tp = sum(p == cls and r["y"] == cls for p, r in zip(predictions, rows))
        fp = sum(p == cls and r["y"] != cls for p, r in zip(predictions, rows))
        fn = sum(p != cls and r["y"] == cls for p, r in zip(predictions, rows))
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"n": len(rows), "error": 1 - correct / len(rows),
            "accuracy": correct / len(rows), "macro_f1": sum(f1s) / len(f1s),
            "nll": sum(-math.log(max(1e-12, p[r["y"]]))
                       for p, r in zip(probs, rows)) / len(rows)}


def fit_empirical(rows, head, schema):
    """The same row-weighted finite empirical conditional model as prior E0."""
    states = {mask_answers(row["z"], mask, schema)
              for row in rows for mask in MASKS}
    states.add(schema.empty_state())
    support = {}
    for state in states:
        indices = [i for i, row in enumerate(rows)
                   if all(value < 0 or row["z"][j] == value
                          for j, value in enumerate(state))]
        support[state] = indices or list(range(len(rows)))
    @lru_cache(None)
    def records(state):
        if state not in support:
            indices = [i for i, row in enumerate(rows)
                       if all(value < 0 or row["z"][j] == value
                              for j, value in enumerate(state))]
            support[state] = indices or list(range(len(rows)))
        return support[state]
    @lru_cache(None)
    def risk(state):
        prediction = head.predict(state)
        indices = records(state)
        return sum(rows[i]["y"] != prediction for i in indices) / len(indices)
    @lru_cache(None)
    def branches(state, group):
        indices = records(state)
        count = Counter(rows[i]["z"][group] for i in indices)
        return tuple((state[:group] + (value,) + state[group + 1:], n / len(indices))
                     for value, n in sorted(count.items()))
    return risk, branches


def make_policy(risk, branches, *, cost, order=None):
    @lru_cache(None)
    def solve(state):
        options = [(risk(state), None)]
        remaining = [j for j, v in enumerate(state) if v < 0]
        if order is not None:
            remaining = [next(j for j in order if j in remaining)] if remaining else []
        for group in remaining:
            value = cost + sum(prob * solve(after)[0]
                               for after, prob in branches(state, group))
            options.append((value, group))
        return min(options, key=lambda choice: (choice[0], choice[1] is not None,
                                                choice[1] if choice[1] is not None else -1))
    return solve


def policy_states(rows, schema, policy):
    states, queries = [], []
    for row in rows:
        state = schema.empty_state()
        count = 0
        while True:
            _, group = policy(state)
            if group is None:
                break
            state = state[:group] + (row["z"][group],) + state[group + 1:]
            count += 1
            if count > 4:
                raise RuntimeError("policy did not terminate")
        states.append(state)
        queries.append(count)
    return states, queries


def evaluate_policy(rows, head, schema, policy, cost):
    states, queries = policy_states(rows, schema, policy)
    out = metrics(rows, states, head, schema)
    out["mean_queries"] = sum(queries) / len(queries)
    out["J"] = out["error"] + cost * out["mean_queries"]
    return out


def paired_family_interval(rows, head, schema, static, dynamic, cost, seed,
                           resamples=1000):
    states_s, queries_s = policy_states(rows, schema, static)
    states_d, queries_d = policy_states(rows, schema, dynamic)
    ps = head.probabilities_many(states_s)
    pd = head.probabilities_many(states_d)
    by_family = defaultdict(lambda: [0.0, 0])
    for row, left, right, qs, qd in zip(rows, ps, pd, queries_s, queries_d):
        pred_s = max(range(schema.num_classes), key=left.__getitem__)
        pred_d = max(range(schema.num_classes), key=right.__getitem__)
        delta = (pred_s != row["y"]) - (pred_d != row["y"]) + cost * (qs - qd)
        by_family[row["group_id"]][0] += delta
        by_family[row["group_id"]][1] += 1
    families = sorted(by_family)
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sample = rng.choices(families, k=len(families))
        draws.append(sum(by_family[f][0] for f in sample) /
                     sum(by_family[f][1] for f in sample))
    draws.sort()
    return {"unit": "family_cluster", "n_families": len(families),
            "resamples": resamples, "seed": seed,
            "point": sum(value[0] for value in by_family.values()) / len(rows),
            "percentile_95": [draws[int(0.025 * resamples)],
                              draws[int(0.975 * resamples) - 1]],
            "scope": "descriptive conditional on frozen heads; not seed or model uncertainty"}


def evaluate_head(head, head_rows, policy_rows, val_rows, schema, *, bootstrap_seed):
    full = lambda rows: [tuple(row["z"]) for row in rows]
    report = {"head_fit": metrics(head_rows, full(head_rows), head, schema),
              "policy_fit": metrics(policy_rows, full(policy_rows), head, schema),
              "validation": metrics(val_rows, full(val_rows), head, schema),
              "head_fit_mask_nll": mask_nll(head_rows, head, schema),
              "validation_mask_nll": mask_nll(val_rows, head, schema), "costs": []}
    risk, branches = fit_empirical(policy_rows, head, schema)
    root = schema.empty_state()
    orders = tuple(itertools.permutations(range(4)))
    for cost_index, cost in enumerate(COSTS):
        dynamic = make_policy(risk, branches, cost=cost)
        fixed = [(order, make_policy(risk, branches, cost=cost, order=order))
                 for order in orders]
        # The policy_fit empirical root objective chooses the order; validation
        # labels are never consulted in order or hyperparameter selection.
        # Floating summation of the same empirical expectation can differ by
        # ~1e-16 across equivalent orders. The legacy evaluator broke exact
        # objective ties lexicographically after policy_fit replay.
        order, chosen = min(fixed, key=lambda entry: (round(entry[1](root)[0], 12), entry[0]))
        fitted_static, fitted_dynamic = chosen(root)[0], dynamic(root)[0]
        val_static = evaluate_policy(val_rows, head, schema, chosen, cost)
        val_dynamic = evaluate_policy(val_rows, head, schema, dynamic, cost)
        interval = paired_family_interval(val_rows, head, schema, chosen,
            dynamic, cost, bootstrap_seed + cost_index)
        fitted_static_observed = evaluate_policy(policy_rows, head, schema, chosen, cost)
        fitted_dynamic_observed = evaluate_policy(policy_rows, head, schema, dynamic, cost)
        val_all = metrics(val_rows, full(val_rows), head, schema)
        val_all["mean_queries"] = 4.0
        val_all["J"] = val_all["error"] + 4 * cost
        report["costs"].append({"cost_per_query": cost, "selected_fixed_order": order,
            "fitted_static_J": fitted_static, "fitted_dynamic_J": fitted_dynamic,
            "fitted_static_observed": fitted_static_observed,
            "fitted_dynamic_observed": fitted_dynamic_observed,
            "validation_static": val_static, "validation_dynamic": val_dynamic,
            "validation_all": val_all,
            "family_bootstrap_static_minus_dynamic_J": interval,
            "validation_static_minus_dynamic_J": val_static["J"] - val_dynamic["J"]})
    return report


def gold_reference(prepared, rows, schema, manifest, automatic_head):
    """Complete public annotations only; report selection and information shift."""
    samples_path = Path(prepared) / "samples.jsonl"
    if file_hash(samples_path) != manifest["prepared_samples_sha256"]:
        raise ValueError("prepared samples differ from response-cache ancestry")
    by_id = {row["sample_id"]: row for row in rows}
    gold = []
    text_rows = []
    with samples_path.open() as stream:
        for line in stream:
            sample = json.loads(line)
            cached = by_id.get(sample["sample_id"])
            if cached is None:
                continue
            if cached["split"] not in {"head_fit", "policy_fit", "validation"}:
                raise ValueError("forbidden cache role in information reference")
            if sample["split"] != ("validation" if cached["split"] == "validation" else "train"):
                raise ValueError("prepared/cache outer split mismatch")
            if sample["group_id"] != cached["group_id"] or sample["target"]["value"] != cached["y"]:
                raise ValueError("gold/automatic sample identity or target differs")
            payload = sample["input"]
            if payload.get("modality") != "text" or not isinstance(payload.get("text"), str):
                raise ValueError("expected text-only prepared CEBaB sample")
            if cached["split"] in {"head_fit", "validation"}:
                text_rows.append({"sample_id": cached["sample_id"], "split": cached["split"],
                                  "y": cached["y"], "text": payload["text"]})
            concepts = {item["concept_id"]: item for item in sample["concepts"]}
            if set(concepts) != {concept.id for concept in schema.concepts}:
                raise ValueError("prepared gold concept schema differs")
            if all(item["annotation_status"] == "OBSERVED" for item in concepts.values()):
                z = [concepts[concept.id]["value"] for concept in schema.concepts]
                if any(type(value) is not int or not 0 <= value < 3 for value in z):
                    raise ValueError("gold annotation value differs from categorical schema")
                gold.append({**cached, "z": z})
    selected = {role: [row for row in gold if row["split"] == role]
                for role in ("head_fit", "policy_fit", "validation")}
    if not selected["head_fit"] or not selected["validation"]:
        raise ValueError("no complete public gold subset in fit/validation")
    gold_head = TabularHead(selected["head_fit"], schema, alpha=20.0)
    same_ids = {row["sample_id"] for row in selected["validation"]}
    auto_subset = [row for row in rows if row["split"] == "validation"
                   and row["sample_id"] in same_ids]
    return {
        "status": "OBSERVED_COMPLETE_GOLD_SUBSET_ONLY",
        "training_role": "head_fit", "fixed_gold_alpha": 20.0,
        "coverage": {role: {"complete_gold": len(selected[role]),
                            "all_cached": sum(row["split"] == role for row in rows)}
                     for role in selected},
        "gold_head_fit": metrics(selected["head_fit"],
                                 [tuple(row["z"]) for row in selected["head_fit"]],
                                 gold_head, schema),
        "gold_validation": metrics(selected["validation"],
                                   [tuple(row["z"]) for row in selected["validation"]],
                                   gold_head, schema),
        "automatic_same_rows": metrics(auto_subset,
                                       [tuple(row["z"]) for row in auto_subset],
                                       automatic_head, schema),
        "limitations": ["Gold and automatic head fit on different information and populations.",
                        "Complete gold annotations are selected, so this is not a full-population oracle bound."]}, text_rows


def text_reference(text_rows, num_classes):
    """Fixed-config character TF-IDF plus multinomial linear classifier."""
    fit = [row for row in text_rows if row["split"] == "head_fit"]
    val = [row for row in text_rows if row["split"] == "validation"]
    if not fit or not val:
        raise ValueError("direct-text reference requires both protected roles")
    def features(text):
        text = " ".join(text.lower().split())
        return Counter(text[i:i+n] for n in (2, 3)
                       for i in range(max(0, len(text) - n + 1)))
    fit_counts = [features(row["text"]) for row in fit]
    val_counts = [features(row["text"]) for row in val]
    df = Counter(term for counts in fit_counts for term in counts)
    vocabulary = {term: i for i, (term, count) in enumerate(
        sorted(((term, count) for term, count in df.items() if count >= 2),
               key=lambda pair: (-pair[1], pair[0]))[:8192])}
    if not vocabulary:
        raise ValueError("empty text vocabulary")
    idf = {term: math.log((len(fit) + 1) / (df[term] + 1)) + 1
           for term in vocabulary}
    def encode(records):
        x = torch.zeros((len(records), len(vocabulary)), dtype=torch.float32)
        for i, counts in enumerate(records):
            for term, n in counts.items():
                j = vocabulary.get(term)
                if j is not None:
                    x[i, j] = (1 + math.log(n)) * idf[term]
        return F.normalize(x, p=2, dim=1)
    xfit, xval = encode(fit_counts), encode(val_counts)
    yfit = torch.tensor([row["y"] for row in fit], dtype=torch.long)
    torch.manual_seed(20260929)
    classifier = torch.nn.Linear(len(vocabulary), num_classes)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.05, weight_decay=0.01)
    for _ in range(50):
        classifier.train()
        loss = F.cross_entropy(classifier(xfit), yfit)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    classifier.eval()
    with torch.no_grad():
        pfit = classifier(xfit).softmax(-1).tolist()
        pval = classifier(xval).softmax(-1).tolist()
    def report(records, probs):
        preds = [max(range(num_classes), key=p.__getitem__) for p in probs]
        f1s = []
        for cls in range(num_classes):
            tp = sum(p == cls and r["y"] == cls for p, r in zip(preds, records))
            fp = sum(p == cls and r["y"] != cls for p, r in zip(preds, records))
            fn = sum(p != cls and r["y"] == cls for p, r in zip(preds, records))
            f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        return {"n": len(records), "accuracy": sum(p == r["y"] for p, r in zip(preds, records)) / len(records),
                "macro_f1": sum(f1s) / num_classes,
                "nll": sum(-math.log(max(1e-12, p[r["y"]]))
                           for p, r in zip(probs, records)) / len(records)}
    return {"status": "FIXED_CONFIG_DEVELOPMENT_REFERENCE",
            "model": "char_2_3gram_tfidf_multinomial_linear",
            "min_document_frequency": 2, "max_features": 8192,
            "vocabulary_size": len(vocabulary), "epochs": 50,
            "learning_rate": 0.05, "weight_decay": 0.01,
            "optimizer": "AdamW", "seed": 20260929,
            "fit_role": "head_fit", "validation_role": "inspected_development_validation",
            "fit": report(fit, pfit), "validation": report(val, pval),
            "limitations": ["Different information and architecture from CBM; reference, not equal-information policy baseline.",
                            "Fixed small linear model, not a tuned transformer or text Bayes bound."]}


def run(models, cache, prepared, out, *, cpu_threads=2):
    torch.set_num_threads(cpu_threads)
    schema, rows, manifest, config, receipt, legacy, _, _ = load_model_bundle(
        models, cache, device="cpu")
    if schema.dataset != "cebab" or schema.num_groups != 4 or schema.num_atoms != 4:
        raise ValueError("requires four singleton CEBaB groups")
    excluded_roles = Counter(row["split"] for row in rows
                             if row["split"] not in {"head_fit", "policy_fit", "validation"})
    allowed = {"head_fit", "policy_fit", "validation"}
    rows = [row for row in rows if row["split"] in allowed]
    if not rows or any(row["split"] in {"test", "calibration", "responder_fit"} for row in rows):
        raise ValueError("protected roles entered evaluation")
    head_rows = [r for r in rows if r["split"] == "head_fit"]
    policy_rows = [r for r in rows if r["split"] == "policy_fit"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    for role_rows in (head_rows, policy_rows, val_rows):
        if not role_rows:
            raise ValueError("missing data role")
    for role in ("head_fit", "policy_fit"):
        training_rows(rows, role, schema)
    family_sets = [{r["group_id"] for r in role_rows}
                   for role_rows in (head_rows, policy_rows, val_rows)]
    if any(family_sets[i] & family_sets[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError("family leaks across roles")
    heads, selection = select_heads(head_rows, schema, legacy, config["seed"])
    gold_report, text_rows = gold_reference(prepared, rows, schema, manifest, legacy)
    text_report = text_reference(text_rows, schema.num_classes)
    mlp_payload = b"".join(value.detach().cpu().contiguous().numpy().tobytes()
                           for _, value in sorted(heads["exhaustive_mlp"].network.state_dict().items()))
    result = {"format": "cebab-predictor-sufficiency-v1", "seed": config["seed"],
      "evidence_status": "INSPECTED_DEVELOPMENT_VALIDATION_NOT_TEST",
      "roles": {name: {"rows": len(role_rows), "families": len(family_sets[i]),
                       "family_hash": stable_hash(sorted(family_sets[i]))}
                for i, (name, role_rows) in enumerate(zip(
                    ("head_fit", "policy_fit", "validation"),
                    (head_rows, policy_rows, val_rows)))},
      "excluded_cache_role_counts": dict(excluded_roles),
      "selection": selection,
      "results": {name: evaluate_head(head, head_rows, policy_rows, val_rows,
                                       schema, bootstrap_seed=config["seed"] * 1000 + i * 10)
                  for i, (name, head) in enumerate(heads.items())},
      "information_references": {
          "gold": gold_report, "direct_text": text_report},
      "bindings": {"legacy_receipt_sha256": file_hash(Path(models) / "receipt.json"),
                   "cache_manifest_sha256": file_hash(Path(cache) / "manifest.json"),
                   "cache_responses_sha256": manifest["responses_sha256"],
                   "prepared_samples_sha256": file_hash(Path(prepared) / "samples.jsonl"),
                   "legacy_model_sha256": receipt["models_sha256"],
                   "script_sha256": file_hash(__file__),
                   "fitted_mlp_tensor_sha256": hashlib.sha256(mlp_payload).hexdigest(),
                   "tabular_count_hash": stable_hash({str(state): counts for state, counts in
                        sorted(heads["shrunk_tabular"].counts.items())})},
      "limitations": ["Finite row-weighted policy_fit empirical DP, not population Bayes bound.",
                      "Validation has been inspected historically and is development only.",
                      "Bundle loading structurally parses all cache rows, including test; protected labels are never selected for fitting, tuning, or metrics.",
                      "Equal-information policy ranking uses hard automatic concepts only; gold and text are separate information references."]}
    out = fresh_dir(out)
    write_json(out / "report.json", result)
    write_json(out / "run_receipt.json", {
        "format": "cebab-predictor-sufficiency-receipt-v1",
        "seed": config["seed"], "script_sha256": file_hash(__file__),
        "report_sha256": file_hash(out / "report.json"),
        "legacy_receipt_sha256": result["bindings"]["legacy_receipt_sha256"],
        "cache_manifest_sha256": result["bindings"]["cache_manifest_sha256"],
        "cache_responses_sha256": result["bindings"]["cache_responses_sha256"],
        "fitted_mlp_tensor_sha256": result["bindings"]["fitted_mlp_tensor_sha256"],
        "tabular_count_hash": result["bindings"]["tabular_count_hash"],
        "inner_fit_group_hash": selection["inner_fit_group_hash"],
        "inner_tune_group_hash": selection["inner_tune_group_hash"],
        "head_fit_group_hash": result["roles"]["head_fit"]["family_hash"],
        "policy_fit_group_hash": result["roles"]["policy_fit"]["family_hash"],
        "validation_group_hash": result["roles"]["validation"]["family_hash"],
        "mlp_tune_candidates": len(selection["mlp_candidates"]),
        "tabular_tune_candidates": len(selection["tabular_candidates"]),
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "device": "cpu", "test_evaluated": False,
        "model_dir": str(Path(models).resolve()), "cache_dir": str(Path(cache).resolve()),
        "prepared_dir": str(Path(prepared).resolve()),
        "reproduction_command": (
            "PYTHONPATH=. .venv/bin/python " + str(Path(__file__).resolve()) +
            " --models " + str(models) +
            " --cache " + str(cache) + " --prepared " + str(prepared) +
            " --out " + str(out) +
            " --cpu-threads " + str(cpu_threads))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "cache", "prepared", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    report = run(args.models, args.cache, args.prepared, args.out,
                 cpu_threads=args.cpu_threads)
    print(json.dumps({"seed": report["seed"], "out": args.out,
                      "mlp_selected": report["selection"]["mlp_selected"],
                      "tabular_selected": report["selection"]["tabular_selected"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()

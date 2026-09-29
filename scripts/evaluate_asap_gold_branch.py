#!/usr/bin/env python3
"""ASAP 18-aspect gold-acquisition diagnostic (development data only).

The adaptive control implements DIME's cross-entropy loss-improvement target
(Gadgil et al., arXiv:2306.03301, Eq. 3), with a frozen separately fitted
partial-input predictor. It is an adaptation, not the authors' joint training
or a claimed faithful reproduction of their published benchmark numbers.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbmjev.asap_data import ASPECTS, file_sha256, load_development


K = len(ASPECTS)
BUDGETS = (0, 2, 4, 8, 12, 18)
LAMBDAS = (0., .05, .10, .20, .40)
THRESHOLD_FACTORS = (0., .25, .5, 1., 2., 4.)


def internal_role(group_id):
    digest = hashlib.sha256(("asap-reopen-b-v1|" + group_id).encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    return "head_fit" if u < .70 else ("policy_fit" if u < .85 else "tune")


def arrays(rows):
    return (np.asarray([r["z"] for r in rows], dtype=np.int8),
            np.asarray([r["y"] for r in rows], dtype=np.int64),
            [r["group_id"] for r in rows])


def encode(z, mask):
    if z.shape != mask.shape or z.shape[-1] != K:
        raise ValueError("ASAP aspect values and acquired mask must be Nx18")
    categories = np.where(mask, z.astype(np.int16) + 3, 0)
    return np.eye(5, dtype=np.float32)[categories].reshape(-1, K * 5)


def sampled_masks(n, rng, cardinalities=BUDGETS):
    sizes = rng.choice(cardinalities, size=n)
    ordering = np.argsort(rng.random((n, K)), axis=1)
    mask = np.zeros((n, K), dtype=bool)
    mask[np.arange(n)[:, None], ordering] = np.arange(K)[None, :] < sizes[:, None]
    return mask


class PartialHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(K * 5, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5))

    def forward(self, x):
        return self.net(x)


class GainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(K * 5, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, K))

    def forward(self, x):
        return self.net(x)


def batched_logits(model, features, batch=2048):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(features), batch):
            x = torch.from_numpy(features[start:start + batch])
            out.append(model(x).numpy())
    return np.concatenate(out, axis=0)


def head_probabilities(head, z, mask):
    logits = batched_logits(head, encode(z, mask))
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def fit_head(train, tune, seed, *, epochs=12):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    z, y, _ = train
    tz, ty, _ = tune
    head = PartialHead()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
    tune_rng = np.random.default_rng(14001)
    tune_masks = np.concatenate([sampled_masks(len(tz), tune_rng) for _ in range(3)])
    tune_z = np.tile(tz, (3, 1))
    tune_y = np.tile(ty, 3)
    tx = torch.from_numpy(encode(tune_z, tune_masks))
    tl = torch.from_numpy(tune_y)
    best, best_epoch, best_state = float("inf"), -1, None
    history = []
    for epoch in range(epochs):
        head.train()
        masks = np.concatenate([sampled_masks(len(z), rng) for _ in range(2)])
        current_z = np.tile(z, (2, 1))
        current_y = np.tile(y, 2)
        x = torch.from_numpy(encode(current_z, masks))
        target = torch.from_numpy(current_y)
        perm = torch.from_numpy(rng.permutation(len(x)))
        losses = []
        for batch in perm.split(1024):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(x[batch]), target[batch])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        head.eval()
        with torch.no_grad():
            tune_loss = float(nn.functional.cross_entropy(head(tx), tl))
        history.append({"epoch": epoch + 1, "train_batch_ce_mean": float(np.mean(losses)),
                        "tune_ce": tune_loss})
        if tune_loss < best:
            best, best_epoch = tune_loss, epoch + 1
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, {"best_epoch": best_epoch, "best_tune_ce": best, "history": history}


def gain_targets(head, z, y, masks, *, chunk=128):
    """DIME Eq. 3 targets: CE before minus CE after each legal acquisition."""
    targets = np.zeros((len(z), K), dtype=np.float32)
    eye = np.eye(K, dtype=bool)
    for start in range(0, len(z), chunk):
        end = min(start + chunk, len(z))
        zz, yy, mm = z[start:end], y[start:end], masks[start:end]
        before = head_probabilities(head, zz, mm)
        base = -np.log(np.maximum(before[np.arange(len(zz)), yy], 1e-12))
        after_masks = (mm[:, None, :] | eye[None, :, :]).reshape(-1, K)
        after_z = np.repeat(zz, K, axis=0)
        after = head_probabilities(head, after_z, after_masks)
        yy_after = np.repeat(yy, K)
        post = -np.log(np.maximum(after[np.arange(len(after_z)), yy_after], 1e-12))
        targets[start:end] = base[:, None] - post.reshape(len(zz), K)
    targets[masks] = 0.
    return targets


def fit_gain(head, policy_fit, tune, seed, *, epochs=12):
    torch.manual_seed(seed + 10000)
    rng = np.random.default_rng(seed + 10000)
    z, y, _ = policy_fit
    tz, ty, _ = tune
    sizes = (0, 2, 4, 8, 12, 16)
    masks = np.concatenate([sampled_masks(len(z), rng, sizes) for _ in range(3)])
    zz, yy = np.tile(z, (3, 1)), np.tile(y, 3)
    targets = gain_targets(head, zz, yy, masks)
    x = torch.from_numpy(encode(zz, masks))
    target = torch.from_numpy(targets)
    legal = torch.from_numpy(~masks)
    trng = np.random.default_rng(15001)
    tmasks = sampled_masks(len(tz), trng, sizes)
    ttarget = torch.from_numpy(gain_targets(head, tz, ty, tmasks))
    tx = torch.from_numpy(encode(tz, tmasks))
    tlegal = torch.from_numpy(~tmasks)
    net = GainNet()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0007, weight_decay=.0003)
    best, best_epoch, best_state = float("inf"), -1, None
    history = []
    for epoch in range(epochs):
        net.train()
        perm = torch.from_numpy(rng.permutation(len(x)))
        losses = []
        for batch in perm.split(1024):
            optimizer.zero_grad(set_to_none=True)
            delta = (net(x[batch]) - target[batch]) ** 2
            loss = (delta * legal[batch]).sum() / legal[batch].sum()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        net.eval()
        with torch.no_grad():
            delta = (net(tx) - ttarget) ** 2
            tune_loss = float((delta * tlegal).sum() / tlegal.sum())
        history.append({"epoch": epoch + 1, "train_batch_mse_mean": float(np.mean(losses)),
                        "tune_mse": tune_loss})
        if tune_loss < best:
            best, best_epoch = tune_loss, epoch + 1
            best_state = {key: value.detach().clone() for key, value in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, {"best_epoch": best_epoch, "best_tune_mse": best, "history": history}


def summarize(y, pred, count, groups, lam):
    n = len(y)
    error = float(np.mean(pred != y))
    f1s = []
    for cls in range(5):
        tp = int(np.sum((pred == cls) & (y == cls)))
        fp = int(np.sum((pred == cls) & (y != cls)))
        fn = int(np.sum((pred != cls) & (y == cls)))
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    return {"n": n, "families": len(set(groups)), "accuracy": 1 - error,
            "macro_f1": float(np.mean(f1s)), "mae": float(np.mean(np.abs(pred - y))),
            "mean_queries": float(np.mean(count)), "error": error,
            "J": error + lam * float(np.mean(count)) / K}


def evaluate_mask(head, data, masks, lam):
    z, y, groups = data
    pred = np.argmax(head_probabilities(head, z, masks), axis=1)
    count = masks.sum(axis=1)
    return summarize(y, pred, count, groups, lam), pred


def greedy_orders(head, policy_fit, *, starts=4, excluded=None):
    """Four deterministic greedy candidates; finite search, never called optimal."""
    z, y, _ = policy_fit
    empty = np.zeros_like(z, dtype=bool)
    legal = [a for a in range(K) if a != excluded]
    first = []
    for aspect in legal:
        trial = empty.copy()
        trial[:, aspect] = True
        report, _ = evaluate_mask(head, policy_fit, trial, 0.)
        first.append((report["error"], aspect))
    orders = []
    for _, root in sorted(first)[:starts]:
        order = [root]
        for _ in range(len(legal) - 1):
            mask = np.zeros_like(z, dtype=bool)
            mask[:, order] = True
            options = []
            for aspect in legal:
                if aspect not in order:
                    trial = mask.copy()
                    trial[:, aspect] = True
                    report, _ = evaluate_mask(head, policy_fit, trial, 0.)
                    options.append((report["error"], aspect))
            order.append(min(options)[1])
        orders.append(tuple(order))
    return orders, {"candidate_count": starts, "rule": "top_starts_first_then_greedy_prefix_0_1",
                    "selection_role": "policy_fit", "excluded_aspect": excluded,
                    "orders": [list(o) for o in orders]}


def rollout(data, head, gain, *, method, budget, threshold, order=None,
            excluded=None, random_seed=None, return_paths=False):
    z, _, _ = data
    mask = np.zeros_like(z, dtype=bool)
    paths = np.full((len(z), K), -1, dtype=np.int8) if return_paths else None
    active = np.ones(len(z), dtype=bool)
    rng = np.random.default_rng(random_seed) if method == "random" else None
    priorities = np.argsort(rng.random(z.shape), axis=1) if rng is not None else None
    permitted = np.ones(K, dtype=bool)
    if excluded is not None:
        permitted[excluded] = False
    budget = min(budget, int(permitted.sum()))
    for step in range(budget):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        if method == "random":
            action = priorities[ids, step]
            if excluded is not None:
                # Re-rank so the excluded concept is never acquired.
                action = np.asarray([next(a for a in priorities[i] if permitted[a] and not mask[i, a])
                                     for i in ids], dtype=int)
        elif method == "fixed":
            action = np.full(len(ids), next(a for a in order if permitted[a] and not mask[ids[0], a]),
                             dtype=int)
        elif method == "adaptive":
            scores = batched_logits(gain, encode(z[ids], mask[ids]))
            scores[:, ~permitted] = -np.inf
            scores[mask[ids]] = -np.inf
            action = np.argmax(scores, axis=1)
        else:
            raise ValueError("unknown rollout method")
        if method == "fixed" and threshold is not None:
            scores = batched_logits(gain, encode(z[ids], mask[ids]))
            keep = scores[np.arange(len(ids)), action] > threshold
        elif method == "adaptive" and threshold is not None:
            keep = scores[np.arange(len(ids)), action] > threshold
        else:
            keep = np.ones(len(ids), dtype=bool)
        active[ids[~keep]] = False
        mask[ids[keep], action[keep]] = True
        if return_paths:
            paths[ids[keep], step] = action[keep]
    return (mask, paths) if return_paths else mask


def family_bootstrap_delta(groups, y, fixed_pred, fixed_mask,
                           adaptive_pred, adaptive_mask, lam, seed, replicates=1000):
    """Row-weighted paired delta; resampling unit is normalized review text."""
    index = {group: j for j, group in enumerate(sorted(set(groups)))}
    group_ids = np.asarray([index[group] for group in groups])
    error_delta = (adaptive_pred != y).astype(float) - (fixed_pred != y).astype(float)
    cost_delta = (adaptive_mask.sum(axis=1) - fixed_mask.sum(axis=1)) / K
    j_delta = error_delta + lam * cost_delta
    counts = np.bincount(group_ids, minlength=len(index))
    err_sums = np.bincount(group_ids, weights=error_delta, minlength=len(index))
    j_sums = np.bincount(group_ids, weights=j_delta, minlength=len(index))
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(index), size=(replicates, len(index)))
    denominators = counts[sample].sum(axis=1)
    err_draws = err_sums[sample].sum(axis=1) / denominators
    j_draws = j_sums[sample].sum(axis=1) / denominators
    return {"unit": "normalized_exact_text_component", "replicates": replicates,
            "adaptive_minus_fixed_error": float(np.mean(error_delta)),
            "adaptive_minus_fixed_J": float(np.mean(j_delta)),
            "error_ci95": np.quantile(err_draws, (.025, .975)).tolist(),
            "J_ci95": np.quantile(j_draws, (.025, .975)).tolist()}


def trajectory_audit(paths, z):
    """Observed-answer-conditioned branching, including STOP as an action."""
    prefixes = defaultdict(lambda: defaultdict(set))
    sequences = Counter()
    for row, sequence in enumerate(paths):
        action_path = tuple(int(action) for action in sequence if action >= 0)
        sequences[action_path] += 1
        for depth in range(K):
            if depth and sequence[depth - 1] < 0:
                break
            prefix = tuple(int(a) for a in sequence[:depth])
            response = tuple(int(z[row, a]) for a in prefix)
            next_action = int(sequence[depth])
            prefixes[prefix][response].add(next_action)
            if next_action < 0:
                break
    branchable = []
    for prefix, by_response in prefixes.items():
        if len(by_response) < 2:
            continue
        decisions = set().union(*by_response.values())
        if len(decisions) > 1:
            branchable.append({"depth": len(prefix), "prefix": list(prefix),
                               "observed_value_patterns": len(by_response),
                               "next_actions": sorted(decisions)})
    return {"unique_acquisition_sequences": len(sequences),
            "first_action_counts": dict(Counter(int(row[0]) for row in paths)),
            "branching_prefix_count": len(branchable),
            "branching_prefixes_by_depth": dict(Counter(row["depth"] for row in branchable)),
            "branching_examples": branchable[:12]}


def select_stopping(data, head, gain, orders, method, lam, excluded=None):
    """Select order and CE-gain threshold on internal tune, never official dev."""
    trials = []
    options = orders if method == "fixed" else (None,)
    for order in options:
        for factor in THRESHOLD_FACTORS:
            threshold = factor * lam / K
            mask = rollout(data, head, gain, method=method, budget=K,
                           threshold=threshold, order=order, excluded=excluded)
            report, _ = evaluate_mask(head, data, mask, lam)
            trials.append((report["J"], report["mean_queries"], factor,
                           tuple(order) if order else (), report))
    chosen = min(trials, key=lambda item: (item[0], item[1], item[2], item[3]))
    return {"order": list(chosen[3]) if method == "fixed" else None,
            "factor": chosen[2], "threshold": chosen[2] * lam / K,
            "tune": chosen[4], "num_candidate_policies": len(trials)}


def evaluate_policy_grid(data, head, gain, orders, selections, seed, *, excluded=None):
    results = {"budget": {}, "cost": {}, "paired": {}, "trajectories": {}}
    z, y, groups = data
    for budget in BUDGETS:
        traces = {}
        for method in ("fixed", "adaptive", "random"):
            order = orders[0] if method == "fixed" else None
            if method == "fixed":
                # All fixed candidates are selected using internal tune accuracy.
                choices = []
                for candidate in orders:
                    mask = rollout(selections["tune_data"], head, gain, method="fixed",
                                   budget=budget, threshold=None, order=candidate,
                                   excluded=excluded)
                    rr, _ = evaluate_mask(head, selections["tune_data"], mask, 0.)
                    choices.append((rr["error"], candidate))
                order = min(choices)[1]
            mask, paths = rollout(data, head, gain, method=method, budget=budget,
                           threshold=None, order=order, excluded=excluded,
                           random_seed=seed + budget * 100 if method == "random" else None,
                           return_paths=True)
            report, pred = evaluate_mask(head, data, mask, 0.)
            traces[method] = (mask, pred, paths)
            results["budget"]["{}:{}".format(method, budget)] = {
                "report": report, "order": list(order) if order else None}
        fm, fp, fpaths = traces["fixed"]
        am, ap, apaths = traces["adaptive"]
        results["paired"]["budget:{}".format(budget)] = family_bootstrap_delta(
            groups, y, fp, fm, ap, am, 0., seed + 1000 + budget)
        results["trajectories"]["budget:{}".format(budget)] = {
            "fixed": trajectory_audit(fpaths, z),
            "adaptive": trajectory_audit(apaths, z),
            "fraction_different_acquisition_order": float(np.mean(np.any(fpaths != apaths, axis=1))),
            "fraction_different_acquired_sets": float(np.mean(np.any(fm != am, axis=1))),
            "fraction_different_predictions": float(np.mean(fp != ap)),
            "adaptive_correct_fixed_wrong": int(np.sum((ap == y) & (fp != y))),
            "fixed_correct_adaptive_wrong": int(np.sum((fp == y) & (ap != y)))}
    for lam in LAMBDAS:
        traces = {}
        for method in ("fixed", "adaptive"):
            selected = selections[(method, lam)]
            mask, paths = rollout(data, head, gain, method=method, budget=K,
                           threshold=selected["threshold"], order=selected["order"],
                           excluded=excluded, return_paths=True)
            report, pred = evaluate_mask(head, data, mask, lam)
            traces[method] = (mask, pred, paths)
            results["cost"]["{}:{}".format(method, lam)] = {
                "report": report, "selected_tune": selected}
        all_mask = np.ones_like(data[0], dtype=bool)
        if excluded is not None:
            all_mask[:, excluded] = False
        report, _ = evaluate_mask(head, data, all_mask, lam)
        results["cost"]["all:{}".format(lam)] = {"report": report}
        fm, fp, fpaths = traces["fixed"]
        am, ap, apaths = traces["adaptive"]
        results["paired"]["cost:{}".format(lam)] = family_bootstrap_delta(
            groups, y, fp, fm, ap, am, lam, seed + 2000 + int(lam * 1000))
        results["trajectories"]["cost:{}".format(lam)] = {
            "fixed": trajectory_audit(fpaths, z),
            "adaptive": trajectory_audit(apaths, z),
            "fraction_different_acquisition_order": float(np.mean(np.any(fpaths != apaths, axis=1))),
            "fraction_different_acquired_sets": float(np.mean(np.any(fm != am, axis=1))),
            "fraction_different_predictions": float(np.mean(fp != ap)),
            "adaptive_correct_fixed_wrong": int(np.sum((ap == y) & (fp != y))),
            "fixed_correct_adaptive_wrong": int(np.sum((fp == y) & (ap != y)))}
    return results


def run(root, out, seed, *, sanity=False, cpu_threads=2):
    torch.set_num_threads(cpu_threads)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("nonempty output directory already exists")
    train_rows, dev_rows, audit = load_development(root)
    roles = {role: [r for r in train_rows if internal_role(r["group_id"]) == role]
             for role in ("head_fit", "policy_fit", "tune")}
    if min(map(len, roles.values())) < 100:
        raise ValueError("internal role too small")
    if sanity:
        # Deterministic run-size restriction for engineering only, not paper evidence.
        roles = {role: selected[:min(len(selected), 384)] for role, selected in roles.items()}
    role_arrays = {role: arrays(selected) for role, selected in roles.items()}
    # A sanity run must not reveal official-dev metrics before protocol freeze.
    dev = role_arrays["tune"] if sanity else arrays(dev_rows)
    head, head_fit = fit_head(role_arrays["head_fit"], role_arrays["tune"], seed,
                              epochs=2 if sanity else 12)
    gain, gain_fit = fit_gain(head, role_arrays["policy_fit"], role_arrays["tune"], seed,
                              epochs=2 if sanity else 12)
    orders, search = greedy_orders(head, role_arrays["policy_fit"], starts=2 if sanity else 4)
    selections = {"tune_data": role_arrays["tune"]}
    for lam in LAMBDAS:
        for method in ("fixed", "adaptive"):
            selections[(method, lam)] = select_stopping(role_arrays["tune"], head, gain,
                                                        orders, method, lam)
    full = evaluate_policy_grid(dev, head, gain, orders, selections, seed)
    food = ASPECTS.index("Food#Recommend")
    excluded_orders, excluded_search = greedy_orders(
        head, role_arrays["policy_fit"], starts=2 if sanity else 4, excluded=food)
    excluded_selections = {"tune_data": role_arrays["tune"]}
    for lam in LAMBDAS:
        for method in ("fixed", "adaptive"):
            excluded_selections[(method, lam)] = select_stopping(role_arrays["tune"], head,
                gain, excluded_orders, method, lam, excluded=food)
    without_food = evaluate_policy_grid(dev, head, gain, excluded_orders, excluded_selections, seed,
                                        excluded=food)
    z = dev[0]
    mentioned = z != -2
    positives = z == 1
    negatives = z == -1
    aspect_audit = {
        "mean_mentioned_aspects": float(mentioned.sum(axis=1).mean()),
        "any_opposed_polarities_fraction": float(np.mean(positives.any(axis=1) &
                                                        negatives.any(axis=1))),
        "mention_rate": {name: float(mentioned[:, i].mean()) for i, name in enumerate(ASPECTS)},
    }
    report = {"format": "cbmjev-asap-gold-branch-v1", "seed": seed,
              "evidence_status": "ENGINEERING_SANITY_SUBSAMPLE" if sanity else
                                 "EXPLORATORY_OFFICIAL_DEV_NOT_TEST",
              "data": audit, "role_counts": {key: {"rows": len(value),
                            "families": len({r["group_id"] for r in value})}
                            for key, value in roles.items()},
              "evaluation_population": {"role": "internal_tune_sanity" if sanity else
                                               "official_dev_exploratory",
                                        "rows": len(dev[1]),
                                        "families": len(set(dev[2]))},
              "protocol": {"head": "partial_input_mlp_same_for_all_policies",
                           "adaptive": "DIME_Eq3_CE_gain_frozen_predictor_adaptation",
                           "adaptive_scope": "frozen_separate_predictor_not_original_joint_on_policy_DIME",
                           "fixed_search": search,
                           "food_exclusion_fixed_search": excluded_search,
                           "fixed_budget_order_selection": "internal_tune",
                           "stop_order_threshold_selection": "internal_tune",
                           "target_metric": "J_lambda=0_1_error+lambda*queries/18",
                           "cost_grid": list(LAMBDAS), "budgets": list(BUDGETS),
                           "threshold_factor_grid": list(THRESHOLD_FACTORS),
                           "test_labels_loaded": False,
                           "gold_concepts_are_simulated_oracle_responses": True,
                           "no_text_or_unqueried_aspect_to_policy": True,
                           "script_sha256": file_sha256(__file__),
                           "adapter_sha256": file_sha256(Path(__file__).resolve().parents[1] /
                                                          "cbmjev" / "asap_data.py")},
              "head_fit": head_fit, "gain_fit": gain_fit,
              "evaluation_results": full,
              "food_recommend_excluded_results": without_food,
              "aspect_audit": aspect_audit}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                          allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="official ASAP data/ directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, choices=(40, 41, 42), required=True)
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    report = run(args.data, args.out, args.seed, sanity=args.sanity,
                 cpu_threads=args.cpu_threads)
    print(json.dumps({"out": args.out, "seed": args.seed,
                      "status": report["evidence_status"],
                      "evaluated_rows": report["evaluation_population"]["rows"]}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Post-v2 exploratory ASAP budget-8 static-set stress test.

Motivation was observed on official dev after v2. This run is *not* a locked
confirmation, does not inspect test labels, and makes no method novelty claim.
The bounded search uses policy_fit only; tune selects among four candidate sets.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbmjev.asap_data import file_sha256, load_development
from evaluate_asap_gold_branch import (
    arrays, evaluate_mask, family_bootstrap_delta, fit_gain, fit_head,
    greedy_orders, internal_role, rollout, trajectory_audit,
)


BUDGET = 8
MAX_PASSES = 2
STARTS = 4
MAX_EVALUATIONS = 700


def set_mask(data, selected):
    mask = np.zeros_like(data[0], dtype=bool)
    mask[:, selected] = True
    return mask


def search_sets(head, policy_fit, starting_orders):
    """Four greedy starts; bounded best-improving 1-for-1 swaps on policy_fit."""
    cache = {}

    def risk(selected):
        selected = tuple(sorted(selected))
        if selected not in cache:
            if len(cache) >= MAX_EVALUATIONS:
                raise RuntimeError("predeclared static-search evaluation cap reached")
            report, _ = evaluate_mask(head, policy_fit, set_mask(policy_fit, selected), 0.)
            cache[selected] = report["error"]
        return cache[selected]

    candidates = []
    for order in starting_orders:
        current = tuple(sorted(order[:BUDGET]))
        visits = [current]
        for _ in range(MAX_PASSES):
            neighbors = []
            other = [i for i in range(18) if i not in current]
            for removed in current:
                for added in other:
                    trial = tuple(sorted((set(current) - {removed}) | {added}))
                    neighbors.append((risk(trial), trial))
            best_risk, best_set = min(neighbors)
            if best_risk >= risk(current) - 1e-12:
                break
            current = best_set
            visits.append(current)
        candidates.append({"selected": list(current), "policy_fit_error": risk(current),
                           "search_path": [list(item) for item in visits]})
    return candidates, len(cache)


def run(data, out, seed, cpu_threads=2):
    torch.set_num_threads(cpu_threads)
    output = Path(out)
    if output.exists() and any(output.iterdir()):
        raise ValueError("nonempty output already exists")
    train_rows, dev_rows, audit = load_development(data)
    roles = {role: arrays([r for r in train_rows if internal_role(r["group_id"]) == role])
             for role in ("head_fit", "policy_fit", "tune")}
    dev = arrays(dev_rows)
    head, head_fit = fit_head(roles["head_fit"], roles["tune"], seed)
    gain, gain_fit = fit_gain(head, roles["policy_fit"], roles["tune"], seed)
    orders, initial_search = greedy_orders(head, roles["policy_fit"], starts=STARTS)
    candidates, evaluated = search_sets(head, roles["policy_fit"], orders)
    for row in candidates:
        rr, _ = evaluate_mask(head, roles["tune"], set_mask(roles["tune"], row["selected"]), 0.)
        row["tune_error"] = rr["error"]
    chosen = min(candidates, key=lambda row: (row["tune_error"], row["policy_fit_error"],
                                              row["selected"]))
    fixed_mask = set_mask(dev, chosen["selected"])
    fixed_report, fixed_pred = evaluate_mask(head, dev, fixed_mask, 0.)
    adaptive_mask, paths = rollout(dev, head, gain, method="adaptive", budget=BUDGET,
                                   threshold=None, return_paths=True)
    adaptive_report, adaptive_pred = evaluate_mask(head, dev, adaptive_mask, 0.)
    paired = family_bootstrap_delta(dev[2], dev[1], fixed_pred, fixed_mask,
                                    adaptive_pred, adaptive_mask, 0., seed + 8000)
    result = {"format": "cbmjev-asap-strong-static-budget8-v1",
              "evidence_status": "POST_V2_EXPLORATORY_OFFICIAL_DEV_NOT_CONFIRMATION",
              "seed": seed, "data": audit,
              "protocol": {"head_and_gain": "same_function_configuration_as_asap_v2",
                           "initial_orders": initial_search,
                           "fixed_set_search": "four_starts_best_improving_one_for_one_swaps",
                           "budget": BUDGET, "max_swap_passes": MAX_PASSES,
                           "max_policy_fit_subset_evaluations": MAX_EVALUATIONS,
                           "selected_on": "internal_tune_only",
                           "search_on": "policy_fit_only",
                           "test_labels_used": False,
                           "script_sha256": file_sha256(__file__),
                           "v2_evaluator_sha256": file_sha256(Path(__file__).with_name(
                               "evaluate_asap_gold_branch.py"))},
              "role_counts": {role: {"rows": len(value[1]), "families": len(set(value[2]))}
                              for role, value in roles.items()},
              "head_fit": head_fit, "gain_fit": gain_fit,
              "search": {"actual_subset_evaluations": evaluated,
                         "candidates": candidates, "selected": chosen},
              "official_dev": {"rows": len(dev[1]), "families": len(set(dev[2])),
                               "strong_static_set": fixed_report,
                               "adaptive_DIME_Eq3_adapter": adaptive_report,
                               "adaptive_minus_static": paired,
                               "adaptive_unique_sequences": trajectory_audit(paths, dev[0])[
                                   "unique_acquisition_sequences"]}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                            allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, choices=(40, 41, 42), required=True)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    report = run(args.data, args.out, args.seed, args.cpu_threads)
    print(json.dumps({"out": args.out, "seed": args.seed,
                      "adaptive_minus_static_J": report["official_dev"][
                          "adaptive_minus_static"]["adaptive_minus_fixed_J"]}))


if __name__ == "__main__":
    main()

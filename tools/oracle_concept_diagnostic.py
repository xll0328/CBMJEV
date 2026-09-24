#!/usr/bin/env python3
"""Measure the gold-concept task upper bound on validation only.

This diagnostic deliberately uses public concept annotations as model inputs. It
is excluded from deployable CBMJev results and never evaluates the test split.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cbmjev.baselines import fit_static_order
from cbmjev.config import learning_config, resolve_config
from cbmjev.evaluation import summarize_traces
from cbmjev.io import environment, fresh_dir, write_json
from cbmjev.learning import fit_models, save_models
from cbmjev.pipeline import load_prepared
from cbmjev.runtime import ReplayEnvironment, run_episode
from cbmjev.contracts import DeclaredCost


def gold_rows(records, membership, schema):
    rows, exclusions = [], Counter()
    allowed = {"head_fit", "policy_fit", "validation"}
    for record in records:
        role = membership[record["sample_id"]]["split"]
        if role not in allowed:
            continue
        if record["target"]["status"] != "OBSERVED":
            exclusions[role + ":missing_target"] += 1
            continue
        concepts = record["concepts"]
        if len(concepts) != schema.num_atoms or any(
                c["annotation_status"] != "OBSERVED" for c in concepts):
            exclusions[role + ":incomplete_concepts"] += 1
            continue
        values = [c["value"] for c in concepts]
        schema.validate_state(values, complete=True)
        rows.append({"sample_id": record["sample_id"], "group_id": record["group_id"],
                     "split": role, "z": values, "y": record["target"]["value"],
                     "response_source": "gold_concept_diagnostic"})
    return rows, dict(sorted(exclusions.items()))


def run(prepared, config_path, out, *, seed, device):
    schema, records, membership = load_prepared(prepared)
    rows, exclusions = gold_rows(records, membership, schema)
    counts = Counter(row["split"] for row in rows)
    missing = {role for role in ("head_fit", "policy_fit", "validation") if not counts[role]}
    if missing:
        raise ValueError("gold diagnostic has empty roles: " + ", ".join(sorted(missing)))
    config = resolve_config(json.loads(Path(config_path).read_text(encoding="utf-8")))
    config["seed"], config["device"] = seed, device
    config = resolve_config(config)
    out = fresh_dir(out)
    head, controller, training = fit_models(rows, schema, learning_config(config))
    save_models(out / "models.pt", head, controller, training)
    order, static = fit_static_order(rows, head, schema)
    traces = []
    for method in ("stop", "all", "fixed", "static", config["learning"]["objective"]):
        for row in (r for r in rows if r["split"] == "validation"):
            trace = run_episode(ReplayEnvironment(row["z"], schema), schema, head,
                method=method, controller=controller, order=order,
                cost=DeclaredCost(**config["cost"]), **config["policy"],
                include_pairs=config["learning"]["include_pairs"],
                include_all=config["learning"]["include_all"], pairs=controller.pairs,
                seed=seed, device=device)
            trace.update(sample_id=row["sample_id"], group_id=row["group_id"],
                         split="validation", y=row["y"], policy_id=method,
                         seed=seed, num_query_groups=schema.num_groups, num_atoms=schema.num_atoms)
            traces.append(trace)
    policies = {method: summarize_traces([t for t in traces if t["policy_id"] == method],
                                         num_classes=schema.num_classes)
                for method in ("stop", "all", "fixed", "static", config["learning"]["objective"])}
    report = {
        "format": "cbmjev-gold-concept-diagnostic-v1",
        "dataset": schema.dataset,
        "split": "validation",
        "test_evaluated": False,
        "seed": seed,
        "complete_case_counts": dict(counts),
        "exclusions": exclusions,
        "policies": policies,
        "selection_warning": "complete concept annotations only; population differs from automatic-response runs",
        "evidence_status": "VALIDATION_ORACLE_DIAGNOSTIC_NOT_DEPLOYABLE_OR_PAPER_RESULT",
        "paper_claim": False,
    }
    write_json(out / "metrics.json", report)
    write_json(out / "training.json", training)
    write_json(out / "static_order.json", {"order": list(order), "report": static})
    write_json(out / "config.json", config)
    write_json(out / "environment.json", environment())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run(args.prepared, args.config, args.out, seed=args.seed, device=args.device)
    print(json.dumps({"out": str(Path(args.out).resolve()),
                      "validation_complete": report["complete_case_counts"]["validation"],
                      "test_evaluated": False}))


if __name__ == "__main__":
    main()

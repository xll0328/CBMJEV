#!/usr/bin/env python3
"""CEBaB concept-only DIME *adaptation*, not a reproduction of Gadgil et al.

The original DIME jointly optimizes a partial-input predictor and a value
network, using on-policy/exploratory states. Here the semantic responder and
masked task head are frozen. A separate state-to-four-values MLP is fitted on
policy_fit responses and labels to predict the signed cross-entropy reduction
from each remaining singleton concept group. Complete responses and labels
are offline targets only: inference consumes solely the revealed state.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from cbmjev.contracts import DeclaredCost, stable_hash
from cbmjev.evaluation import summarize_traces
from cbmjev.io import file_hash, fresh_dir, write_json, write_jsonl
from cbmjev.learning import encode_states
from cbmjev.pipeline import load_model_bundle
from cbmjev.runtime import ReplayEnvironment


LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.40)
PENALTIES = (0.0, 0.01, 0.03, 0.10, 0.20)


def state_for_mask(schema, response, mask):
    state = list(schema.empty_state())
    for group in range(schema.num_groups):
        if mask & (1 << group):
            for atom in schema.groups[group].atoms:
                state[atom] = response[atom]
    return tuple(state)


def make_training_examples(rows, schema, head):
    """All legal singleton deltas; only policy_fit rows are accepted."""
    if not rows or any(row["split"] != "policy_fit" for row in rows):
        raise ValueError("DIME adaptation fits only nonempty policy_fit rows")
    states, targets, legal = [], [], []
    for row in rows:
        response = tuple(row["z"])
        schema.validate_state(response, complete=True)
        y = row["y"]
        if type(y) is not int or not 0 <= y < schema.num_classes:
            raise ValueError("invalid task label")
        masks = range(1 << schema.num_groups)
        row_states = [state_for_mask(schema, response, mask) for mask in masks]
        probabilities = head.probabilities_many(row_states)
        if len(probabilities) != len(row_states):
            raise ValueError("frozen head returned wrong batch length")
        losses = []
        for probs in probabilities:
            if len(probs) != schema.num_classes or any(
                    not math.isfinite(p) or not 0 <= p <= 1 for p in probs):
                raise ValueError("invalid task-head probabilities")
            losses.append(-math.log(max(probs[y], 1e-300)))
        for mask in masks:
            state_targets, state_legal = [], []
            for group in range(schema.num_groups):
                available = not bool(mask & (1 << group))
                state_legal.append(available)
                state_targets.append(losses[mask] - losses[mask | (1 << group)]
                                     if available else 0.0)
            states.append(row_states[mask])
            targets.append(state_targets)
            legal.append(state_legal)
    return (encode_states(states, schema, "cpu"),
            torch.tensor(targets, dtype=torch.float32),
            torch.tensor(legal, dtype=torch.bool))


class DimeAdapter:
    def __init__(self, schema, hidden=128):
        self.schema = schema
        self.network = nn.Sequential(nn.Linear(sum(schema.num_categories) + schema.num_atoms,
                                                hidden), nn.ReLU(), nn.Linear(hidden, schema.num_groups))

    def predict(self, state):
        self.network.eval()
        with torch.no_grad():
            values = self.network(encode_states((state,), self.schema, "cpu"))[0]
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite DIME-adapter values")
        return tuple(float(value) for value in values)


def fit_value_net(features, targets, legal, *, seed, hidden=128, epochs=20,
                  batch_size=256, learning_rate=0.001):
    if features.ndim != 2 or targets.shape != legal.shape or targets.ndim != 2:
        raise ValueError("invalid training tensor shapes")
    if features.shape[0] != targets.shape[0] or not bool(legal.any()):
        raise ValueError("empty or misaligned legal training examples")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("invalid training hyperparameters")
    torch.manual_seed(seed + 550021)
    network = nn.Sequential(nn.Linear(features.shape[1], hidden), nn.ReLU(),
                            nn.Linear(hidden, targets.shape[1]))
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed + 550022)
    losses = []
    for _ in range(epochs):
        network.train()
        total, count = 0.0, 0
        for indices in torch.randperm(len(features), generator=generator).split(batch_size):
            output = network(features[indices])
            mask = legal[indices]
            loss = F.mse_loss(output[mask], targets[indices][mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * int(mask.sum())
            count += int(mask.sum())
        losses.append(total / count)
    network.eval()
    return network, losses


def select_action(state, schema, value_net, cost, penalty):
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and nonnegative")
    seen = schema.group_mask(state)
    available = [group for group in range(schema.num_groups) if not seen[group]]
    if not available:
        return (), {}, "complete"
    values = value_net.predict(state)
    if len(values) != schema.num_groups:
        raise ValueError("DIME-adapter value width mismatch")
    if any(not math.isfinite(values[group]) for group in available):
        raise ValueError("nonfinite available value")
    best = min(available, key=lambda group: (-(values[group] -
                         penalty * cost(state, (group,))), group))
    detail = {str(group): values[group] for group in available}
    if values[best] - penalty * cost(state, (best,)) <= 0:
        return (), detail, "nonpositive_net_value"
    return (best,), detail, "acquire"


def replay(row, schema, head, value_net, cost, penalty):
    env = ReplayEnvironment(row["z"], schema)
    steps, spent = [], 0.0
    for _ in range(schema.num_groups + 1):
        state = env.state()
        action, detail, reason = select_action(state, schema, value_net, cost, penalty)
        step = {"before": list(state), "action": list(action), "scores": detail,
                "reason": reason, "declared_cost": cost(state, action),
                "legal_candidate_count": 1 + schema.num_groups - sum(schema.group_mask(state))}
        if not action:
            step["decision"] = "STOP"
            steps.append(step)
            break
        values = env.query(action)
        spent += step["declared_cost"]
        step.update(decision="ACQUIRE", atom_ids=list(schema.expand(action)), values=list(values))
        steps.append(step)
    else:
        raise RuntimeError("DIME-adapter episode failed to stop")
    probabilities = tuple(float(p) for p in head.probabilities(env.state()))
    prediction = max(range(schema.num_classes), key=lambda i: probabilities[i])
    return {"sample_id": row["sample_id"], "group_id": row["group_id"],
            "split": row["split"], "y": row["y"], "method": "dime_cebab_adapter",
            "prediction": prediction, "probabilities": list(probabilities),
            "queried_groups": [g for g, present in enumerate(schema.group_mask(env.state())) if present],
            "queried_atoms": [a for a, v in enumerate(env.state()) if v != -1],
            "calls": env.calls, "declared_cost": spent, "cost_units": cost.units,
            "mode": "offline_replay", "steps": steps, "final_state": list(env.state())}


def external_j(report, lam, num_groups):
    return report["error"] + lam * report["mean_queried_groups"] / num_groups


def choose_penalty(fit_reports, lam, num_groups):
    if set(fit_reports) != set(PENALTIES):
        raise ValueError("incomplete policy_fit penalty grid")
    return min(PENALTIES, key=lambda p: (external_j(fit_reports[p], lam, num_groups),
                                        fit_reports[p]["mean_queried_groups"], p))


def run(models, cache, out, *, limit=None, cpu_threads=2, epochs=20,
        hidden=128, batch_size=256, learning_rate=0.001):
    if limit is not None and limit < 1 or cpu_threads < 1 or hidden < 1:
        raise ValueError("invalid limit, cpu_threads, or hidden")
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("output must be new or empty")
    torch.set_num_threads(cpu_threads)
    schema, rows, manifest, config, receipt, head, _, _ = load_model_bundle(
        models, cache, device="cpu")
    if schema.dataset != "cebab" or schema.num_groups != 4:
        raise ValueError("DIME adaptation requires four-group CEBaB")
    if config["policy"]["max_cost"] is not None:
        raise ValueError("hard max_cost is not implemented")
    fit = [row for row in rows if row["split"] == "policy_fit"]
    val = [row for row in rows if row["split"] == "validation"]
    if not fit or not val:
        raise ValueError("policy_fit and validation are required")
    original_counts = {"policy_fit": len(fit), "validation": len(val)}
    if limit is not None:
        fit, val = fit[:limit], val[:limit]
    features, targets, legal = make_training_examples(fit, schema, head)
    network, losses = fit_value_net(features, targets, legal, seed=config["seed"],
        hidden=hidden, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    value_net = DimeAdapter(schema, hidden=hidden)
    value_net.network = network
    cost = DeclaredCost(**config["cost"])
    fit_reports = {}
    for penalty in PENALTIES:
        traces = [replay(row, schema, head, value_net, cost, penalty) for row in fit]
        fit_reports[penalty] = summarize_traces(traces, num_classes=schema.num_classes)
    selected, traces_out = {}, []
    for lam in LAMBDAS:
        penalty = choose_penalty(fit_reports, lam, schema.num_groups)
        traces = [replay(row, schema, head, value_net, cost, penalty) for row in val]
        summary = summarize_traces(traces, num_classes=schema.num_classes)
        selected[str(lam)] = {"penalty": penalty,
                              "policy_fit_j": external_j(fit_reports[penalty], lam, schema.num_groups),
                              "validation_j": external_j(summary, lam, schema.num_groups),
                              "validation": summary}
        traces_out.extend({**trace, "external_lambda": lam,
                           "selected_penalty": penalty} for trace in traces)
    report = {"format": "cbmjev-cebab-dime-adapter-v1", "seed": config["seed"],
        "evidence_status": "PILOT_SUBSAMPLE_NOT_CLAIM_EVIDENCE" if limit is not None else
                           "EXPLORATORY_VALIDATION_NOT_TEST",
        "test_evaluated": False,
        "roles": {"policy_fit_samples": len(fit), "validation_samples": len(val),
                  "original_role_counts": original_counts, "limit": limit},
        "protocol": {"name": "DIME-style signed CE-delta value regression adaptation",
            "original_primary_paper": "https://arxiv.org/abs/2306.03301",
            "original_author_code": "https://github.com/suinleelab/DIME",
            "not_reproduction": True,
            "differences": ["Frozen concept-only responder and masked task head, not joint predictor/value training",
                            "Exhaustive policy_fit subset/singleton targets, not on-policy epsilon exploration",
                            "Four concept groups and fixed semantic head rather than original feature-level tasks"],
            "theory_caveat": "The original CMI equivalence assumes a Bayes-optimal predictor; it is not asserted for this frozen learned head.",
            "value_target": "CE(head(x_S), y) - CE(head(x_S+i), y), signed and possibly negative",
            "inference_observations": "partial hard concept state only; no y, hidden responses, or future head output",
            "training_role": "policy_fit", "penalty_selection_role": "policy_fit (same role as value training; in-sample selection)",
            "evaluation_role": "validation", "external_j": "error_rate + lambda * mean_queried_groups / 4",
            "lambdas": LAMBDAS, "penalties": PENALTIES, "cost": cost.to_dict(),
            "training": {"hidden": hidden, "epochs": epochs, "batch_size": batch_size,
                         "learning_rate": learning_rate, "cpu_threads": cpu_threads,
                         "seed_offset_initialization": 550021,
                         "seed_offset_shuffle": 550022,
                         "masked_states": len(features), "legal_singleton_targets": int(legal.sum())}},
        "train_epoch_mse": losses,
        "policy_fit_grid": [{"penalty": p, "report": fit_reports[p]} for p in PENALTIES],
        "selected_validation": selected,
        "bindings": {"models_receipt_sha256": file_hash(Path(models) / "receipt.json"),
                     "model_weights_sha256": receipt["models_sha256"],
                     "head_component_sha256": receipt["head_component_sha256"],
                     "cache_manifest_sha256": file_hash(Path(cache) / "manifest.json"),
                     "cache_responses_sha256": manifest["responses_sha256"],
                     "script_sha256": file_hash(__file__)}}
    out = fresh_dir(out)
    torch.save({"network_state_dict": network.state_dict(),
                "schema_hash": schema.hash, "hidden": hidden,
                "train_role": "policy_fit"}, out / "dime_adapter.pt")
    report["bindings"]["adapter_weights_sha256"] = file_hash(out / "dime_adapter.pt")
    report["report_hash"] = stable_hash(report)
    write_jsonl(out / "validation_traces.jsonl", traces_out)
    write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "cache", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--limit", type=int, help="pilot only: first N per role")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    report = run(args.models, args.cache, args.out, limit=args.limit,
                 cpu_threads=args.cpu_threads, epochs=args.epochs,
                 hidden=args.hidden, batch_size=args.batch_size,
                 learning_rate=args.learning_rate)
    print(json.dumps({"out": args.out, "seed": report["seed"],
        "validation_samples": report["roles"]["validation_samples"],
        "selected_validation": {lam: {"penalty": row["penalty"],
                                  "validation_j": row["validation_j"]}
                                for lam, row in report["selected_validation"].items()}},
        sort_keys=True))


if __name__ == "__main__":
    main()

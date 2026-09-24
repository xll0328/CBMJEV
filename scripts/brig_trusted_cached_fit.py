"""BRiG fitter caching exact rollout risks and trusting internal states.

The fast paths are private to this fitter. Public inference still uses the
original validating ``GroupQ.forward``; training inputs are checked once by
``training_rows`` and all subsequent partial states are built from those
validated answers and legal group masks/actions.
"""

import hashlib
import json
import math
import random

import torch
from torch.nn import functional as F

from cbmjev.brig import UPSTREAM_COMMIT, BRiGPolicy, _terminal
from cbmjev.learning import mask_answers, training_rows
from scripts.brig_trusted_ops import (
    TrustedTrainingGroupQ, _reveal_trusted, bellman_targets_trusted,
    precompute_empty_rollout_terminal_states_trusted)


def fit_brig_trusted_cached(rows, head, schema, config=None, *,
                            excluded_head_group_ids):
    """Fit with exact cached risks and validation-free private training ops."""
    defaults = dict(seed=17, device="cpu", hidden=128, max_budget=schema.num_groups,
                    epochs=8, batch_size=64, learning_rate=.001,
                    empty_rollout=True, empty_weight=1.0)
    if set(config or {}) - set(defaults):
        raise ValueError("unknown BRiG configuration key")
    cfg = {**defaults, **(config or {})}
    for key in ("hidden", "max_budget", "epochs", "batch_size"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError("positive integer required: " + key)
    if type(cfg["seed"]) is not int or type(cfg["empty_rollout"]) is not bool:
        raise ValueError("invalid seed or rollout flag")
    for key in ("learning_rate", "empty_weight"):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError("invalid " + key)
    if not cfg["learning_rate"] or cfg["max_budget"] > schema.num_groups:
        raise ValueError("invalid learning rate or maximum budget")

    rows = list(rows)
    selected = [row for row in rows if row.get("split") == "policy_fit"]
    examples = training_rows(rows, "policy_fit", schema)
    if not {row["group_id"] for row in selected} <= set(excluded_head_group_ids):
        raise ValueError("task head must exclude every policy-fit group")
    if hasattr(head, "schema") and head.schema.hash != schema.hash:
        raise ValueError("head/schema mismatch")

    epochs = max(cfg["epochs"], 8)
    rng = random.Random(cfg["seed"])
    device = torch.device(cfg["device"])
    models, history = {}, []
    event_digest = hashlib.sha256()
    cache_state_digest = hashlib.sha256()
    for budget in range(1, cfg["max_budget"] + 1):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cfg["seed"] + budget)
            model = TrustedTrainingGroupQ(schema, cfg["hidden"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

        cached_terminal_states = None
        if budget > 1 and cfg["empty_rollout"]:
            cached_terminal_states = precompute_empty_rollout_terminal_states_trusted(
                schema, models, examples, budget, chunk_size=cfg["batch_size"])
            cache_state_digest.update(json.dumps(
                [budget, len(cached_terminal_states), schema.num_groups],
                separators=(",", ":")).encode())
            for terminal_states in cached_terminal_states:
                cache_state_digest.update(json.dumps(
                    terminal_states, separators=(",", ":")).encode())

        for epoch in range(epochs):
            order = list(range(len(examples)))
            rng.shuffle(order)
            mse_sum, aux_sum, count, aux_count, updates = 0., 0., 0, 0, 0
            for start in range(0, len(order), cfg["batch_size"]):
                batch_indices = order[start:start + cfg["batch_size"]]
                states, actions, next_states, labels = [], [], [], []
                for index in batch_indices:
                    answers, label = examples[index]
                    kind = rng.randrange(3)
                    n = rng.randrange(schema.num_groups - budget + 1)
                    visible = set(rng.sample(range(schema.num_groups), n)) if kind else set()
                    state = mask_answers(
                        answers, tuple(g in visible for g in range(schema.num_groups)), schema)
                    for action in range(schema.num_groups):
                        if action in visible:
                            continue
                        states.append(state)
                        actions.append(action)
                        next_states.append(_reveal_trusted(schema, state, action, answers))
                        labels.append(label)

                targets = bellman_targets_trusted(
                    head, schema, models, next_states, labels, budget, device)
                pred = model.forward_trusted(states, actions, budget)
                loss = F.mse_loss(pred, targets)
                mse_sum += float(loss.detach()) * len(actions)
                count += len(actions)
                event_digest.update(json.dumps(
                    [budget, epoch, states, actions, targets.cpu().tolist()],
                    separators=(",", ":")).encode())

                if cached_terminal_states is not None:
                    batch_states = [state for index in batch_indices
                                    for state in cached_terminal_states[index]]
                    batch_labels = [examples[index][1] for index in batch_indices
                                    for _ in range(schema.num_groups)]
                    batch_targets = _terminal(
                        head, batch_states, batch_labels, device)
                    empty_states = [schema.empty_state()] * batch_targets.numel()
                    empty_actions = [action for _ in batch_indices
                                     for action in range(schema.num_groups)]
                    aux = F.mse_loss(
                        model.forward_trusted(empty_states, empty_actions, budget),
                        batch_targets)
                    loss = loss + cfg["empty_weight"] * aux
                    aux_sum += float(aux.detach()) * len(empty_actions)
                    aux_count += len(empty_actions)
                    event_digest.update(json.dumps(
                        ["rollout-terminal-state-cache-v1", budget, epoch, batch_indices,
                         batch_targets.cpu().tolist()],
                        separators=(",", ":")).encode())

                if not torch.isfinite(loss):
                    raise ValueError("nonfinite BRiG loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                updates += 1
            history.append(dict(budget=budget, epoch=epoch, bellman_targets=count,
                                rollout_targets=aux_count, optimizer_steps=updates,
                                mse=mse_sum / count,
                                rollout_mse=aux_sum / aux_count if aux_count else None))
        model.eval().requires_grad_(False)
        models[budget] = model

    report = dict(method="grouped_categorical_budget_specific_brig_v1",
                  upstream_commit=UPSTREAM_COMMIT, config=cfg,
                  epochs_effective=epochs, split="policy_fit", sample_count=len(examples),
                  group_count=len({r["group_id"] for r in selected}), schema_hash=schema.hash,
                  head_exclusion_coverage_checked=True, head_exclusion_provenance_checked=False,
                  risk="unweighted_terminal_ce_probability_floor_1e-12",
                  sampling="generic_no_optional_rollin",
                  internal_state_validation="validated-inputs-plus-legal-construction",
                  cached_rollout_examples=len(examples) if cfg["empty_rollout"] else 0,
                  cached_rollout_terminal_state_sha256=cache_state_digest.hexdigest(),
                  ordered_training_rows_sha256=hashlib.sha256(json.dumps(
                      [(r["sample_id"], r["group_id"], r["z"], r["y"]) for r in selected],
                      separators=(",", ":")).encode()).hexdigest(),
                  event_digest_protocol="terminal-state-cache-minibatch-risk-v1",
                  event_sha256=event_digest.hexdigest(), history=history,
                  parameter_count=sum(p.numel() for m in models.values() for p in m.parameters()))
    return BRiGPolicy(schema, models), report

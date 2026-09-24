"""BRiG fitter with an exact per-budget cache for empty-rollout risks.

The optimization changes only when deterministic auxiliary targets are
computed. The same examples, batches, targets, losses, optimizer updates and
random-number stream are retained. A cache-specific event digest replaces the
old terminal-state digest because terminal states need not be retained after
their exact scalar risks have been computed.
"""

import hashlib
import json
import math
import random

import torch
from torch.nn import functional as F

from cbmjev.brig import (UPSTREAM_COMMIT, BRiGPolicy, GroupQ, _reveal,
                         bellman_targets)
from cbmjev.learning import mask_answers, training_rows
from scripts.brig_cached_rollout_targets import precompute_empty_rollout_risks


def fit_brig_cached(rows, head, schema, config=None, *, excluded_head_group_ids):
    """Fit BRiG while precomputing each frozen-budget rollout target once."""
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
    cache_target_digest = hashlib.sha256()
    for budget in range(1, cfg["max_budget"] + 1):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cfg["seed"] + budget)
            model = GroupQ(schema, cfg["hidden"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

        # Q_<budget and the task head are frozen for this budget's complete
        # fit, so the terminal risk for (example, first action) is invariant.
        cached_risks = None
        if budget > 1 and cfg["empty_rollout"]:
            cached_risks = precompute_empty_rollout_risks(
                schema, models, head, examples, budget, device=device,
                chunk_size=cfg["batch_size"])
            cache_target_digest.update(json.dumps(
                [budget, list(cached_risks.shape), str(cached_risks.dtype)],
                separators=(",", ":")).encode())
            cache_target_digest.update(
                cached_risks.detach().cpu().contiguous().numpy().tobytes())

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
                    state = mask_answers(answers, tuple(g in visible for g in range(schema.num_groups)), schema)
                    for action in range(schema.num_groups):
                        if action in visible:
                            continue
                        states.append(state)
                        actions.append(action)
                        next_states.append(_reveal(schema, state, action, answers))
                        labels.append(label)

                targets = bellman_targets(head, schema, models, next_states, labels, budget, device)
                pred = model(states, actions, budget)
                loss = F.mse_loss(pred, targets)
                mse_sum += float(loss.detach()) * len(actions)
                count += len(actions)
                event_digest.update(json.dumps(
                    [budget, epoch, states, actions, targets.cpu().tolist()],
                    separators=(",", ":")).encode())

                if cached_risks is not None:
                    batch_targets = cached_risks[batch_indices].reshape(-1)
                    empty_states = [schema.empty_state()] * batch_targets.numel()
                    empty_actions = [action for _ in batch_indices
                                     for action in range(schema.num_groups)]
                    aux = F.mse_loss(model(empty_states, empty_actions, budget), batch_targets)
                    loss = loss + cfg["empty_weight"] * aux
                    aux_sum += float(aux.detach()) * len(empty_actions)
                    aux_count += len(empty_actions)
                    event_digest.update(json.dumps(
                        ["rollout-cache-v1", budget, epoch, batch_indices,
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
                  cached_rollout_examples=len(examples) if cfg["empty_rollout"] else 0,
                  cached_rollout_target_sha256=cache_target_digest.hexdigest(),
                  ordered_training_rows_sha256=hashlib.sha256(json.dumps(
                      [(r["sample_id"], r["group_id"], r["z"], r["y"]) for r in selected],
                      separators=(",", ":")).encode()).hexdigest(),
                  event_digest_protocol="cache-indexed-objective-v1",
                  event_sha256=event_digest.hexdigest(), history=history,
                  parameter_count=sum(p.numel() for m in models.values() for p in m.parameters()))
    return BRiGPolicy(schema, models), report

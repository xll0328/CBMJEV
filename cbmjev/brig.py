"""Grouped categorical adaptation of budget-specific Learned-Q BRiG-AFA.

Not an official reproduction. MIT attribution is in docs/BRIG_ADAPTATION.md.
Offline complete responses produce Bellman targets;
the deployable policy receives only observed states and a remaining group budget.
"""
import hashlib
import json
import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from .learning import encode_states, mask_answers, training_rows, validate_observed

UPSTREAM_COMMIT = "c4001404c5b5affeed6340e5d45e3229224148eb"


class GroupQ(nn.Module):
    def __init__(self, schema, hidden=128):
        super().__init__()
        self.schema = schema
        width = sum(schema.num_categories) + schema.num_atoms + schema.num_groups + 1
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, observed_states, candidates, remaining_budget):
        if len(observed_states) != len(candidates) or not candidates:
            raise ValueError("nonempty aligned states/candidates required")
        if type(remaining_budget) is not int or not 1 <= remaining_budget <= self.schema.num_groups:
            raise ValueError("invalid remaining budget")
        for state, candidate in zip(observed_states, candidates):
            state = validate_observed(state, self.schema)
            if type(candidate) is not int or not 0 <= candidate < self.schema.num_groups:
                raise ValueError("invalid candidate")
            mask = self.schema.group_mask(state)
            if mask[candidate] or remaining_budget > sum(not x for x in mask):
                raise ValueError("reacquisition or infeasible budget")
        device = next(self.parameters()).device
        state = encode_states(observed_states, self.schema, device)
        action = F.one_hot(torch.tensor(candidates, device=device), self.schema.num_groups).float()
        budget = state.new_full((len(candidates), 1), remaining_budget / self.schema.num_groups)
        return self.net(torch.cat((state, action, budget), dim=1)).squeeze(-1)


class BRiGPolicy:
    def __init__(self, schema, models):
        self.schema, self.models = schema, dict(models)
        if sorted(models) != list(range(1, len(models) + 1)) or not models:
            raise ValueError("consecutive trained budgets 1..B required")

    def predict_budget(self, observed, remaining_budget):
        """Return ordered (candidate, risk-to-go) pairs, never hidden values."""
        observed = validate_observed(observed, self.schema)
        if type(remaining_budget) is not int or remaining_budget < 0:
            raise ValueError("invalid budget")
        if remaining_budget == 0:
            return ()
        if remaining_budget not in self.models:
            raise ValueError("budget not trained")
        candidates = [g for g, present in enumerate(self.schema.group_mask(observed)) if not present]
        if remaining_budget > len(candidates):
            raise ValueError("budget exceeds available groups")
        model = self.models[remaining_budget]
        model.eval()
        with torch.no_grad():
            values = model([observed] * len(candidates), candidates, remaining_budget)
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite Q prediction")
        return tuple(zip(candidates, values.cpu().tolist()))

    def choose(self, observed, remaining_budget):
        values = self.predict_budget(observed, remaining_budget)
        return () if not values else (min(values, key=lambda pair: (pair[1], pair[0]))[0],)


def _reveal(schema, state, candidate, answers):
    atoms = schema.expand((candidate,))
    return schema.reveal(state, (candidate,), tuple(answers[a] for a in atoms))


def _terminal(head, states, labels, device):
    probabilities = head.probabilities_many(states)
    if len(probabilities) != len(states):
        raise ValueError("head returned incorrect batch length")
    risks = []
    for p, y in zip(probabilities, labels):
        if (not p or any(not math.isfinite(v) or v < 0 or v > 1 for v in p)
                or abs(sum(p) - 1) > 1e-4):
            raise ValueError("invalid terminal probabilities")
        risks.append(-math.log(max(p[y], 1e-12)))
    return torch.tensor(risks, dtype=torch.float32, device=device)


@torch.no_grad()
def bellman_targets(head, schema, models, next_states, labels, remaining_budget, device="cpu"):
    """Teacher after first acquisition. Labels affect only r=1 terminal risk."""
    if not next_states or len(next_states) != len(labels) or remaining_budget < 1:
        raise ValueError("nonempty aligned Bellman examples required")
    if remaining_budget == 1:
        return _terminal(head, next_states, labels, device)
    previous = models[remaining_budget - 1]
    previous.eval()
    states, actions, lengths = [], [], []
    for state in next_states:
        candidates = [g for g, seen in enumerate(schema.group_mask(state)) if not seen]
        lengths.append(len(candidates))
        states.extend([state] * len(candidates))
        actions.extend(candidates)
    values = previous(states, actions, remaining_budget - 1)
    return torch.stack([part.min() for part in values.split(lengths)]).to(device)


def fit_brig(rows, head, schema, config=None, *, excluded_head_group_ids):
    """Fit Q_1..Q_B, returning (policy, ordered machine-readable report).

    Caller MUST supply the validated head exclusion groups; this function verifies
    coverage, not the provenance of that assertion. Use automatic response rows,
    not gold concepts. This low-level API cannot authenticate response provenance.
    """
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
    epochs = max(cfg["epochs"], 8)  # Upstream's effective minimum, explicitly reported.
    rng = random.Random(cfg["seed"])
    device = torch.device(cfg["device"])
    models, history = {}, []
    event_digest = hashlib.sha256()
    for budget in range(1, cfg["max_budget"] + 1):
        # Local RNG context avoids changing unrelated model initialization.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cfg["seed"] + budget)
            model = GroupQ(schema, cfg["hidden"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        for epoch in range(epochs):
            order = list(range(len(examples)))
            rng.shuffle(order)
            mse_sum, aux_sum, count, aux_count, updates = 0., 0., 0, 0, 0
            for start in range(0, len(order), cfg["batch_size"]):
                states, actions, next_states, labels, answers_batch = [], [], [], [], []
                for index in order[start:start + cfg["batch_size"]]:
                    answers, label = examples[index]
                    # Upstream generic sampler without optional static/learned roll-in:
                    # one empty component, two exchangeable uniform-prefix components.
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
                    answers_batch.append((answers, label))
                targets = bellman_targets(head, schema, models, next_states, labels, budget, device)
                pred = model(states, actions, budget)
                loss = F.mse_loss(pred, targets)
                mse_sum += float(loss.detach()) * len(actions)
                count += len(actions)
                event_digest.update(json.dumps([budget, epoch, states, actions, targets.cpu().tolist()], separators=(",", ":")).encode())
                if budget > 1 and cfg["empty_rollout"]:
                    empty_states, empty_actions, terminal_states, terminal_labels = [], [], [], []
                    policy = BRiGPolicy(schema, models)
                    for answers, label in answers_batch:
                        for action in range(schema.num_groups):
                            state = _reveal(schema, schema.empty_state(), action, answers)
                            for left in range(budget - 1, 0, -1):
                                chosen = policy.choose(state, left)[0]
                                state = _reveal(schema, state, chosen, answers)
                            empty_states.append(schema.empty_state())
                            empty_actions.append(action)
                            terminal_states.append(state)
                            terminal_labels.append(label)
                    target = _terminal(head, terminal_states, terminal_labels, device)
                    aux = F.mse_loss(model(empty_states, empty_actions, budget), target)
                    loss = loss + cfg["empty_weight"] * aux
                    aux_sum += float(aux.detach()) * len(empty_actions)
                    aux_count += len(empty_actions)
                    event_digest.update(json.dumps(["rollout", budget, epoch, terminal_states, empty_actions, target.cpu().tolist()], separators=(",", ":")).encode())
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite BRiG loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                updates += 1
            history.append(dict(budget=budget, epoch=epoch, bellman_targets=count,
                                rollout_targets=aux_count, optimizer_steps=updates,
                                mse=mse_sum / count, rollout_mse=aux_sum / aux_count if aux_count else None))
        model.eval().requires_grad_(False)
        models[budget] = model
    report = dict(method="grouped_categorical_budget_specific_brig_v1", upstream_commit=UPSTREAM_COMMIT,
                  config=cfg, epochs_effective=epochs, split="policy_fit", sample_count=len(examples),
                  group_count=len({r["group_id"] for r in selected}), schema_hash=schema.hash,
                  head_exclusion_coverage_checked=True, head_exclusion_provenance_checked=False,
                  risk="unweighted_terminal_ce_probability_floor_1e-12", sampling="generic_no_optional_rollin",
                  ordered_training_rows_sha256=hashlib.sha256(json.dumps(
                      [(r["sample_id"], r["group_id"], r["z"], r["y"]) for r in selected],
                      separators=(",", ":")).encode()).hexdigest(),
                  event_sha256=event_digest.hexdigest(), history=history,
                  parameter_count=sum(p.numel() for m in models.values() for p in m.parameters()))
    return BRiGPolicy(schema, models), report

"""Validated, deduplicated GroupQ scoring for the BRiG training path.

The public GroupQ.forward remains the reference implementation. This helper
preserves its validation contract while encoding each distinct partial state
once per call, rather than once for every candidate-action pair.
"""

import torch
from torch.nn import functional as F

from cbmjev.learning import validate_observed


def _unique_state_features(states, schema, device):
    unique = []
    lookup = {}
    inverse = []
    for state in states:
        if not isinstance(state, (list, tuple)):
            raise ValueError("observed must be a fixed-width partial hard-state sequence")
        key = tuple(state)
        index = lookup.get(key)
        if index is None:
            validate_observed(key, schema)
            index = len(unique)
            lookup[key] = index
            unique.append(key)
        inverse.append(index)

    if not unique:
        raise ValueError("nonempty aligned states/candidates required")

    values = torch.tensor(unique, dtype=torch.long, device=device)
    observed = values >= 0
    safe = torch.where(observed, values, torch.zeros_like(values))
    offsets = []
    offset = 0
    for category_count in schema.num_categories:
        offsets.append(offset)
        offset += category_count
    offsets = torch.tensor(offsets, dtype=torch.long, device=device)
    categories = torch.zeros((len(unique), offset), dtype=torch.float32, device=device)
    categories.scatter_(1, safe + offsets, observed.to(torch.float32))
    encoded = torch.cat((categories, observed.to(torch.float32)), dim=1)
    return encoded, torch.tensor(inverse, dtype=torch.long, device=device), unique


def accelerated_groupq_forward(self, observed_states, candidates, remaining_budget):
    """Numerically equivalent GroupQ scoring with unique-state encoding."""
    if len(observed_states) != len(candidates) or not candidates:
        raise ValueError("nonempty aligned states/candidates required")
    schema = self.schema
    if type(remaining_budget) is not int or not 1 <= remaining_budget <= schema.num_groups:
        raise ValueError("invalid remaining budget")

    device = next(self.parameters()).device
    encoded_unique, inverse, unique = _unique_state_features(observed_states, schema, device)

    # Group completeness has already been checked by validate_observed. A
    # group's first atom therefore determines its visibility, as in Schema.
    visible = [tuple(state[group.atoms[0]] >= 0 for group in schema.groups)
               for state in unique]
    unique_index = {state: i for i, state in enumerate(unique)}
    for state, candidate in zip(observed_states, candidates):
        key = tuple(state)
        index = unique_index[key]
        if type(candidate) is not int or not 0 <= candidate < schema.num_groups:
            raise ValueError("invalid candidate")
        mask = visible[index]
        if mask[candidate] or remaining_budget > sum(not value for value in mask):
            raise ValueError("reacquisition or infeasible budget")

    state_features = encoded_unique.index_select(0, inverse)
    action_features = F.one_hot(
        torch.tensor(candidates, device=device), schema.num_groups).to(torch.float32)
    budget_features = state_features.new_full(
        (len(candidates), 1), remaining_budget / schema.num_groups)
    return self.net(torch.cat((state_features, action_features, budget_features), dim=1)).squeeze(-1)

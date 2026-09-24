"""Run-length state encoding for the unchanged grouped BRiG Q network.

Training and rollout callers enumerate all legal actions for one state
consecutively. Encode and validate that state once, then gather its feature
row for each candidate. This changes no parameters, targets, action order,
or optimizer. It remains a performance variant until exact parity is tested.
"""

import torch
from torch.nn import functional as F

from cbmjev.brig import GroupQ
from cbmjev.learning import encode_states, validate_observed


class GroupQRunLength(GroupQ):
    def forward(self, observed_states, candidates, remaining_budget):
        if len(observed_states) != len(candidates) or not candidates:
            raise ValueError("nonempty aligned states/candidates required")
        if type(remaining_budget) is not int or not 1 <= remaining_budget <= self.schema.num_groups:
            raise ValueError("invalid remaining budget")

        unique, inverse = [], []
        previous_raw, previous_mask, available = None, None, None
        for raw, candidate in zip(observed_states, candidates):
            if previous_raw is None or (raw is not previous_raw and raw != previous_raw):
                state = validate_observed(raw, self.schema)
                previous_mask = self.schema.group_mask(state)
                available = sum(not seen for seen in previous_mask)
                unique.append(state)
                previous_raw = raw
            if type(candidate) is not int or not 0 <= candidate < self.schema.num_groups:
                raise ValueError("invalid candidate")
            if previous_mask[candidate] or remaining_budget > available:
                raise ValueError("reacquisition or infeasible budget")
            inverse.append(len(unique) - 1)

        device = next(self.parameters()).device
        encoded = encode_states(unique, self.schema, device)
        state = encoded.index_select(0, torch.tensor(inverse, dtype=torch.long, device=device))
        action = F.one_hot(torch.tensor(candidates, device=device), self.schema.num_groups).float()
        budget = state.new_full((len(candidates), 1), remaining_budget / self.schema.num_groups)
        return self.net(torch.cat((state, action, budget), dim=1)).squeeze(-1)

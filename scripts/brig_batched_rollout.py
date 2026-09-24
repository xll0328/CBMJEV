"""Batch the unchanged BRiG empty-state rollout target construction.

This is an isolated acceleration candidate. Live BRiG jobs still use the
original serial implementation; no receipt or result silently switches code.
"""

import torch

from cbmjev.brig import _reveal


@torch.no_grad()
def empty_rollout_targets(schema, models, answers_batch, budget, *, max_candidate_evals=8192):
    """Return the serial fitter's four aligned auxiliary-target inputs.

    Model calls are batched across trajectories at each remaining budget.
    Argmin uses ascending candidate order, matching BRiGPolicy.choose's
    `(value, candidate)` tie rule. Chunking bounds transient feature memory.
    """
    if type(budget) is not int or budget < 2 or budget > schema.num_groups:
        raise ValueError("budget must be 2..num_groups")
    if type(max_candidate_evals) is not int or max_candidate_evals < schema.num_groups:
        raise ValueError("candidate-evaluation chunk must fit one complete action set")
    if not answers_batch:
        raise ValueError("nonempty answers_batch required")
    if any(left not in models for left in range(1, budget)):
        raise ValueError("missing previous-budget BRiG model")

    empty = schema.empty_state()
    answers = []
    labels = []
    first_actions = []
    states = []
    for answer, label in answers_batch:
        for action in range(schema.num_groups):
            answers.append(answer)
            labels.append(label)
            first_actions.append(action)
            states.append(_reveal(schema, empty, action, answer))

    for left in range(budget - 1, 0, -1):
        previous = models[left]
        previous.eval()
        next_states = []
        cursor = 0
        while cursor < len(states):
            end = cursor
            candidates = []
            while end < len(states):
                available = tuple(g for g, seen in enumerate(schema.group_mask(states[end])) if not seen)
                if candidates and len(candidates) + len(available) > max_candidate_evals:
                    break
                candidates.extend((end, action) for action in available)
                end += 1
            values = previous([states[i] for i, _ in candidates],
                              [action for _, action in candidates], left)
            if not torch.isfinite(values).all():
                raise ValueError("nonfinite BRiG rollout prediction")
            best = {}
            for (index, action), value in zip(candidates, values.tolist()):
                pair = (value, action)
                if index not in best or pair < best[index]:
                    best[index] = pair
            for index in range(cursor, end):
                next_states.append(_reveal(schema, states[index], best[index][1], answers[index]))
            cursor = end
        states = next_states

    return ([empty] * len(states), first_actions, states, labels)

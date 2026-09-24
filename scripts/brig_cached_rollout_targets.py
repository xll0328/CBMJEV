"""Exact per-budget cache for BRiG empty-state auxiliary risk targets.

At a fixed budget, rollout decisions depend only on the already-frozen lower
budget Q models and each training answer. They do not depend on the current
budget Q parameters or optimizer step. This helper computes the scalar
terminal-risk targets once, in bounded sample chunks, then lets every epoch
reuse those targets without storing all terminal hard states.
"""

import torch

from cbmjev.brig import _terminal
from scripts.brig_batched_rollout import empty_rollout_targets


@torch.no_grad()
def precompute_empty_rollout_risks(schema, models, head, examples, budget,
                                   device="cpu", chunk_size=64):
    """Return an ``[num_examples, num_groups]`` tensor of exact cached risks.

    The row/action ordering matches ``empty_rollout_targets``: training example
    order first, then ascending first-action order. Chunking bounds the
    intermediate trajectory state memory. Values use the same terminal head
    and log-probability floor as the uncached training path.
    """
    if not examples:
        raise ValueError("nonempty examples required")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("positive integer chunk_size required")
    if type(budget) is not int or not 2 <= budget <= schema.num_groups:
        raise ValueError("cached empty rollout requires budget 2..num_groups")

    chunks = []
    for start in range(0, len(examples), chunk_size):
        batch = examples[start:start + chunk_size]
        _, _, terminal_states, terminal_labels = empty_rollout_targets(
            schema, models, batch, budget)
        risks = _terminal(head, terminal_states, terminal_labels, device)
        expected = len(batch) * schema.num_groups
        if risks.numel() != expected or not torch.isfinite(risks).all():
            raise ValueError("invalid empty-rollout risk target batch")
        chunks.append(risks.reshape(len(batch), schema.num_groups))
    return torch.cat(chunks, dim=0)

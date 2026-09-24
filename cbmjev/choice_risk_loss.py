"""Question-balanced one-step utility regression for Choice controls.

This module is intentionally separate from the active soft-CE training path.
The target is privileged training supervision; only visible candidate features
may reach a Choice head. This objective says nothing about multi-step value.
"""

import math

import torch

from .nano_choice import _validate_mask


def choice_utility_regression_loss(logits, realized_errors, incremental_costs,
                                   cost_weight, valid):
    """Mean over questions of mean active squared error to ``-(e + λ c)``.

    ``logits`` is [B,K] with -inf padding. ``realized_errors`` and
    ``incremental_costs`` are finite [B,K] tensors with zero padding. Costs and
    λ are public; errors must be Boolean labels and never model inputs.
    Equal question weighting avoids long candidate menus dominating training.
    """
    _validate_mask(logits, valid, 2)
    for name, value in (("realized_errors", realized_errors),
                        ("incremental_costs", incremental_costs)):
        if (not isinstance(value, torch.Tensor) or value.shape != logits.shape
                or value.device != logits.device or not value.is_floating_point()
                or not torch.isfinite(value).all()):
            raise ValueError(name + " must be a finite colocated [B,K] tensor")
        if torch.count_nonzero(value[~valid]):
            raise ValueError(name + " padding must be zero")
    active_errors = realized_errors[valid]
    active_costs = incremental_costs[valid]
    if not torch.all((active_errors == 0) | (active_errors == 1)):
        raise ValueError("realized errors must be Boolean")
    if torch.any(active_costs < 0):
        raise ValueError("incremental costs must be nonnegative")
    if (type(cost_weight) not in (int, float) or not math.isfinite(cost_weight)
            or cost_weight < 0):
        raise ValueError("cost_weight must be finite nonnegative scalar")
    targets = -(realized_errors.float() + float(cost_weight) * incremental_costs.float())
    if not torch.isfinite(targets[valid]).all():
        raise ValueError("nonfinite utility target")
    per_action = (logits[valid].float() - targets[valid]).square()
    per_question = torch.zeros(logits.shape[0], dtype=per_action.dtype,
                               device=logits.device)
    question_ids = torch.arange(logits.shape[0], device=logits.device)[:, None]
    question_ids = question_ids.expand_as(valid)[valid]
    per_question.scatter_add_(0, question_ids, per_action)
    counts = valid.sum(-1).to(per_question.dtype)
    result = (per_question / counts).mean()
    if not torch.isfinite(result):
        raise ValueError("nonfinite utility regression loss")
    return result

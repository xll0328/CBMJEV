"""Complete-set Choice head adapted from pinned NanoJev, not a full reproduction.

Reference: TianyuCodings/NanoJev commit 76fdfc9ecdca45a9bcef17991a07d3041a87685a,
scripts/train_toy_decisions.py::DecisionModel. This module contains no backbone,
target generator, controller runtime, or official JEV/RLCD implementation.

Adapted portions: Copyright (c) 2026 OpenJev contributors (MIT License).
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
from torch import nn


def _validate_mask(values, valid, ndim):
    if not isinstance(values, torch.Tensor) or values.ndim != ndim:
        raise ValueError("unexpected values shape")
    if not values.is_floating_point():
        raise ValueError("floating point values required")
    if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or valid.ndim != 2 or tuple(valid.shape) != tuple(values.shape[:2])
            or valid.device != values.device):
        raise ValueError("valid must be a colocated boolean [B,K] mask")
    if not valid.numel() or not valid.any(dim=1).all():
        raise ValueError("every nonempty question requires at least one valid candidate")
    if not torch.isfinite(values[valid]).all():
        raise ValueError("nonfinite active candidate values")


class NanoChoiceHead(nn.Module):
    """Permutation-equivariant logits with optional set or independent residual.

    Input raw_features[B,K,D], valid[B,K]. Output float32 logits[B,K], padded
    positions -inf. A singleton is legal. Padding features are ignored, including
    NaNs. STOP is an ordinary candidate whose semantics belong to the caller.
    """

    def __init__(self, hidden_size, set_head="attention"):
        super().__init__()
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if set_head not in ("none", "attention", "independent_mlp"):
            raise ValueError("set_head must be none, attention, or independent_mlp")
        self.hidden_size, self.set_head = hidden_size, set_head
        self.norm = nn.LayerNorm(hidden_size)
        self.scalar = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.scalar.weight, std=0.02)
        nn.init.zeros_(self.scalar.bias)
        if set_head == "attention":
            self.set_project = nn.Linear(hidden_size + 1, 128)
            self.set_attention = nn.MultiheadAttention(128, 4, dropout=0.0, batch_first=True)
            self.set_output = nn.Linear(128, 1)
            nn.init.zeros_(self.set_output.weight)
            nn.init.zeros_(self.set_output.bias)
        elif set_head == "independent_mlp":
            # Matched-capacity control: the attention branch has 66,177
            # post-projection parameters; this per-row branch has 66,561.
            # Both receive the same normalized row and log candidate count.
            self.set_project = nn.Linear(hidden_size + 1, 128)
            self.independent_hidden = nn.Linear(128, 512)
            self.independent_output = nn.Linear(512, 1)
            nn.init.zeros_(self.independent_output.weight)
            nn.init.zeros_(self.independent_output.bias)

    def forward(self, raw_features, valid):
        _validate_mask(raw_features, valid, 3)
        if raw_features.shape[-1] != self.hidden_size:
            raise ValueError("feature width differs from hidden_size")
        clean = raw_features.masked_fill(~valid.unsqueeze(-1), 0)
        h = self.norm(clean)
        logits = self.scalar(h).squeeze(-1).float()
        if self.set_head != "none":
            log_k = valid.sum(-1).float().log()[:, None, None].expand(-1, valid.shape[1], 1)
            u = self.set_project(torch.cat((h, log_k.to(h.dtype)), dim=-1))
            if self.set_head == "attention":
                mixed, _ = self.set_attention(u, u, u, key_padding_mask=~valid, need_weights=False)
                residual = self.set_output(torch.tanh(u + mixed))
            else:
                residual = self.independent_output(torch.tanh(self.independent_hidden(torch.tanh(u))))
            logits = logits + residual.squeeze(-1).float()
        if not torch.isfinite(logits[valid]).all():
            raise ValueError("nonfinite active Choice logits")
        return logits.masked_fill(~valid, -torch.inf)


def choice_soft_target_loss(logits, target, valid):
    """Mean complete-question soft CE; not independent risk or a Q-value loss.

    Each target row must be a valid distribution, zero on padding. Uniform ties
    are allowed; label/teacher meaning is explicitly the caller's responsibility.
    """
    _validate_mask(logits, valid, 2)
    if (not isinstance(target, torch.Tensor) or target.shape != logits.shape
            or target.device != logits.device or not target.is_floating_point()
            or not torch.isfinite(target).all() or (target < 0).any()):
        raise ValueError("finite nonnegative colocated target distribution required")
    if target[~valid].count_nonzero() or not torch.allclose(
            target.float().sum(-1), torch.ones(logits.shape[0], device=logits.device),
            atol=1e-6, rtol=0):
        raise ValueError("targets must sum to one over exactly the valid candidates")
    normalized = logits.float().masked_fill(~valid, -torch.inf).log_softmax(-1)
    # Never form 0 * -inf at padded locations.
    normalized = normalized.masked_fill(~valid, 0)
    return -(target.float() * normalized).sum(-1).mean()

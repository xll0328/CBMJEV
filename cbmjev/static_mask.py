"""Global learned fixed-K concept-group mask, using an explicitly biased ST gate.

Local baseline, not a named SOTA reproduction or an optimal subset algorithm.
Complete automatic responses are an offline training input only. Deployment calls
selected_groups() with no sample input, then measures exactly those groups.
"""
import copy
import hashlib
import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import stable_hash
from .learning import encode_states, training_rows, validate_training_separation


class _CardinalitySigmoid(torch.autograd.Function):
    """First-order implicit derivative of a fixed-mass logistic projection."""
    @staticmethod
    def forward(ctx, logits, k, temperature):
        x = logits.detach().to(torch.float64)
        if k in (0, x.numel()):
            soft = torch.full_like(x, float(k > 0))
        else:
            ranked = x.sort(descending=True).values
            # Half each term before addition avoids overflow at large finite x.
            center = ranked[k - 1] * .5 + ranked[k] * .5
            delta = x - center
            normalized = delta / temperature
            overflow = ~torch.isfinite(delta)
            # Opposite-sign extremes can overflow subtraction even when their
            # temperature-scaled difference is representable. Scale first only
            # in that case (scaling all entries first could create inf-inf).
            normalized[overflow] = x[overflow] / temperature - center / temperature
            # Remaining +/-inf from scaling is safe: sigmoid saturates, never NaN.
            # At center, top K have z>=0 and the rest z<=0. This margin
            # brackets the root to ~machine precision, including wide flat gaps.
            bound = math.log(x.numel()) + 40.
            low, high = x.new_tensor(-bound), x.new_tensor(bound)
            for _ in range(100):
                midpoint = (low + high) * .5
                candidate = torch.sigmoid(normalized - midpoint)
                if float(candidate.sum()) > k:
                    low = midpoint
                else:
                    high = midpoint
            soft = torch.sigmoid(normalized - (low + high) * .5)
            if not torch.isfinite(soft).all() or abs(float(soft.sum()) - k) > 1e-10 * max(1, x.numel()):
                raise ValueError("cardinality gate numerical solve failed")
        ctx.save_for_backward(soft)
        ctx.temperature = temperature
        ctx.input_dtype = logits.dtype
        return soft.to(logits.dtype)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        soft, = ctx.saved_tensors
        sensitivity = soft * (1. - soft)
        total = sensitivity.sum()
        if float(total) == 0.:
            gradient = torch.zeros_like(soft)
        else:
            v = upstream.to(torch.float64)
            weighted_mean = (sensitivity * v).sum() / total
            gradient = (sensitivity * (v - weighted_mean)) / ctx.temperature
        return gradient.to(ctx.input_dtype), None, None


def cardinality_soft_gate(logits, k, *, temperature=1.):
    """Return sigmoid((logits-tau)/T), with sum soft=K (floating precision).

    Tau is solved in float64 by bounded bisection. Backward is the implicit
    Jacobian (diag(s)-s*s.T/sum(s))/T, s=soft*(1-soft), which annihilates
    common-mode shifts. Fully saturated numerical solutions use zero derivative.
    First derivatives only; no gradient for integer K or scalar temperature.
    This continuous gate alone is not a hard-subset optimizer or unbiased ST.
    """
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1 or not logits.numel() or not logits.is_floating_point():
        raise ValueError("logits must be a nonempty one-dimensional floating tensor")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    if type(k) is not int or not 0 <= k <= logits.numel():
        raise ValueError("k must be an integer within the logit count")
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return _CardinalitySigmoid.apply(logits, k, float(temperature))


def cardinality_st_topk(logits, k, *, temperature=1., seed=17):
    """Exact hard top-K forward, cardinality-aware soft backward (still biased).

    Optional cardinality-aware variant; the existing sigmoid-ST default remains
    unchanged. This does not make the hard-subset gradient unbiased.
    """
    soft = cardinality_soft_gate(logits, k, temperature=temperature)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    order = list(range(logits.numel()))
    random.Random(seed).shuffle(order)
    ranks = {g: i for i, g in enumerate(order)}
    values = logits.detach().cpu().tolist()
    chosen = sorted(range(len(values)), key=lambda g: (-values[g], ranks[g]))[:k]
    hard = torch.zeros_like(logits)
    hard[chosen] = 1.
    return hard + (soft - soft.detach())


class StaticGroupMask(nn.Module):
    def __init__(self, schema, k, *, temperature=1., seed=17, device="cpu", relaxation="sigmoid"):
        super().__init__()
        if type(k) is not int or not 0 <= k <= schema.num_groups:
            raise ValueError("k must be an integer within the group count")
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if relaxation not in ("sigmoid", "cardinality"):
            raise ValueError("relaxation must be sigmoid or cardinality")
        self.relaxation = relaxation
        self.schema, self.k, self.temperature, self.seed = schema, k, float(temperature), seed
        self.logits = nn.Parameter(torch.zeros(schema.num_groups, device=device),
                                   requires_grad=0 < k < schema.num_groups)
        order = list(range(schema.num_groups))
        random.Random(seed).shuffle(order)
        ranks = [order.index(g) for g in range(schema.num_groups)]
        self.register_buffer("tie_rank", torch.tensor(ranks, dtype=torch.long, device=device))

    def selected_groups(self):
        """Sample-independent global hard subset; no observations/IDs/labels accepted."""
        values = self.logits.detach().cpu().tolist()
        if any(not math.isfinite(v) for v in values):
            raise ValueError("nonfinite mask logits")
        ranks = self.tie_rank.cpu().tolist()
        selected = sorted(range(self.schema.num_groups), key=lambda g: (-values[g], ranks[g]))[:self.k]
        return tuple(sorted(selected))

    def forward(self):
        hard = torch.zeros_like(self.logits)
        if self.k:
            hard[list(self.selected_groups())] = 1.
        if self.k in (0, self.schema.num_groups):
            return hard
        soft = (cardinality_soft_gate(self.logits, self.k, temperature=self.temperature)
                if self.relaxation == "cardinality" else torch.sigmoid(self.logits / self.temperature))
        # Parentheses make forward exactly hard, while backward differentiates soft.
        # This is NOT the gradient of hard top-K or an unbiased relaxation estimator.
        return hard + (soft - soft.detach())

    def encode_training_responses(self, responses):
        """Offline-only differentiable mask; hard values equal encode_states(masked).

        Does not consume labels. Never use complete hidden response rows as an
        online policy API; online acquisition uses selected_groups() alone.
        """
        responses = list(responses)
        if not responses:
            raise ValueError("nonempty automatic response batch required")
        for state in responses:
            self.schema.validate_state(state, complete=True)
        encoded = encode_states(responses, self.schema, self.logits.device)
        gate = self()
        owner = [None] * self.schema.num_atoms
        for g, group in enumerate(self.schema.groups):
            for atom in group.atoms:
                owner[atom] = g
        atom_gates = gate[torch.tensor(owner, device=gate.device)]
        # encode_states order: categorical blocks for each atom, then atom visibility.
        scale = torch.cat([atom_gates[a].expand(n) for a, n in enumerate(self.schema.num_categories)]
                          + [atom_gates])
        return encoded * scale.unsqueeze(0)


def _head_digest(network):
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def fit_static_mask(rows, head, schema, config=None, *, excluded_head_group_ids):
    """Return (global mask, report); caller must authenticate head/cache ancestry.

    Verifies coverage of supplied head exclusions, not authenticity of that list.
    Wrapper must validate exclusion against fitted-head provenance and authenticate
    automatic predicted-concept rows. No gold-concept training is authorized here.
    """
    defaults = dict(k=1, seed=17, device="cpu", epochs=30, batch_size=64,
                    learning_rate=.01, temperature=1., relaxation="sigmoid")
    if set(config or {}) - set(defaults):
        raise ValueError("unknown static-mask configuration")
    cfg = {**defaults, **(config or {})}
    for key in ("epochs", "batch_size"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError("positive integer required: " + key)
    if type(cfg["learning_rate"]) not in (int, float) or not math.isfinite(cfg["learning_rate"]) or cfg["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    rows = list(rows)
    validate_training_separation(rows)
    records = training_rows(rows, "policy_fit", schema)
    selected_rows = [r for r in rows if r.get("split") == "policy_fit"]
    groups = {r["group_id"] for r in selected_rows}
    if isinstance(excluded_head_group_ids, (str, bytes)) or not groups <= set(excluded_head_group_ids):
        raise ValueError("task head must exclude every policy-fit group")
    if head.schema.hash != schema.hash:
        raise ValueError("head/schema mismatch")
    before = _head_digest(head.network)
    # Clone exact weights to avoid changing the caller's mode, gradients or flags.
    frozen_head = copy.deepcopy(head.network).to(cfg["device"]).eval().requires_grad_(False)
    mask = StaticGroupMask(schema, cfg["k"], temperature=cfg["temperature"],
                           seed=cfg["seed"], device=cfg["device"], relaxation=cfg["relaxation"])
    history, selections, updates, exposures = [], [], 0, 0
    generator = torch.Generator().manual_seed(cfg["seed"])
    if 0 < cfg["k"] < schema.num_groups:
        optimizer = torch.optim.Adam(mask.parameters(), lr=cfg["learning_rate"])
        for _ in range(cfg["epochs"]):
            total = 0.
            for indices in torch.randperm(len(records), generator=generator).split(cfg["batch_size"]):
                batch = [records[int(i)] for i in indices]
                answers, labels = zip(*batch)
                features = mask.encode_training_responses(answers)
                logits = frozen_head(features)
                loss = F.cross_entropy(logits, torch.tensor(labels, device=logits.device))
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite static-mask loss")
                optimizer.zero_grad()
                loss.backward()
                if mask.logits.grad is None or not torch.isfinite(mask.logits.grad).all():
                    raise ValueError("invalid static-mask gate gradient")
                optimizer.step()
                total += float(loss.detach()) * len(batch)
                updates += 1
                exposures += len(batch)
            history.append(total / len(records))
            selections.append(list(mask.selected_groups()))
    if _head_digest(head.network) != before or _head_digest(frozen_head) != before:
        raise ValueError("frozen head weights changed")
    report = dict(method="global_static_ST_topK_" + cfg["relaxation"] + "_v1", config=cfg, schema_hash=schema.hash,
                  head_tensor_sha256=before, split="policy_fit", sample_count=len(records),
                  group_count=len(groups), head_exclusion_coverage_checked=True,
                  head_exclusion_provenance_checked=False, input_provenance_checked=False,
                  training_loss=history, loss="unweighted_CE", loss_aggregation="sample_weighted_per_epoch",
                  estimator="biased_straight_through_hard_topK_forward_" + cfg["relaxation"] + "_backward",
                  backward_preserves_cardinality=cfg["relaxation"] == "cardinality", optimizer_steps=updates,
                  event_exposures=exposures, effective_epochs=len(history),
                  endpoint_no_training=cfg["k"] in (0, schema.num_groups),
                  selected_group_ids=list(mask.selected_groups()), selected_group_history=selections,
                  parameter_count=schema.num_groups, sample_independent=True, head_weights_unchanged=True,
                  ordered_training_rows_sha256=stable_hash([(r["sample_id"], r["group_id"], r["z"], r["y"])
                                                           for r in selected_rows]),
                  paper_evidence=False, global_optimality_claim=False)
    return mask.eval(), report

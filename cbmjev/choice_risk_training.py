"""Isolated paired one-step utility-regression control for Choice.

Do not conflate this objective with the existing soft-CE Choice artifact or
with multi-step Bellman-value learning. This module does not alter that fit.
"""

import copy
from dataclasses import asdict
import hashlib
import math

import torch

from .choice_risk_loss import choice_utility_regression_loss
from .choice_targets import ChoiceExample, ChoiceInputs, ChoiceSupervision
from .choice_training import _source_hash, _tensor_hash, _weights
from .contracts import stable_hash
from .nano_choice import NanoChoiceHead


def fit_choice_utility_pair(examples, features, *, epochs, batch_size, lr, seed,
                            device="cpu"):
    """Fit scalar and candidate-attention heads to negative realized utility.

    Matching the existing fitter's seed, batch order and initial weights makes
    this an objective control. It does not certify upstream split/provenance.
    """
    if (type(examples) is not tuple or not examples or type(features) is not tuple
            or len(examples) != len(features)):
        raise ValueError("aligned nonempty immutable tuples required")
    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be positive integer")
    if (type(seed) is not int or seed < 0 or type(lr) not in (int, float)
            or not math.isfinite(lr) or lr <= 0):
        raise ValueError("invalid seed or learning rate")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("supported available CPU/CUDA device required")
    width, weight = None, None
    for example, value in zip(examples, features):
        if (type(example) is not ChoiceExample or type(example.model_inputs) is not ChoiceInputs
                or type(example.supervision) is not ChoiceSupervision
                or type(example.derived_id) is not str or not example.derived_id):
            raise ValueError("ChoiceExample required")
        k = len(example.model_inputs.actions)
        if (not isinstance(value, torch.Tensor) or value.ndim != 2 or value.device.type != "cpu"
                or value.requires_grad or value.grad_fn is not None or not value.is_floating_point()
                or value.shape[0] != k or k < 1 or value.shape[1] < 1
                or not torch.isfinite(value).all()):
            raise ValueError("finite detached CPU [K,D] features required")
        if width is not None and width != value.shape[1]:
            raise ValueError("feature widths differ")
        width = value.shape[1]
        inputs, supervision = example.model_inputs, example.supervision
        if (len(inputs.incremental_costs) != k or len(supervision.realized_errors) != k
                or any(type(e) not in (int, float) or e not in (0, 1)
                       for e in supervision.realized_errors)
                or any(type(c) not in (int, float) or not math.isfinite(c) or c < 0
                       for c in inputs.incremental_costs)
                or type(inputs.cost_weight) not in (int, float)
                or not math.isfinite(inputs.cost_weight) or inputs.cost_weight < 0):
            raise ValueError("aligned Boolean errors and nonnegative finite costs required")
        if weight is not None and weight != inputs.cost_weight:
            raise ValueError("matched control requires one common cost weight")
        weight = float(inputs.cost_weight)
        if not all(math.isfinite(float(e) + weight * float(c))
                   for e, c in zip(supervision.realized_errors, inputs.incremental_costs)):
            raise ValueError("nonfinite realized utility")
    before = _source_hash(examples, features)
    saved_examples = copy.deepcopy(examples)
    saved_features = tuple(value.detach().clone().float() for value in features)
    if any(not torch.isfinite(value).all() for value in saved_features):
        raise ValueError("feature conversion to float32 overflowed")

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        scalar, attention = NanoChoiceHead(width, "none"), NanoChoiceHead(width, "attention")
    attention.norm.load_state_dict(scalar.norm.state_dict())
    attention.scalar.load_state_dict(scalar.scalar.state_dict())
    shared_hash = _weights(scalar, shared=True)
    if shared_hash != _weights(attention, shared=True):
        raise AssertionError("paired initialization differs")
    heads = (scalar.to(device), attention.to(device))

    def batch(indices):
        size = max(saved_features[i].shape[0] for i in indices)
        x = torch.zeros((len(indices), size, width), device=device)
        valid = torch.zeros((len(indices), size), dtype=torch.bool, device=device)
        errors = torch.zeros((len(indices), size), device=device)
        costs = torch.zeros((len(indices), size), device=device)
        for row, i in enumerate(indices):
            k = saved_features[i].shape[0]
            x[row, :k] = saved_features[i].to(device)
            valid[row, :k] = True
            errors[row, :k] = torch.tensor(saved_examples[i].supervision.realized_errors,
                                           dtype=torch.float32, device=device)
            costs[row, :k] = torch.tensor(saved_examples[i].model_inputs.incremental_costs,
                                          dtype=torch.float32, device=device)
        return x, valid, errors, costs

    initial_logits = hashlib.sha256()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            x, valid, _, _ = batch(list(range(start, min(start + batch_size, len(examples)))))
            left, right = heads[0](x, valid), heads[1](x, valid)
            if not torch.equal(left[valid], right[valid]):
                raise AssertionError("paired initial active logits differ")
            initial_logits.update(_tensor_hash(left[valid]).encode("ascii"))
    initial_hashes = [_weights(head) for head in heads]
    optimizers = [torch.optim.Adam(head.parameters(), lr=lr, weight_decay=0) for head in heads]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses, order_hashes = [[], []], []
    steps = 0
    for _ in range(epochs):
        order = torch.randperm(len(examples), generator=generator).tolist()
        order_hashes.append(stable_hash(order))
        sums = [0.0, 0.0]
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            x, valid, errors, costs = batch(indices)
            for index, (head, optimizer) in enumerate(zip(heads, optimizers)):
                loss = choice_utility_regression_loss(head(x, valid), errors, costs,
                                                       weight, valid)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                sums[index] += float(loss.detach()) * len(indices)
            steps += 1
        for index in range(2):
            losses[index].append(sums[index] / len(examples))
    if before != _source_hash(examples, features):
        raise ValueError("Choice source mutated during paired fitting")
    for head in heads:
        if any(not torch.isfinite(parameter).all() for parameter in head.parameters()):
            raise ValueError("nonfinite fitted Choice weights")
        head.eval().requires_grad_(False)
    report = {"format": "cbmjev-paired-choice-utility-fit-v1",
        "comparison": "scalar-vs-attention-set-on-identical-supplied-features",
        "objective_control_vs": "cbmjev-paired-choice-head-fit-v1 soft-CE",
        "source_content_sha256": before, "upstream_provenance_verified": False,
        "provenance_scope": "content binding only; upstream reader/encoder verification required",
        "effective_features_sha256": stable_hash([_tensor_hash(v) for v in saved_features]),
        "derived_ids_in_order": [e.derived_id for e in saved_examples],
        "config": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed,
                   "device": str(device), "optimizer": "Adam", "weight_decay": 0,
                   "feature_dtype": "float32", "loss": "question-mean-active-utility-MSE",
                   "cost_weight": weight},
        "initial_shared_tensors_sha256": shared_hash,
        "initial_active_logits_equal": True,
        "initial_active_logits_sha256": initial_logits.hexdigest(),
        "epoch_order_sha256": order_hashes, "order_sha256": stable_hash(order_hashes),
        "questions_exposed_per_head": len(examples) * epochs,
        "candidates_exposed_per_head": sum(v.shape[0] for v in saved_features) * epochs,
        "optimizer_steps_per_head": steps, "teachers_regenerated": False,
        "heads": {name: {"initial_weights_sha256": initial_hashes[i],
                         "final_weights_sha256": _weights(heads[i]), "epoch_loss": losses[i]}
                  for i, name in enumerate(("scalar", "attention"))}}
    return scalar, attention, report

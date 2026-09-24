"""Paired scalar/set-head fitting on caller-supplied immutable Choice features.

This low-level fitter checks content, not upstream provenance authenticity. It
does not construct teachers, encode inputs, or certify a representation as MLP.
"""
import copy
from dataclasses import asdict
import hashlib
import math

import torch

from .choice_targets import ChoiceExample, ChoiceInputs, ChoiceSupervision
from .contracts import stable_hash
from .nano_choice import NanoChoiceHead, choice_soft_target_loss


def _tensor_hash(tensor):
    raw = tensor.detach().cpu().contiguous()
    return stable_hash({"dtype": str(raw.dtype), "shape": list(raw.shape),
        "bytes_sha256": hashlib.sha256(raw.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()})


def _weights(head, shared=False):
    return stable_hash({name: _tensor_hash(value) for name, value in head.state_dict().items()
                       if not shared or name.startswith(("norm.", "scalar."))})


def _source_hash(examples, features):
    return stable_hash({"examples_in_order": [asdict(example) for example in examples],
                        "features_in_order": [_tensor_hash(value) for value in features]})


def fit_choice_head_pair(examples, features, *, epochs, batch_size, lr, seed,
                         device="cpu", capacity_control=False):
    """Return scalar head, attention-set head, and content-bound paired reports.

    Inputs must remain unchanged during the call. Features are cloned before
    fitting and only feature tensors/valid masks reach either head's forward.
    The existing teacher probabilities are consumed verbatim by the loss.
    """
    if type(capacity_control) is not bool:
        raise ValueError("capacity_control must be boolean")
    if (not isinstance(examples, tuple) or not examples or not isinstance(features, tuple)
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
    width = None
    for example, value in zip(examples, features):
        if (not isinstance(example, ChoiceExample)
                or not isinstance(example.model_inputs, ChoiceInputs)
                or not isinstance(example.supervision, ChoiceSupervision)
                or not isinstance(example.derived_id, str) or not example.derived_id):
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
        probabilities = example.supervision.teacher_probabilities
        errors, temperature = example.supervision.realized_errors, example.supervision.temperature
        if (len(errors) != k or any(type(e) not in (int, float) or e not in (0, 1) for e in errors)
                or type(temperature) not in (int, float) or not math.isfinite(temperature)
                or temperature <= 0):
            raise ValueError("aligned realized errors and positive teacher temperature required")
        if (len(probabilities) != k or any(type(p) not in (int, float) or not math.isfinite(p) or p < 0
                                         for p in probabilities)
                or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6, rel_tol=0)):
            raise ValueError("aligned teacher distribution required")
    before = _source_hash(examples, features)
    saved_examples = copy.deepcopy(examples)
    saved_features = tuple(value.detach().clone().float() for value in features)
    if any(not torch.isfinite(value).all() for value in saved_features):
        raise ValueError("feature conversion to float32 overflowed")

    # Initialization is independent of labels, teacher distributions and features.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        scalar, attention = NanoChoiceHead(width, "none"), NanoChoiceHead(width, "attention")
        independent = NanoChoiceHead(width, "independent_mlp") if capacity_control else None
    names = ("scalar", "attention", "independent_mlp") if capacity_control else ("scalar", "attention")
    model_list = (scalar, attention, independent) if capacity_control else (scalar, attention)
    for head in model_list[1:]:
        head.norm.load_state_dict(scalar.norm.state_dict())
        head.scalar.load_state_dict(scalar.scalar.state_dict())
    shared_hash = _weights(scalar, shared=True)
    if any(shared_hash != _weights(head, shared=True) for head in model_list[1:]):
        raise AssertionError("paired initialization differs")
    heads = tuple(head.to(device) for head in model_list)

    def batch(indices):
        size = max(saved_features[i].shape[0] for i in indices)
        x = torch.zeros((len(indices), size, width), device=device)
        valid = torch.zeros((len(indices), size), dtype=torch.bool, device=device)
        target = torch.zeros((len(indices), size), device=device)
        for row, i in enumerate(indices):
            k = saved_features[i].shape[0]
            x[row, :k] = saved_features[i].to(device)
            valid[row, :k] = True
            target[row, :k] = torch.tensor(saved_examples[i].supervision.teacher_probabilities,
                                          dtype=torch.float32, device=device)
        return x, valid, target

    initial_logits = hashlib.sha256()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            x, valid, _ = batch(list(range(start, min(start + batch_size, len(examples)))))
            left = heads[0](x, valid)
            if any(not torch.equal(left[valid], head(x, valid)[valid]) for head in heads[1:]):
                raise AssertionError("paired initial active logits differ")
            initial_logits.update(_tensor_hash(left[valid]).encode("ascii"))
    initial_hashes = [_weights(head) for head in heads]
    optimizers = [torch.optim.Adam(head.parameters(), lr=lr, weight_decay=0) for head in heads]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses, order_hashes = [[] for _ in heads], []
    steps = 0
    for _ in range(epochs):
        order = torch.randperm(len(examples), generator=generator).tolist()
        order_hashes.append(stable_hash(order))
        sums = [0.0 for _ in heads]
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            x, valid, target = batch(indices)
            for index, (head, optimizer) in enumerate(zip(heads, optimizers)):
                loss = choice_soft_target_loss(head(x, valid), target, valid)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite Choice loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                sums[index] += float(loss.detach()) * len(indices)
            steps += 1
        for index in range(len(heads)):
            losses[index].append(sums[index] / len(examples))
    if before != _source_hash(examples, features):
        raise ValueError("Choice source mutated during paired fitting")
    for head in heads:
        if any(not torch.isfinite(parameter).all() for parameter in head.parameters()):
            raise ValueError("nonfinite fitted Choice weights")
        head.eval().requires_grad_(False)
    report = {"format": ("cbmjev-capacity-controlled-choice-head-fit-v1" if capacity_control
                         else "cbmjev-paired-choice-head-fit-v1"),
        "comparison": ("scalar-vs-attention-set-vs-independent-mlp-on-identical-supplied-features"
                       if capacity_control else "scalar-vs-attention-set-on-identical-supplied-features"),
        "source_content_sha256": before, "upstream_provenance_verified": False,
        "provenance_scope": "content binding only; upstream reader/encoder verification required",
        "effective_features_sha256": stable_hash([_tensor_hash(v) for v in saved_features]),
        "derived_ids_in_order": [e.derived_id for e in saved_examples],
        "config": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed,
                   "device": str(device), "optimizer": "Adam", "weight_decay": 0,
                   "feature_dtype": "float32", "loss": "question-mean-soft-CE"},
        "initial_shared_tensors_sha256": shared_hash,
        "initial_active_logits_equal": True, "initial_active_logits_sha256": initial_logits.hexdigest(),
        "epoch_order_sha256": order_hashes, "order_sha256": stable_hash(order_hashes),
        "questions_exposed_per_head": len(examples) * epochs,
        "candidates_exposed_per_head": sum(v.shape[0] for v in saved_features) * epochs,
        "optimizer_steps_per_head": steps,
        "teachers_regenerated": False,
        "heads": {name: {"initial_weights_sha256": initial_hashes[i],
                         "final_weights_sha256": _weights(heads[i]), "epoch_loss": losses[i]}
                  for i, name in enumerate(names)}}
    if capacity_control:
        report["capacity_control"] = {"candidate_interaction": False,
            "same_visible_features_and_log_candidate_count": True,
            "parameter_count": {name: sum(p.numel() for p in head.parameters())
                                for name, head in zip(names, heads)}}
    return (*heads, report)

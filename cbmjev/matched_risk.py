"""Matched-event MLP diagnostic for the local Nano-style risk adapter.

This does not reproduce official NanoJev. Supply the *same* event tuple to this
fitter and ``fit_nano_risk``; no targets, masks, or actions are regenerated here.
Event identity proves content equality, not correctness of upstream ancestry.
"""
from __future__ import annotations

import hashlib
import json
import math

import torch
from torch.nn import functional as F

from .learning import (ActionController, encode_actions, encode_states,
                       normalize_config, validate_action, validate_observed)
from .nanojev import RiskTrainingExample, risk_prompts


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def matched_event_manifest(examples, schema):
    """Validate a materialized sequence and bind its order, duplicates and labels.

    Lists are accepted as containers but every event and its payload must be
    immutable. Fitters should share a tuple snapshot to avoid caller mutation.
    No task labels, raw payload, sample IDs or unseen concept answers are accepted.
    """
    if not isinstance(examples, (tuple, list)) or not examples:
        raise ValueError("nonempty materialized RiskTrainingExample sequence required")
    records = []
    for event in examples:
        if (type(event) is not RiskTrainingExample or event.split != "policy_fit"
                or type(event.observed) is not tuple or type(event.action) is not tuple
                or type(event.error) not in (int, float)
                or not math.isfinite(event.error) or not 0 <= event.error <= 1):
            raise ValueError("immutable policy_fit events with finite [0,1] targets required")
        state = validate_observed(event.observed, schema)
        validate_action(event.action, state, schema)
        # Match the adapter's group/action validation as well as the MLP contract.
        risk_prompts(schema, state, [event.action])
        records.append({"observed": event.observed, "action": event.action,
                        "error": float(event.error), "split": event.split})
    return {"event_sha256": _digest({"schema_hash": schema.hash, "events": records}),
            "schema_hash": schema.hash, "split": "policy_fit", "n": len(records),
            "event_order_bound": True, "duplicate_events_preserved": True,
            "upstream_target_provenance_verified": False}


def fit_matched_mlp_risk(examples, schema, *, epochs=5, batch_size=16,
                         learning_rate=.001, seed=0, hidden=128, device="cpu"):
    """Fit the ordinary controller to exactly the supplied independent risk events.

    Mirrors ``fit_nano_risk``: Adam (zero weight decay), one seeded CPU randperm
    per epoch, every supplied event once per epoch, mean BCE per minibatch. A
    short final minibatch has equal optimizer-step weight, in BOTH fitters.
    Initialization is explicit here; the Nano scorer must separately be created
    with the intended seed before fitting (its fitter does not initialize it).
    """
    manifest = matched_event_manifest(examples, schema)
    events = tuple(examples)
    for name, value in (("epochs", epochs), ("batch_size", batch_size), ("hidden", hidden)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if (type(learning_rate) not in (int, float) or not math.isfinite(learning_rate)
            or learning_rate <= 0):
        raise ValueError("learning_rate must be finite and positive")
    cfg = normalize_config({"objective": "risk", "seed": seed, "device": device,
                            "hidden": hidden, "dropout": 0., "weight_decay": 0.,
                            "policy_epochs": epochs, "batch_size": batch_size,
                            "learning_rate": learning_rate}, schema)
    # CPU construction avoids dependence on CUDA RNG state; transfer after init.
    # Preserve caller RNG so a later seeded Nano initialization is not perturbed.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        controller = ActionController(schema, {**cfg, "device": "cpu"})
    controller.config = cfg
    controller.device = torch.device(cfg["device"])
    controller.network.to(controller.device)
    optimizer = torch.optim.Adam(controller.network.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    weighted_losses, minibatch_losses, order_hashes = [], [], []
    for _ in range(epochs):
        controller.network.train()
        order = torch.randperm(len(events), generator=generator)
        order_hashes.append(_digest(order.tolist()))
        weighted_total, step_total, steps = 0., 0., 0
        for indices in order.split(batch_size):
            batch = [events[int(index)] for index in indices]
            # Forward inputs intentionally project ONLY observed states/actions.
            inputs = torch.cat((encode_states([e.observed for e in batch], schema, controller.device),
                                encode_actions([e.action for e in batch], schema, controller.device)), -1)
            logits = controller.network(inputs).flatten()
            targets = torch.tensor([e.error for e in batch], dtype=logits.dtype, device=logits.device)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite matched risk training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            value = float(loss.detach())
            weighted_total += value * len(batch)
            step_total += value
            steps += 1
        weighted_losses.append(weighted_total / len(events))
        minibatch_losses.append(step_total / steps)
    if matched_event_manifest(events, schema) != manifest:
        raise ValueError("risk training events changed during fitting")
    controller.network.eval().requires_grad_(False)
    parameters = sum(p.numel() for p in controller.network.parameters())
    report = {**manifest, "objective": "risk", "seed": seed, "device": str(controller.device),
              "config": cfg, "epochs": epochs, "batch_size": batch_size,
              "learning_rate": learning_rate, "optimizer": "Adam", "weight_decay": 0.,
              "dropout": 0., "loss": "independent_event_BCE", "training_loss": weighted_losses,
              "aggregation": "sample-weighted mean of online minibatch event BCE",
              "training_loss_minibatch_mean": minibatch_losses,
              "nano_loss_reporting_difference": "fit_nano_risk reports unweighted mean of minibatch means",
              "optimization_weighting": "mean BCE per minibatch; equal optimizer-step weight including tail",
              "examples_per_epoch": [len(events)] * epochs, "event_exposures": len(events) * epochs,
              "optimizer_steps": math.ceil(len(events) / batch_size) * epochs,
              "epoch_order_sha256": order_hashes, "parameter_count": parameters,
              "trainable_parameter_count_during_fit": parameters,
              "initialization": "CPU torch.manual_seed(seed), independent of event labels",
              "model_input_fields": ["observed", "action"], "targets_regenerated": False,
              "baseline_comparator": "local NanoRiskController with identical supplied events",
              "calibrated": False, "paper_evidence": False,
              "limitations": ["not official NanoJev reproduction",
                              "same exposure does not match architecture capacity or compute",
                              "Nano scorer initialization must be seeded separately",
                              "event digest does not verify split ancestry or label-generation independence"]}
    return controller, report

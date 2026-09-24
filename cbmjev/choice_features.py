"""Visible-only Choice serializers; no model, target, metadata, or feature cache.

Each returned row/path is one candidate of the supplied complete question.
The caller preserves occurrence multiplicity and pads questions outside this
module. Candidate ordering is preserved, never used as a positional feature.
"""
import json
import math

import torch

from .choice_targets import ChoiceInputs
from .contracts import Schema, stable_hash
from .learning import encode_actions, encode_states, validate_action


CHOICE_FEATURE_VERSION = "cbmjev-visible-choice-features-v2-compact-missing-groups"


def validate_choice_inputs(inputs, schema):
    if type(inputs) is not ChoiceInputs or not isinstance(schema, Schema):
        raise ValueError("only ChoiceInputs and Schema are accepted")
    if any(type(v) is not tuple for v in (inputs.observed, inputs.actions, inputs.incremental_costs)):
        raise ValueError("visible state, actions and costs must be immutable tuples")
    schema.validate_state(inputs.observed)
    if (type(inputs.remaining_groups) is not int
            or not 0 <= inputs.remaining_groups <= schema.num_groups):
        raise ValueError("remaining group budget must be an integer within schema bounds")
    if not inputs.actions or len(inputs.actions) != len(inputs.incremental_costs):
        raise ValueError("nonempty aligned candidates/costs required")
    if any(type(a) is not tuple for a in inputs.actions):
        raise ValueError("actions must be immutable tuples")
    for action in inputs.actions:
        validate_action(action, inputs.observed, schema)
        if len(action) > inputs.remaining_groups:
            raise ValueError("candidate exceeds remaining group budget")
    if len(set(inputs.actions)) != len(inputs.actions) or () not in inputs.actions:
        raise ValueError("unique actions including STOP required")
    for cost in (*inputs.incremental_costs, inputs.cost_weight):
        if type(cost) not in (int, float) or cost < 0 or cost > torch.finfo(torch.float32).max:
            raise ValueError("declared costs and cost weight must be finite nonnegative numbers")
        # Structured inputs are float32. Refuse silent overflow/Inf conversion.
        if not math.isfinite(cost):
            raise ValueError("declared cost exceeds finite float32 feature range")
    if inputs.incremental_costs[inputs.actions.index(())] != 0:
        raise ValueError("STOP incremental cost must be zero")
    return inputs


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def choice_prompts(inputs, schema):
    """Return tuple[str] for scorer.features; only observed semantic evidence."""
    validate_choice_inputs(inputs, schema)
    state = []
    for concept, value in zip(schema.concepts, inputs.observed):
        if value == -1:
            continue
        item = {"concept": concept.description}
        if value < len(concept.values):
            item.update(status="OBSERVED", value=concept.values[value])
        else:
            item["status"] = "UNCERTAIN" if value == len(concept.values) else "NOT_APPLICABLE"
        state.append(item)
    prompts = []
    for action, cost in zip(inputs.actions, inputs.incremental_costs):
        candidate = {"type": "STOP"} if not action else {
            "type": "ACQUIRE", "concepts": [
                {"concept": schema.concepts[a].description,
                 "possible_values": list(schema.concepts[a].values)}
                for a in schema.expand(action)]}
        prompts.append(_canonical({"format": CHOICE_FEATURE_VERSION,
            "question_type": "action_choice",
            "question": "Choose the next feasible acquisition or STOP under the stated cost weight and remaining budget.",
            "observed_state": state, "remaining_group_budget": inputs.remaining_groups,
            "unobserved_group_indices": [i for i, present in enumerate(schema.group_mask(inputs.observed)) if not present],
            "missing_semantics": "Every atom in an unobserved group is UNOBSERVED, not false, uncertain, or not applicable. Group indices are fixed schema query slots, not sample identifiers.",
            "cost_weight": float(inputs.cost_weight), "candidate": candidate,
            "candidate_incremental_cost": float(cost)}))
    return tuple(prompts)


def choice_feature_width(schema):
    return sum(schema.num_categories) + schema.num_atoms + schema.num_groups + 1 + 3


def encode_choice_features(inputs, schema, device="cpu"):
    """Float32 [K,D]: state one-hots+visibility, group/STOP bits, budget, λ, cost.

    Last three coordinates are raw declared numeric values, not learned scaling.
    There are no sample/group metadata IDs, hidden labels or pretrained features.
    """
    validate_choice_inputs(inputs, schema)
    state = encode_states([inputs.observed], schema, device=device)
    actions = encode_actions(inputs.actions, schema, device=device)
    numeric = torch.tensor([[inputs.remaining_groups, inputs.cost_weight, c]
                            for c in inputs.incremental_costs], dtype=torch.float32, device=device)
    result = torch.cat((state.expand(len(inputs.actions), -1), actions, numeric), dim=-1)
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite structured Choice feature")
    return result


def choice_feature_manifest(inputs, schema):
    """Hash exact visible prompt order and a fixed structured-feature recipe.

    Schema hash is provenance only; it is not inserted into model inputs.
    Permuting candidates changes this alignment hash but permutes rows/paths.
    """
    prompts = choice_prompts(inputs, schema)
    recipe = {"version": CHOICE_FEATURE_VERSION, "schema_hash": schema.hash,
              "structured_width": choice_feature_width(schema),
              "structured_layout": "state_categories_then_visibility;group_and_STOP_bits;raw_remaining_groups,lambda,incremental_cost",
              "numeric_dtype": "float32", "prompt_sha256": stable_hash(prompts)}
    return {**recipe, "sha256": stable_hash(recipe)}

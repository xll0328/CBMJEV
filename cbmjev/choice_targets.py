"""Explicit one-step Choice imitation targets, not calibrated action risks.

Use only DecisionSets returned by the provenance-validating reader. The teacher
uses offline realized error; its expected softmax is not the softmax of expected
error. Neither its probabilities nor its winners are Bayes-optimality labels.
"""
from dataclasses import dataclass
import math

from .contracts import stable_hash
from .decision_sets import DecisionSet


@dataclass(frozen=True)
class ChoiceInputs:
    observed: tuple
    actions: tuple
    remaining_groups: int
    incremental_costs: tuple
    cost_weight: float


@dataclass(frozen=True)
class ChoiceSupervision:
    realized_errors: tuple
    teacher_probabilities: tuple
    temperature: float


@dataclass(frozen=True)
class ChoiceExample:
    model_inputs: ChoiceInputs
    supervision: ChoiceSupervision
    derived_id: str
    source: object


def _nonnegative(value, name, *, positive=False):
    if (type(value) not in (float, int) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(name + " must be finite and " + ("positive" if positive else "nonnegative"))
    return float(value)


def singleton_choice_example(decision, schema, *, remaining_groups,
                             cost_weight, temperature, group_costs=None):
    """Filter a verified complete set to STOP and every feasible singleton.

    Costs are public additive query costs, not measured encoder/latency costs.
    Temperature affects training supervision only. Cost weight and budget enter
    model inputs; runtime must not add this cost penalty again to Choice logits.
    Original occurrence multiplicity is preserved in the derived identity.
    """
    if not isinstance(decision, DecisionSet):
        raise ValueError("a verified DecisionSet is required")
    if type(remaining_groups) is not int or not 0 <= remaining_groups <= schema.num_groups:
        raise ValueError("remaining_groups must be an integer within the schema")
    weight = _nonnegative(cost_weight, "cost_weight")
    tau = _nonnegative(temperature, "temperature", positive=True)
    costs = tuple(1.0 for _ in schema.groups) if group_costs is None else tuple(group_costs)
    if len(costs) != schema.num_groups:
        raise ValueError("group_costs must cover every group")
    costs = tuple(_nonnegative(c, "group cost") for c in costs)
    observed = decision.model_inputs.observed
    schema.validate_state(observed)
    actions = decision.model_inputs.actions
    errors = decision.supervision.risk_targets
    if len(actions) != len(errors) or len(set(actions)) != len(actions):
        raise ValueError("candidate/target alignment or uniqueness mismatch")
    if any(type(e) not in (int, float) or e not in (0, 1) for e in errors):
        raise ValueError("Choice teacher requires realized Boolean errors, not arbitrary risks")
    lookup = dict(zip(actions, errors))
    # Check singleton completeness even when a zero budget will retain only STOP.
    required = ((),) + tuple((g,) for g, group in enumerate(schema.groups)
                             if observed[group.atoms[0]] < 0)
    if any(a not in lookup for a in required):
        raise ValueError("source does not contain every required singleton and STOP")
    selected = required if remaining_groups else ((),)
    selected_errors = tuple(float(lookup[a]) for a in selected)
    incremental = tuple(0.0 if not a else costs[a[0]] for a in selected)
    utilities = tuple(e + weight * c for e, c in zip(selected_errors, incremental))
    if not all(math.isfinite(u) for u in utilities):
        raise ValueError("nonfinite error-plus-cost teacher utility")
    minimum = min(utilities)
    # Subtract before division: avoid overflow of a common utility offset.
    unnormalized = tuple(math.exp(-(u - minimum) / tau) for u in utilities)
    total = sum(unnormalized)
    probabilities = tuple(v / total for v in unnormalized)
    inputs = ChoiceInputs(observed, selected, remaining_groups, incremental, weight)
    supervision = ChoiceSupervision(selected_errors, probabilities, tau)
    identity = dict(format="cbmjev-singleton-choice-imitation-v1",
        source_package=decision.metadata.source_package_sha256,
        source_artifact=decision.metadata.source_artifact_id,
        occurrence=decision.metadata.occurrence, schema=schema.hash,
        observed=observed, actions=selected, remaining_groups=remaining_groups,
        incremental_costs=incremental, cost_weight=weight, temperature=tau,
        realized_errors=selected_errors, teacher_probabilities=probabilities)
    return ChoiceExample(inputs, supervision, "choice-example:" + stable_hash(identity),
                         decision.metadata)

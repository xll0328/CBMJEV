"""Hard-concept, one-step verification with no hidden-answer runtime inputs.

Source predictions and labels use zero-based semantic category IDs. ``None``
means an unrevealed B answer or a missing training/evaluation annotation; it is
never a semantic category. Full answer caches belong to the caller's offline
environment, not to observations, fusion, or action selection.

This module contains NumPy decision/measurement primitives, not a trained task
head or selector. The caller must enforce fit/tune/confirmation role separation
and freeze fitted components before evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional, Tuple

import numpy as np


STOP = "STOP"
FUSION_METHODS = ("a_only", "reliability", "independent", "joint")


def _category(value, size, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer semantic category")
    if not 0 <= value < size:
        raise ValueError(f"{name} must be in [0, {size})")
    return int(value)


def _category_counts(values):
    counts = tuple(values)
    if not counts or any(isinstance(v, (bool, np.bool_)) or
                         not isinstance(v, Integral) or v < 2 for v in counts):
        raise ValueError("category_counts must contain integers >= 2")
    return tuple(int(v) for v in counts)


def _nonnegative_number(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or
            not np.isfinite(value) or value < 0 or (positive and value == 0)):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return float(value)


@dataclass(frozen=True)
class VerificationObservation:
    """An immutable visible state; no text, labels, IDs, or hidden B cache.

    ``observed_b`` defaults to all ``None``. Multiple revealed coordinates can
    represent the full-fusion reference, but the deployment action menu below
    permits only one query followed by STOP.
    """

    initial_a: Tuple[int, ...]
    category_counts: Tuple[int, ...]
    observed_b: Optional[Tuple[Optional[int], ...]] = None

    def __post_init__(self):
        counts = _category_counts(self.category_counts)
        initial = tuple(self.initial_a)
        observed = ((None,) * len(counts) if self.observed_b is None
                    else tuple(self.observed_b))
        if len(initial) != len(counts) or len(observed) != len(counts):
            raise ValueError("observation width must match category_counts")
        initial = tuple(_category(v, c, "initial_a") for v, c in zip(initial, counts))
        observed = tuple(None if v is None else _category(v, c, "observed_b")
                         for v, c in zip(observed, counts))
        object.__setattr__(self, "category_counts", counts)
        object.__setattr__(self, "initial_a", initial)
        object.__setattr__(self, "observed_b", observed)


def observe_b(state, j, answer):
    """Reveal exactly one typed answer, returning a new state; reject repeats."""
    j = _category(j, len(state.category_counts), "concept index")
    if state.observed_b[j] is not None:
        raise ValueError("a concept cannot be verified twice")
    answer = _category(answer, state.category_counts[j], "B answer")
    observed = list(state.observed_b)
    observed[j] = answer
    return VerificationObservation(state.initial_a, state.category_counts, tuple(observed))


def legal_actions(state):
    """The single-decision deployment menu, ordered STOP then concept index."""
    if any(value is not None for value in state.observed_b):
        return (STOP,)
    return (STOP,) + tuple(range(len(state.category_counts)))


def one_hot_initial(state):
    """Encode only initial hard A categories for a one-step selector."""
    return np.concatenate([np.eye(size, dtype=np.float64)[value]
                           for size, value in zip(state.category_counts, state.initial_a)])


def _training_matrix(values, counts, name, *, missing=False):
    array = np.asarray(values, dtype=object)
    if array.ndim != 2 or array.shape[1] != len(counts) or array.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty (samples, concepts) matrix")
    for row in array:
        for value, size in zip(row, counts):
            if not (missing and value is None):
                _category(value, size, name)
    return array


@dataclass(frozen=True)
class _Confusion:
    prior: np.ndarray
    a: np.ndarray
    b: np.ndarray
    joint: np.ndarray
    preferred: str
    labeled: int
    paired: int


class ConfusionFusion:
    """Per-concept, concept-supervised smoothed confusion fusion.

    All variants use the same P(C|A) at unqueried positions. ``a_only`` also
    retains it after a reveal. ``reliability`` uses the confusion posterior of
    the more accurate source on paired labeled fit examples (A wins ties).
    ``independent`` multiplies P(A|C) and P(B|C); ``joint`` uses P(A,B|C).
    ``alpha`` is total uniform pseudo-count mass for each categorical table.

    Thus A-only is calibrated A, which can differ from raw ``initial_a``.
    Fitting accepts concept labels only. Task labels cannot be passed to fit.
    """

    def __init__(self, category_counts, *, method="joint", alpha=1.0):
        self.category_counts = _category_counts(category_counts)
        if method not in FUSION_METHODS:
            raise ValueError(f"method must be one of {FUSION_METHODS}")
        self.method = method
        self.alpha = _nonnegative_number(alpha, "alpha", positive=True)
        self._records = None

    def fit(self, initial_a, source_b, concept_labels):
        """Fit on legal concept-supervision roles; skip only explicit None C.

        B can be None in training when that answer is unavailable. Its table
        denominators and source reliability use only paired observations.
        Every concept needs at least one labeled example.
        """
        a = _training_matrix(initial_a, self.category_counts, "initial_a")
        b = _training_matrix(source_b, self.category_counts, "source_b", missing=True)
        gold = _training_matrix(concept_labels, self.category_counts,
                                "concept_labels", missing=True)
        if a.shape != b.shape or a.shape != gold.shape:
            raise ValueError("fit matrices must have identical shapes")
        records = []
        for j, size in enumerate(self.category_counts):
            prior = np.zeros(size, dtype=np.float64)
            ca, cb = np.zeros((2, size, size), dtype=np.float64)
            cab = np.zeros((size, size, size), dtype=np.float64)
            paired_counts = np.zeros(size, dtype=np.float64)
            correct_a = correct_b = 0
            for av, bv, c in zip(a[:, j], b[:, j], gold[:, j]):
                if c is None:
                    continue
                prior[c] += 1
                ca[c, av] += 1
                if bv is not None:
                    cb[c, bv] += 1
                    cab[c, av, bv] += 1
                    paired_counts[c] += 1
                    correct_a += int(av == c)
                    correct_b += int(bv == c)
            labeled = int(prior.sum())
            if not labeled:
                raise ValueError(f"no labeled concept examples for concept {j}")
            pa = (ca + self.alpha / size) / (prior[:, None] + self.alpha)
            pb = (cb + self.alpha / size) / (paired_counts[:, None] + self.alpha)
            pab = ((cab + self.alpha / (size * size)) /
                   (paired_counts[:, None, None] + self.alpha))
            p = (prior + self.alpha / size) / (labeled + self.alpha)
            for array in (p, pa, pb, pab):
                array.setflags(write=False)
            records.append(_Confusion(p, pa, pb, pab,
                                      "b" if correct_b > correct_a else "a",
                                      labeled, int(paired_counts.sum())))
        self._records = tuple(records)
        return self

    @property
    def labeled_counts(self):
        self._require_fitted()
        return tuple(record.labeled for record in self._records)

    @property
    def paired_counts(self):
        self._require_fitted()
        return tuple(record.paired for record in self._records)

    @property
    def preferred_sources(self):
        self._require_fitted()
        return tuple(record.preferred for record in self._records)

    def _require_fitted(self):
        if self._records is None:
            raise ValueError("fusion must be fitted before inference")

    def probabilities(self, state):
        """Return normalized concept posteriors from visible state alone."""
        self._require_fitted()
        if state.category_counts != self.category_counts:
            raise ValueError("state and fusion category_counts differ")
        result = []
        for a, b, record in zip(state.initial_a, state.observed_b, self._records):
            if b is None or self.method == "a_only":
                log_likelihood = np.log(record.a[:, a])
            elif self.method == "reliability":
                log_likelihood = np.log(record.b[:, b] if record.preferred == "b"
                                        else record.a[:, a])
            elif self.method == "independent":
                log_likelihood = np.log(record.a[:, a]) + np.log(record.b[:, b])
            else:
                log_likelihood = np.log(record.joint[:, a, b])
            log_p = np.log(record.prior) + log_likelihood
            weights = np.exp(log_p - log_p.max())
            result.append(weights / weights.sum())
        return tuple(result)

    def fuse(self, state):
        """The task head receives this tuple only; category ties favor low IDs."""
        return tuple(int(np.argmax(p)) for p in self.probabilities(state))


def _probability_array(values, name, *, ndim, normalized_axes):
    array = np.asarray(values, dtype=np.float64)
    if (array.ndim != ndim or any(size == 0 for size in array.shape) or
            not np.isfinite(array).all() or (array < 0).any()):
        raise ValueError(f"{name} must be a finite nonnegative probability array")
    if not np.allclose(array.sum(axis=normalized_axes), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError(f"{name} must be normalized")
    return array


def joint_from_conditionals(q_y, response_given_y):
    """Build each q_j(y,v|h) from one shared q_Y and r_j(v|y,h).

    q_y has shape (Y,), and response_given_y is a sequence of (Y, V_j)
    arrays. Inference integrates over every y; there is no true-label input.
    """
    q_y = _probability_array(q_y, "q_y", ndim=1, normalized_axes=0)
    joints = []
    for conditional in response_given_y:
        conditional = _probability_array(conditional, "response_given_y", ndim=2,
                                         normalized_axes=1)
        if conditional.shape[0] != q_y.size:
            raise ValueError("all actions must share the same task label space")
        joints.append(q_y[:, None] * conditional)
    if not joints:
        raise ValueError("at least one response conditional is required")
    return tuple(joints)


def product_of_marginals(joint):
    """Remove dependence from this same fitted q, preserving both marginals."""
    joint = _probability_array(joint, "joint", ndim=2, normalized_axes=(0, 1))
    return joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]


def expected_signed_gain(joint, before, after_by_answer):
    """Expected 0/1 loss reduction, possibly negative, summed over all Y,V."""
    joint = _probability_array(joint, "joint", ndim=2, normalized_axes=(0, 1))
    num_y, num_v = joint.shape
    before = _category(before, num_y, "before prediction")
    after = tuple(after_by_answer)
    if len(after) != num_v:
        raise ValueError("one after prediction is required per answer category")
    after = np.asarray([_category(v, num_y, "after prediction") for v in after])
    y = np.arange(num_y)[:, None]
    delta = (before != y).astype(np.float64) - (after[None, :] != y)
    return float(np.sum(joint * delta))


def expected_action_gains(joints, before, after_by_action, *, product_ablation=False):
    """Value every action against one before prediction and a shared Y marginal."""
    joints, after_by_action = tuple(joints), tuple(after_by_action)
    if not joints or len(joints) != len(after_by_action):
        raise ValueError("one joint and after-prediction table are required per action")
    checked = [_probability_array(q, "joint", ndim=2, normalized_axes=(0, 1))
               for q in joints]
    q_y = checked[0].sum(axis=1)
    for joint in checked[1:]:
        if joint.shape[0] != q_y.size or not np.allclose(
                joint.sum(axis=1), q_y, rtol=1e-7, atol=1e-9):
            raise ValueError("action joints must have the same Y marginal")
    return np.asarray([expected_signed_gain(product_of_marginals(q) if product_ablation
                                            else q, before, after)
                       for q, after in zip(checked, after_by_action)])


def singleton_predictions(state, fusion, predict):
    """Enumerate hypothetical typed replies; never access a sample's hidden B.

    ``predict`` accepts only the fused hard tuple and returns an integer task
    category. STOP is evaluated once. Each candidate starts from the same
    initial state, so no action can acquire a different before prediction.
    """
    if any(value is not None for value in state.observed_b):
        raise ValueError("singleton prediction tables require an initial state")
    before = predict(fusion.fuse(state))
    after = tuple(tuple(predict(fusion.fuse(observe_b(state, j, answer)))
                        for answer in range(size))
                  for j, size in enumerate(state.category_counts))
    for value in (before,) + tuple(v for row in after for v in row):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
            raise ValueError("task predict must return a nonnegative integer category")
    return int(before), tuple(tuple(int(v) for v in row) for row in after)


def choose_action(state, gains, costs, *, cost_weight=0.0):
    """Maximize signed gain minus incremental cost; STOP wins all zero ties.

    Tied positive singletons favor the lowest concept index. Costs are declared
    incremental costs relative to STOP, never hidden-answer-dependent telemetry.
    """
    gains, costs = np.asarray(gains, dtype=np.float64), np.asarray(costs, dtype=np.float64)
    expected_shape = (len(state.category_counts),)
    if gains.shape != expected_shape or costs.shape != expected_shape:
        raise ValueError("gains and costs must have one entry per concept")
    if not np.isfinite(gains).all() or not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("gains must be finite and costs finite nonnegative")
    cost_weight = _nonnegative_number(cost_weight, "cost_weight")
    best_action, best_value = STOP, 0.0
    for action in legal_actions(state)[1:]:
        net_value = gains[action] - cost_weight * costs[action]
        if net_value > best_value:
            best_action, best_value = action, net_value
    return best_action


def _label_array(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "iu" or (array < 0).any():
        raise ValueError(f"{name} must contain nonnegative integer task categories")
    return array


def signed_gains(before, after, gold):
    """Evaluation/training-only signed 0/1 targets; NumPy broadcasting applies."""
    before, after, gold = (_label_array(v, name) for v, name in
                           ((before, "before"), (after, "after"), (gold, "gold")))
    before, after, gold = np.broadcast_arrays(before, after, gold)
    return (before != gold).astype(np.int8) - (after != gold).astype(np.int8)


def repair_damage_metrics(before, after, gold, *, concept_before=None,
                          concept_after=None, concept_gold=None):
    """Evaluation-only event counts, with explicit semantic missingness.

    Task arrays are (N,). Optional concept arrays have the same shape and refer
    to the selected coordinate for each example, not an all-concept average.
    A None gold concept is counted as missing, never correct or unchanged.
    ``unchanged`` means correctness did not change; wrong-to-different-wrong
    substitutions are counted separately in ``concept_changed_still_wrong``.
    """
    before, after, gold = (_label_array(v, name) for v, name in
                           ((before, "before"), (after, "after"), (gold, "gold")))
    if before.ndim != 1 or before.shape != after.shape or before.shape != gold.shape:
        raise ValueError("task metric arrays must have the same one-dimensional shape")
    gains = signed_gains(before, after, gold)
    samples = len(gains)
    counts = {"repair": int((gains == 1).sum()), "damage": int((gains == -1).sum()),
              "unchanged": int((gains == 0).sum())}
    result = {"samples": samples, "task_counts": counts,
              "task_repair_rate": counts["repair"] / samples if samples else None,
              "task_damage_rate": counts["damage"] / samples if samples else None,
              "mean_signed_gain": float(gains.mean()) if samples else None}
    supplied = (concept_before is not None, concept_after is not None, concept_gold is not None)
    if not any(supplied):
        return result
    if not all(supplied):
        raise ValueError("all three concept arrays are required for semantic metrics")
    cb = _label_array(concept_before, "concept_before")
    ca = _label_array(concept_after, "concept_after")
    cg = np.asarray(concept_gold, dtype=object)
    if cb.shape != before.shape or ca.shape != before.shape or cg.shape != before.shape:
        raise ValueError("concept metric arrays must match task metric shape")
    concept_counts = dict.fromkeys(("repair", "damage", "unchanged", "missing"), 0)
    cross = {task: dict.fromkeys(concept_counts, 0) for task in counts}
    changed_still_wrong = 0
    for delta, old, new, label in zip(gains, cb, ca, cg):
        task_event = "repair" if delta == 1 else "damage" if delta == -1 else "unchanged"
        if label is None:
            concept_event = "missing"
        else:
            if isinstance(label, (bool, np.bool_)) or not isinstance(label, Integral) or label < 0:
                raise ValueError("concept_gold must contain integer categories or None")
            concept_event = ("repair" if old != label and new == label else
                             "damage" if old == label and new != label else "unchanged")
            changed_still_wrong += int(old != label and new != label and old != new)
        concept_counts[concept_event] += 1
        cross[task_event][concept_event] += 1
    labeled = samples - concept_counts["missing"]
    result.update({"concept_labeled": labeled, "concept_missing": concept_counts["missing"],
                   "concept_counts": concept_counts,
                   "concept_repair_rate": concept_counts["repair"] / labeled if labeled else None,
                   "concept_damage_rate": concept_counts["damage"] / labeled if labeled else None,
                   "concept_changed_still_wrong": changed_still_wrong,
                   "task_by_concept": cross})
    return result

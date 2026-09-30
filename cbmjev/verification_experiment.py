"""Offline verification preparation with group-disjoint fusion supervision.

This module owns labeled caches, never selector inference. Task-head training
uses a fixed mixture of A-only/all-fused/singleton states. Each OOF fusion model
selects its recipe by concept NLL inside its own training groups. Task-head tune
groups cannot influence those fits or recipe choices. The final runtime recipe
is frozen before task-head tuning; only its confusion counts are then refitted.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral

import numpy as np

from .verification import (
    STOP, FUSION_METHODS, ConfusionFusion, VerificationObservation,
    _category_counts, _label_array, _training_matrix, observe_b,
    repair_damage_metrics, signed_gains,
)


SPLIT_SEED = 20260930
FUSION_GRID = tuple((method, alpha) for method in FUSION_METHODS for alpha in (0.1, 1.0, 10.0))
FUSION_OBJECTIVE = "0.5 * mean_per_concept_nll(A-only) + 0.5 * mean_per_concept_nll(all-fused)"


def _readonly(values, dtype=None):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _hard_matrix(values, categories, name):
    if not isinstance(values, np.ndarray):
        # Preserve Python booleans until validation; np.asarray([0, True])
        # would otherwise silently turn the boolean into category 1.
        _training_matrix(values, categories, name)
    array = np.asarray(values)
    if (array.ndim != 2 or array.shape[1] != len(categories) or
            array.dtype.kind not in "iu" or (array < 0).any() or
            (array >= np.asarray(categories)[None, :]).any()):
        raise ValueError(f"{name} must be a hard-category (samples, concepts) matrix")
    return array


def _ids(values, name, *, unique=False):
    result = tuple(values)
    if not result or any(not isinstance(v, str) or not v for v in result):
        raise ValueError(f"{name} must contain nonempty strings")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


@dataclass(frozen=True)
class VerificationData:
    """Offline cache for one caller-validated role, with fully observed task Y.

    A/B are complete hard predictions; C may contain None. IDs never enter a
    fitted predictor. Role provenance and text/author connected components must
    be verified by the caller before constructing this container.
    """

    sample_ids: tuple
    group_ids: tuple
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Y: np.ndarray
    categories: tuple

    def __post_init__(self):
        categories = _category_counts(self.categories)
        sample_ids = _ids(self.sample_ids, "sample_ids", unique=True)
        group_ids = _ids(self.group_ids, "group_ids")
        a, b = _hard_matrix(self.A, categories, "A"), _hard_matrix(self.B, categories, "B")
        c = _training_matrix(self.C, categories, "C", missing=True)
        if any(isinstance(value, (bool, np.bool_)) for value in np.asarray(self.Y, dtype=object).flat):
            raise ValueError("Y must contain integer categories, not booleans")
        y = _label_array(self.Y, "Y")
        n = len(sample_ids)
        if a.shape != (n, len(categories)) or b.shape != a.shape or c.shape != a.shape:
            raise ValueError("A, B, and C must match sample_ids and categories")
        if y.shape != (n,) or len(group_ids) != n:
            raise ValueError("Y and group_ids must match sample_ids")
        for name, value in (("sample_ids", sample_ids), ("group_ids", group_ids),
                            ("categories", categories), ("A", _readonly(a, np.int64)),
                            ("B", _readonly(b, np.int64)), ("C", _readonly(c, object)),
                            ("Y", _readonly(y, np.int64))):
            object.__setattr__(self, name, value)

    @property
    def category_counts(self):
        return self.categories

    def __len__(self):
        return len(self.sample_ids)

    def subset(self, indices):
        indices = np.asarray(indices)
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or
                not len(indices) or (indices < 0).any() or (indices >= len(self)).any()):
            raise ValueError("subset indices must be a nonempty valid integer vector")
        return VerificationData(tuple(self.sample_ids[i] for i in indices),
                                tuple(self.group_ids[i] for i in indices),
                                self.A[indices], self.B[indices], self.C[indices],
                                self.Y[indices], self.categories)


def _ordered_groups(group_ids, seed, purpose):
    group_ids = _ids(group_ids, "group_ids")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    def key(group):
        payload = json.dumps(["verification", purpose, int(seed), group], separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).digest(), group
    return group_ids, sorted(set(group_ids), key=key)


def group_holdout(group_ids, *, seed=SPLIT_SEED, tune_fraction=0.2):
    """Deterministic 80/20 group split; proportions refer to groups, not rows."""
    group_ids, groups = _ordered_groups(group_ids, seed, "holdout")
    if not 0 < tune_fraction < 1 or len(groups) < 2:
        raise ValueError("group holdout needs >= 2 groups and a fraction in (0, 1)")
    count = min(len(groups) - 1, max(1, int(np.ceil(len(groups) * tune_fraction))))
    tune_groups = set(groups[:count])
    tune = np.asarray([i for i, group in enumerate(group_ids) if group in tune_groups], dtype=np.int64)
    train = np.asarray([i for i, group in enumerate(group_ids) if group not in tune_groups], dtype=np.int64)
    return train, tune


def group_folds(group_ids, *, n_folds=5, seed=SPLIT_SEED, allow_small_groups=False):
    """Balanced-by-group hash order; optional smaller folds are synthetic-only."""
    group_ids, groups = _ordered_groups(group_ids, seed, "oof")
    if isinstance(n_folds, bool) or not isinstance(n_folds, Integral) or n_folds < 2:
        raise ValueError("n_folds must be an integer >= 2")
    if len(groups) < n_folds:
        if not allow_small_groups:
            raise ValueError(f"at least {n_folds} groups required; smaller folds are synthetic-only")
        n_folds = len(groups)
    if n_folds < 2:
        raise ValueError("OOF fusion needs at least two groups")
    assignments = {group: rank % n_folds for rank, group in enumerate(groups)}
    return np.asarray([assignments[group] for group in group_ids], dtype=np.int64)


class FusionLookup:
    """Finite per-concept tables computed through legal observation states.

    Lookup construction depends only on fitted U. Batched transformations never
    call fusion again, regardless of N. Table axes are A, B-status, semantic C;
    B-status 0 means unrevealed and 1+v means the observed answer v.
    """

    def __init__(self, fusion):
        self.categories = fusion.category_counts
        probabilities = []
        for j, size in enumerate(self.categories):
            table = np.empty((size, size + 1, size), dtype=np.float64)
            for a in range(size):
                initial = [0] * len(self.categories)
                initial[j] = a
                state = VerificationObservation(initial, self.categories)
                table[a, 0] = fusion.probabilities(state)[j]
                for b in range(size):
                    table[a, b + 1] = fusion.probabilities(observe_b(state, j, b))[j]
            probabilities.append(_readonly(table))
        self.probabilities = tuple(probabilities)
        self.hard = tuple(_readonly(table.argmax(axis=2), np.int64) for table in probabilities)

    def initial(self, initial_a):
        a = _hard_matrix(initial_a, self.categories, "initial_a")
        return np.column_stack([table[a[:, j], 0] for j, table in enumerate(self.hard)])

    def all_fused(self, initial_a, source_b):
        a = _hard_matrix(initial_a, self.categories, "initial_a")
        b = _hard_matrix(source_b, self.categories, "source_b")
        if a.shape != b.shape:
            raise ValueError("A and B shapes differ")
        return np.column_stack([table[a[:, j], b[:, j] + 1] for j, table in enumerate(self.hard)])

    def singleton_states(self, initial_a, source_b):
        before, full = self.initial(initial_a), self.all_fused(initial_a, source_b)
        k = len(self.categories)
        states = np.repeat(before[:, None, :], k, axis=1)
        for j in range(k):
            states[:, j, j] = full[:, j]
        return states

    def hypothetical_states(self, initial_a, j):
        """All candidate answers for concept j; has no actual B argument."""
        if isinstance(j, bool) or not isinstance(j, Integral) or not 0 <= j < len(self.categories):
            raise ValueError("invalid concept index")
        a = _hard_matrix(initial_a, self.categories, "initial_a")
        states = np.repeat(self.initial(a)[:, None, :], self.categories[j], axis=1)
        states[:, :, j] = self.hard[j][a[:, j], 1:]
        return states


def _concept_nll(data, lookup):
    per_concept, labeled = [], []
    for j, probabilities in enumerate(lookup.probabilities):
        rows = np.asarray([i for i, gold in enumerate(data.C[:, j]) if gold is not None], dtype=np.int64)
        labeled.append(len(rows))
        if not len(rows):
            per_concept.append(None)
            continue
        gold = np.asarray(data.C[rows, j], dtype=np.int64)
        p_a = probabilities[data.A[rows, j], 0, gold]
        p_ab = probabilities[data.A[rows, j], data.B[rows, j] + 1, gold]
        per_concept.append({"a_only": float(-np.log(p_a).mean()),
                            "all_fused": float(-np.log(p_ab).mean())})
    present = [value for value in per_concept if value is not None]
    if not present:
        raise ValueError("fusion tune split has no observed concept labels")
    score = float(np.mean([0.5 * (value["a_only"] + value["all_fused"]) for value in present]))
    return score, per_concept, labeled


@dataclass(frozen=True)
class FusionSelection:
    method: str
    alpha: float
    model: ConfusionFusion
    log: dict


def _fit_fusion(data, method, alpha):
    # Deliberately enumerate concept arrays: data.Y is never an argument.
    return ConfusionFusion(data.categories, method=method, alpha=alpha).fit(data.A, data.B, data.C)


def select_fusion(data, *, seed=SPLIT_SEED):
    """Select the fixed 12-item recipe grid using concept supervision only.

    The objective averages A-only/all-fused NLL equally, then concepts equally
    among concepts with held-out labels. Missing concept counts are explicit.
    Exact ties follow FUSION_GRID order. The returned model refits the selected
    recipe to all provided data, so callers must provide training groups only.
    """
    fit_indices, tune_indices = group_holdout(data.group_ids, seed=seed)
    fit, tune = data.subset(fit_indices), data.subset(tune_indices)
    candidates = []
    for method, alpha in FUSION_GRID:
        model = _fit_fusion(fit, method, alpha)
        score, per_concept, labeled = _concept_nll(tune, FusionLookup(model))
        candidates.append({"method": method, "alpha": alpha, "concept_nll": score,
                           "per_concept_nll": per_concept, "labeled_per_concept": labeled})
    best = min(range(len(candidates)), key=lambda i: candidates[i]["concept_nll"])
    method, alpha = FUSION_GRID[best]
    log = {"objective": FUSION_OBJECTIVE, "seed": int(seed), "grid": candidates,
           "selected_method": method, "selected_alpha": alpha,
           "fit_samples": len(fit), "tune_samples": len(tune),
           "fit_group_ids": sorted(set(fit.group_ids)), "tune_group_ids": sorted(set(tune.group_ids)),
           "supervision": "concept labels only; task Y not consulted"}
    return FusionSelection(method, alpha, _fit_fusion(data, method, alpha), log)


@dataclass(frozen=True)
class MixedStates:
    """Row-major f inputs; kinds 0=A-only, 1=all-B, 2+j=singleton j.

    All K singleton states are included exactly. Each source row has total
    weight one: 1/3 A-only, 1/3 all-B, and 1/(3K) per singleton. Metadata and
    kinds are for auditing only, never inputs to the common task head.
    """

    concepts: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    row_indices: np.ndarray
    kinds: np.ndarray


def _package_mixture(data, concepts):
    n, k = len(data), len(data.categories)
    if concepts.shape != (n, k + 2, k):
        raise ValueError("mixture concepts have wrong shape")
    weights = np.asarray([1 / 3, 1 / 3] + [1 / (3 * k)] * k)
    return MixedStates(_readonly(concepts.reshape(-1, k), np.int64),
                       _readonly(np.repeat(data.Y, k + 2), np.int64),
                       _readonly(np.tile(weights, n)),
                       _readonly(np.repeat(np.arange(n), k + 2), np.int64),
                       _readonly(np.tile(np.arange(k + 2), n), np.int64))


def mixed_states(data, lookup):
    if data.categories != lookup.categories:
        raise ValueError("data and fusion categories differ")
    return _package_mixture(data, np.concatenate((lookup.initial(data.A)[:, None, :],
                                                 lookup.all_fused(data.A, data.B)[:, None, :],
                                                 lookup.singleton_states(data.A, data.B)), axis=1))


def crossfit_states(data, *, n_folds=5, seed=SPLIT_SEED, allow_small_groups=False):
    """OOF states with recipe selection nested wholly inside each training fold."""
    folds = group_folds(data.group_ids, n_folds=n_folds, seed=seed,
                        allow_small_groups=allow_small_groups)
    n, k = len(data), len(data.categories)
    concepts = np.empty((n, k + 2, k), dtype=np.int64)
    records = []
    for fold in sorted(set(folds.tolist())):
        fit_indices, target_indices = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit, target = data.subset(fit_indices), data.subset(target_indices)
        if len(set(fit.group_ids)) < 2:
            raise ValueError("nested concept-only selection needs >= 2 training groups in every fold")
        selected = select_fusion(fit, seed=seed)
        mixed = mixed_states(target, FusionLookup(selected.model))
        concepts[target_indices] = mixed.concepts.reshape(len(target), k + 2, k)
        records.append({"fold": fold, "fit_group_ids": sorted(set(fit.group_ids)),
                        "target_group_ids": sorted(set(target.group_ids)),
                        "fit_samples": len(fit), "target_samples": len(target),
                        "selection": selected.log})
    log = {"seed": int(seed), "requested_folds": int(n_folds), "actual_folds": len(records),
           "synthetic_small_group_fallback": len(records) != n_folds,
           "folds": records}
    return _package_mixture(data, concepts), log


@dataclass(frozen=True)
class HeadPreparation:
    train: MixedStates
    tune: MixedStates
    selection: FusionSelection
    train_indices: np.ndarray
    tune_indices: np.ndarray
    log: dict


def prepare_head_training(data, *, n_folds=5, seed=SPLIT_SEED, allow_small_groups=False):
    """Reserve task-head tune groups before any fusion fit or recipe selection."""
    train_indices, tune_indices = group_holdout(data.group_ids, seed=seed)
    train, tune = data.subset(train_indices), data.subset(tune_indices)
    selected = select_fusion(train, seed=seed)
    train_states, folds = crossfit_states(train, n_folds=n_folds, seed=seed,
                                          allow_small_groups=allow_small_groups)
    tune_states = mixed_states(tune, FusionLookup(selected.model))
    log = {"task_head_train_group_ids": sorted(set(train.group_ids)),
           "task_head_tune_group_ids": sorted(set(tune.group_ids)),
           "runtime_recipe_selection": selected.log, "train_fusion_oof": folds,
           "tune_fusion_fit": "task-head training groups only",
           "state_weights": {"a_only": 1 / 3, "all_fused": 1 / 3, "singletons_total": 1 / 3}}
    return HeadPreparation(train_states, tune_states, selected,
                           _readonly(train_indices), _readonly(tune_indices), log)


def prepare_final_refit(data, method, alpha, *, n_folds=5, seed=SPLIT_SEED,
                        allow_small_groups=False):
    """After f selection, generate all-head-fit OOF states and refit frozen U recipe.

    This does not train f or choose new f hyperparameters/epochs. The caller must
    refit its selected task-head configuration on the returned mixture.
    """
    final_fusion = _fit_fusion(data, method, alpha)
    states, log = crossfit_states(data, n_folds=n_folds, seed=seed,
                                  allow_small_groups=allow_small_groups)
    log["runtime_method"], log["runtime_alpha"] = method, float(alpha)
    log["runtime_recipe_reselected"] = False
    return final_fusion, states, log


def _head_probabilities(head, concepts, *, batch_size=4096):
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    chunks, classes = [], None
    for start in range(0, len(concepts), batch_size):
        batch = concepts[start:start + batch_size]
        probabilities = np.asarray(head.predict_proba(batch), dtype=np.float64)
        if (probabilities.ndim != 2 or probabilities.shape[0] != len(batch) or
                probabilities.shape[1] < 2 or not np.isfinite(probabilities).all() or
                (probabilities < 0).any() or
                not np.allclose(probabilities.sum(axis=1), 1, rtol=1e-5, atol=1e-7)):
            raise ValueError("task head must return normalized finite class probabilities")
        if classes is not None and probabilities.shape[1] != classes:
            raise ValueError("task head changed its class count across batches")
        classes = probabilities.shape[1]
        # Float32 softmax may sum to 1 +/- a few ulps. Validate first, then
        # normalize in float64 so downstream expectations use proper mass.
        chunks.append(probabilities / probabilities.sum(axis=1, keepdims=True))
    if not chunks:
        raise ValueError("task prediction requires at least one sample")
    return np.concatenate(chunks)


@dataclass(frozen=True)
class ActionTable:
    """Offline actions: column 0 STOP; 1+j singleton j; final column all-B.

    `gains` contains only the K singleton signed task targets. This table owns
    realized answers/outcomes and must never be passed as a selector input.
    """

    sample_ids: tuple
    concepts: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    gains: np.ndarray


def evaluate_actions(data, lookup, task_head, *, batch_size=4096):
    if data.categories != lookup.categories:
        raise ValueError("data and fusion categories differ")
    n, k = len(data), len(data.categories)
    concepts = np.concatenate((lookup.initial(data.A)[:, None, :],
                               lookup.singleton_states(data.A, data.B),
                               lookup.all_fused(data.A, data.B)[:, None, :]), axis=1)
    probabilities = _head_probabilities(task_head, concepts.reshape(-1, k), batch_size=batch_size)
    if (data.Y >= probabilities.shape[1]).any():
        raise ValueError("task label exceeds task-head class count")
    probabilities = probabilities.reshape(n, k + 2, -1)
    predictions = probabilities.argmax(axis=2)
    gains = signed_gains(predictions[:, :1], predictions[:, 1:k + 1], data.Y[:, None])
    return ActionTable(data.sample_ids, _readonly(concepts), _readonly(probabilities),
                       _readonly(predictions), _readonly(gains))


def hypothetical_predictions(initial_a, lookup, task_head, *, batch_size=4096):
    """Per-action (N,V_j) predictions for q value integration; initial A only."""
    a = _hard_matrix(initial_a, lookup.categories, "initial_a")
    result = []
    for j, size in enumerate(lookup.categories):
        states = lookup.hypothetical_states(a, j)
        p = _head_probabilities(task_head, states.reshape(-1, len(lookup.categories)),
                                batch_size=batch_size)
        result.append(p.argmax(axis=1).reshape(len(a), size))
    return tuple(result)


def selected_event_metrics(data, table, actions):
    """Task metrics for all rows, semantic events only for selected concepts."""
    actions = tuple(actions)
    if table.sample_ids != data.sample_ids or len(actions) != len(data):
        raise ValueError("action table and actions must align with data sample_ids")
    k = len(data.categories)
    columns, queried_rows, concepts = [], [], []
    for i, action in enumerate(actions):
        if isinstance(action, str) and action == STOP:
            columns.append(0)
        elif (not isinstance(action, (bool, np.bool_)) and isinstance(action, Integral)
              and 0 <= action < k):
            columns.append(1 + int(action))
            queried_rows.append(i)
            concepts.append(int(action))
        else:
            raise ValueError("actions must be STOP or valid singleton concept indices")
    columns = np.asarray(columns, dtype=np.int64)
    before = table.predictions[:, 0]
    after = table.predictions[np.arange(len(data)), columns]
    result = repair_damage_metrics(before, after, data.Y)
    rows, concept_indices = np.asarray(queried_rows, dtype=np.int64), np.asarray(concepts, dtype=np.int64)
    selected = repair_damage_metrics(
        before[rows], after[rows], data.Y[rows],
        concept_before=table.concepts[rows, 0, concept_indices],
        concept_after=table.concepts[rows, columns[rows], concept_indices],
        concept_gold=data.C[rows, concept_indices])
    result.update({"queried_samples": len(rows), "stopped_samples": len(data) - len(rows),
                   "query_fraction": len(rows) / len(data), "selected_concept_events": selected})
    return result

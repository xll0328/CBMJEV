"""Synthetic-only tests for offline fusion cross-fitting and action replay."""
import inspect
import unittest
from unittest.mock import patch

import numpy as np

from cbmjev.verification import STOP, ConfusionFusion, VerificationObservation, observe_b
from cbmjev.verification_experiment import (
    FUSION_GRID, FUSION_OBJECTIVE, FusionLookup, VerificationData, crossfit_states,
    evaluate_actions, group_folds, group_holdout, hypothetical_predictions,
    mixed_states, prepare_final_refit, prepare_head_training, select_fusion,
    selected_event_metrics,
)


def make_data(groups=20, categories=(2, 3), seed=42):
    rng = np.random.default_rng(seed)
    n = 2 * groups
    gold = np.column_stack([rng.integers(size, size=n) for size in categories])
    a, b = gold.copy(), gold.copy()
    for j, size in enumerate(categories):
        a[::3, j] = (a[::3, j] + 1) % size
        b[::5, j] = (b[::5, j] + 1) % size
    return VerificationData(tuple(f"s{i}" for i in range(n)),
                            tuple(f"g{i // 2}" for i in range(n)),
                            a, b, gold.astype(object), (gold[:, 0] + gold[:, -1]) % 2,
                            categories)


class ParityHead:
    def __init__(self):
        self.calls = []

    def predict_proba(self, concepts):
        assert concepts.ndim == 2 and concepts.dtype.kind in "iu"
        self.calls.append(concepts.copy())
        labels = concepts.sum(axis=1) % 2
        return np.eye(2)[labels] * 0.8 + 0.1


class DataAndSplitTests(unittest.TestCase):
    def test_data_validates_copies_and_preserves_explicit_missingness(self):
        a, b, c = [[0, 4], [1, 2]], [[1, 4], [0, 2]], [[None, 4], [1, None]]
        data = VerificationData(["a", "b"], ["g1", "g2"], a, b, c, [0, 1], [2, 5])
        a[0][0], c[0][1] = 1, 0
        self.assertEqual(data.A[0, 0], 0)
        self.assertEqual(data.C[0].tolist(), [None, 4])
        self.assertEqual(data.category_counts, (2, 5))
        self.assertEqual(data.subset([1]).sample_ids, ("b",))
        with self.assertRaises(ValueError):
            data.A[0, 0] = 1
        for indices in ([], [2], [0, 0], [0.0], [True]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                data.subset(indices)

    def test_malformed_categories_labels_and_ids_rejected(self):
        base = dict(sample_ids=["a", "b"], group_ids=["g1", "g2"],
                    A=[[0], [1]], B=[[1], [0]], C=[[None], [1]], Y=[0, 1], categories=(2,))
        for change in ({"sample_ids": ["a", "a"]}, {"group_ids": ["g1"]},
                       {"A": [[0.0], [1.0]]}, {"B": [[None], [0]]},
                       {"A": [[0], [True]]}, {"B": [[False], [0]]},
                       {"C": [[-1], [1]]}, {"C": [[False], [1]]},
                       {"Y": [0, None]}, {"Y": [0, True]}, {"categories": (3, 5)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                VerificationData(**dict(base, **change))

    def test_group_splits_are_deterministic_and_row_order_invariant(self):
        data = make_data(groups=11)
        train, tune = group_holdout(data.group_ids)
        train_groups, tune_groups = ({data.group_ids[i] for i in index} for index in (train, tune))
        self.assertFalse(train_groups & tune_groups)
        self.assertEqual(len(tune_groups), 3)
        self.assertEqual(set(train) | set(tune), set(range(len(data))))
        folds = group_folds(data.group_ids)
        mapping = dict(zip(data.group_ids, folds))
        self.assertEqual(set(folds), set(range(5)))
        for group in set(data.group_ids):
            self.assertEqual(len({folds[i] for i, value in enumerate(data.group_ids) if value == group}), 1)
        perm = np.random.default_rng(40).permutation(len(data))
        reordered = tuple(data.group_ids[i] for i in perm)
        self.assertEqual(dict(zip(reordered, group_folds(reordered))), mapping)
        _, tune_reordered = group_holdout(reordered)
        self.assertEqual({reordered[i] for i in tune_reordered}, tune_groups)
        with self.assertRaisesRegex(ValueError, "synthetic-only"):
            group_folds(("a", "b", "c"))
        self.assertEqual(set(group_folds(("a", "b", "c"), allow_small_groups=True)), {0, 1, 2})


class LookupAndMixtureTests(unittest.TestCase):
    def test_lookup_matches_legal_state_fusion_for_mixed_cardinality(self):
        data = make_data(categories=(2, 3, 5))
        fusion = ConfusionFusion(data.categories, method="joint").fit(data.A, data.B, data.C)
        lookup = FusionLookup(fusion)
        before, full, singletons = lookup.initial(data.A), lookup.all_fused(data.A, data.B), lookup.singleton_states(data.A, data.B)
        for i in range(4):
            state = VerificationObservation(data.A[i], data.categories)
            self.assertEqual(tuple(before[i]), fusion.fuse(state))
            all_state = state
            for j, size in enumerate(data.categories):
                self.assertEqual(tuple(singletons[i, j]), fusion.fuse(observe_b(state, j, data.B[i, j])))
                hypothetical = lookup.hypothetical_states(data.A[i:i + 1], j)
                for answer in range(size):
                    self.assertEqual(tuple(hypothetical[0, answer]), fusion.fuse(observe_b(state, j, answer)))
                all_state = observe_b(all_state, j, data.B[i, j])
            self.assertEqual(tuple(full[i]), fusion.fuse(all_state))
        # Once compiled, batched calls must not invoke fusion per sample.
        with patch.object(fusion, "probabilities", side_effect=AssertionError("unexpected fusion call")):
            lookup.initial(data.A)
            lookup.all_fused(data.A, data.B)
            lookup.singleton_states(data.A, data.B)
            lookup.hypothetical_states(data.A, 0)

    def test_21_by_5_lookup_and_exact_mixture_weights(self):
        data = make_data(groups=3, categories=(5,) * 21)
        fusion = ConfusionFusion(data.categories).fit(data.A, data.B, data.C)
        lookup = FusionLookup(fusion)
        mixed = mixed_states(data, lookup)
        n, k = len(data), 21
        self.assertEqual(mixed.concepts.shape, (n * (k + 2), k))
        self.assertEqual(lookup.hypothetical_states(data.A, 20).shape, (n, 5, k))
        weights = mixed.weights.reshape(n, k + 2)
        np.testing.assert_allclose(weights[:, 0], 1 / 3)
        np.testing.assert_allclose(weights[:, 1], 1 / 3)
        np.testing.assert_allclose(weights[:, 2:], 1 / 63)
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        np.testing.assert_array_equal(mixed.labels, np.repeat(data.Y, k + 2))
        np.testing.assert_array_equal(mixed.row_indices, np.repeat(np.arange(n), k + 2))
        np.testing.assert_array_equal(mixed.kinds[:k + 2], np.arange(k + 2))
        features = mixed.concepts.reshape(n, k + 2, k)
        for j in range(k):
            others = [c for c in range(k) if c != j]
            np.testing.assert_array_equal(features[:, 2 + j, others], features[:, 0, others])


class FusionPreparationTests(unittest.TestCase):
    def test_selection_uses_fixed_objective_and_no_task_labels(self):
        data = make_data(groups=12)
        swapped = VerificationData(data.sample_ids, data.group_ids, data.A, data.B, data.C,
                                   1 - data.Y, data.categories)
        selected, other = select_fusion(data), select_fusion(swapped)
        self.assertEqual(selected.log, other.log)
        self.assertEqual(len(selected.log["grid"]), 12)
        self.assertEqual(selected.log["objective"], FUSION_OBJECTIVE)
        self.assertEqual([(row["method"], row["alpha"]) for row in selected.log["grid"]], list(FUSION_GRID))
        self.assertFalse(set(selected.log["fit_group_ids"]) & set(selected.log["tune_group_ids"]))
        expected = min(selected.log["grid"], key=lambda row: row["concept_nll"])
        self.assertEqual((selected.method, selected.alpha), (expected["method"], expected["alpha"]))
        for row in selected.log["grid"]:
            scores = [0.5 * (c["a_only"] + c["all_fused"]) for c in row["per_concept_nll"] if c]
            self.assertAlmostEqual(row["concept_nll"], np.mean(scores))

    def test_every_fold_excludes_target_groups_from_fit_and_recipe_selection(self):
        data = make_data(groups=15)
        states, log = crossfit_states(data)
        self.assertEqual(log["actual_folds"], 5)
        self.assertFalse(log["synthetic_small_group_fallback"])
        seen = set()
        for record in log["folds"]:
            targets, fit = set(record["target_group_ids"]), set(record["fit_group_ids"])
            inner_fit = set(record["selection"]["fit_group_ids"])
            inner_tune = set(record["selection"]["tune_group_ids"])
            self.assertFalse(targets & fit)
            self.assertFalse(targets & (inner_fit | inner_tune))
            self.assertFalse(inner_fit & inner_tune)
            self.assertEqual(inner_fit | inner_tune, fit)
            self.assertFalse(seen & targets)
            seen |= targets
        self.assertEqual(seen, set(data.group_ids))
        self.assertEqual(states.concepts.shape, (len(data) * 4, 2))

    def test_target_concept_poisoning_cannot_change_its_oof_states(self):
        data = make_data(groups=12)
        folds = group_folds(data.group_ids)
        target_rows = np.flatnonzero(folds == 0)
        changed_c = data.C.copy()
        for j, size in enumerate(data.categories):
            changed_c[target_rows, j] = (changed_c[target_rows, j] + 1) % size
        poisoned = VerificationData(data.sample_ids, data.group_ids, data.A, data.B,
                                    changed_c, 1 - data.Y, data.categories)
        normal, normal_log = crossfit_states(data)
        altered, altered_log = crossfit_states(poisoned)
        np.testing.assert_array_equal(normal.concepts.reshape(len(data), 4, 2)[target_rows],
                                      altered.concepts.reshape(len(data), 4, 2)[target_rows])
        self.assertEqual(normal_log["folds"][0], altered_log["folds"][0])

    def test_task_head_tune_concepts_and_labels_cannot_influence_fusion(self):
        data = make_data(groups=15)
        prepared = prepare_head_training(data)
        changed_c, changed_y = data.C.copy(), data.Y.copy()
        for j, size in enumerate(data.categories):
            changed_c[prepared.tune_indices, j] = (changed_c[prepared.tune_indices, j] + 1) % size
        changed_y[prepared.tune_indices] = 1 - changed_y[prepared.tune_indices]
        poisoned = VerificationData(data.sample_ids, data.group_ids, data.A, data.B,
                                    changed_c, changed_y, data.categories)
        other = prepare_head_training(poisoned)
        self.assertEqual(prepared.log, other.log)
        np.testing.assert_array_equal(prepared.train.concepts, other.train.concepts)
        np.testing.assert_array_equal(prepared.train.labels, other.train.labels)
        np.testing.assert_array_equal(prepared.tune.concepts, other.tune.concepts)
        self.assertTrue(np.any(prepared.tune.labels != other.tune.labels))
        tune_groups = set(prepared.log["task_head_tune_group_ids"])
        for fold in prepared.log["train_fusion_oof"]["folds"]:
            self.assertFalse(tune_groups & set(fold["fit_group_ids"]))
            self.assertFalse(tune_groups & set(fold["target_group_ids"]))

    def test_final_refit_keeps_frozen_runtime_recipe_and_nested_oof(self):
        data = make_data(groups=8)
        final_u, states, log = prepare_final_refit(data, "independent", 10.0)
        self.assertEqual((final_u.method, final_u.alpha), ("independent", 10.0))
        self.assertFalse(log["runtime_recipe_reselected"])
        self.assertEqual(final_u.labeled_counts, (len(data), len(data)))
        self.assertEqual(log["actual_folds"], 5)
        self.assertEqual(states.labels.shape, (len(data) * 4,))

    def test_small_fold_fallback_is_explicit_and_logged(self):
        data = make_data(groups=4)
        with self.assertRaisesRegex(ValueError, "synthetic-only"):
            crossfit_states(data)
        _, log = crossfit_states(data, allow_small_groups=True)
        self.assertEqual(log["actual_folds"], 4)
        self.assertTrue(log["synthetic_small_group_fallback"])
        with self.assertRaisesRegex(ValueError, "training groups"):
            crossfit_states(make_data(groups=2), allow_small_groups=True)


class ActionEvaluationTests(unittest.TestCase):
    def test_offline_action_table_matches_real_reveals_and_signed_gain(self):
        data = make_data(groups=4)
        fusion = ConfusionFusion(data.categories, method="joint", alpha=0.1).fit(data.A, data.B, data.C)
        lookup, head = FusionLookup(fusion), ParityHead()
        table = evaluate_actions(data, lookup, head, batch_size=3)
        self.assertEqual(table.predictions.shape, (len(data), 4))
        for i in range(len(data)):
            state = VerificationObservation(data.A[i], data.categories)
            self.assertEqual(tuple(table.concepts[i, 0]), fusion.fuse(state))
            for j in range(2):
                expected = fusion.fuse(observe_b(state, j, data.B[i, j]))
                self.assertEqual(tuple(table.concepts[i, 1 + j]), expected)
                expected_gain = int(table.predictions[i, 0] != data.Y[i]) - int(table.predictions[i, 1 + j] != data.Y[i])
                self.assertEqual(table.gains[i, j], expected_gain)
        self.assertTrue(all(chunk.shape[1] == len(data.categories) for chunk in head.calls))

    def test_hypothetical_inference_has_no_b_or_y_input(self):
        data = make_data(groups=4)
        lookup = FusionLookup(ConfusionFusion(data.categories).fit(data.A, data.B, data.C))
        self.assertEqual(tuple(inspect.signature(hypothetical_predictions).parameters),
                         ("initial_a", "lookup", "task_head", "batch_size"))
        original = hypothetical_predictions(data.A, lookup, ParityHead())
        poisoned = VerificationData(data.sample_ids, data.group_ids, data.A,
                                    np.column_stack([(data.B[:, j] + 1) % size for j, size in enumerate(data.categories)]),
                                    data.C, 1 - data.Y, data.categories)
        altered = hypothetical_predictions(poisoned.A, lookup, ParityHead())
        for p, other, size in zip(original, altered, data.categories):
            self.assertEqual(p.shape, (len(data), size))
            np.testing.assert_array_equal(p, other)
        for j in range(2):
            states = lookup.hypothetical_states(data.A, j)
            np.testing.assert_array_equal(original[j], states.sum(axis=2) % 2)

    def test_selected_semantic_events_exclude_stop_and_track_missing_gold(self):
        data = make_data(groups=4)
        c = data.C.copy()
        c[1, 0] = None
        data = VerificationData(data.sample_ids, data.group_ids, data.A, data.B, c, data.Y, data.categories)
        lookup = FusionLookup(ConfusionFusion(data.categories).fit(data.A, data.B, data.C))
        table = evaluate_actions(data, lookup, ParityHead())
        actions = [STOP, 0, 1] + [STOP] * (len(data) - 3)
        metrics = selected_event_metrics(data, table, actions)
        self.assertEqual(metrics["samples"], len(data))
        self.assertEqual(metrics["queried_samples"], 2)
        selected = metrics["selected_concept_events"]
        self.assertEqual(selected["samples"], 2)
        self.assertEqual(selected["concept_missing"], 1)
        self.assertEqual(selected["concept_labeled"], 1)
        stop = selected_event_metrics(data, table, [STOP] * len(data))
        self.assertEqual(stop["mean_signed_gain"], 0)
        self.assertEqual(stop["selected_concept_events"]["samples"], 0)
        self.assertEqual(stop["selected_concept_events"]["concept_missing"], 0)
        self.assertIsNone(stop["selected_concept_events"]["concept_damage_rate"])
        with self.assertRaises(ValueError):
            selected_event_metrics(data, table, [False] * len(data))

    def test_bad_head_output_fails_before_metrics(self):
        data = make_data(groups=3)
        lookup = FusionLookup(ConfusionFusion(data.categories).fit(data.A, data.B, data.C))
        class BadHead:
            def predict_proba(self, concepts):
                return np.zeros((len(concepts), 2))
        with self.assertRaisesRegex(ValueError, "normalized"):
            evaluate_actions(data, lookup, BadHead())

    def test_float32_probability_roundoff_is_normalized_in_float64(self):
        data = make_data(groups=3)
        lookup = FusionLookup(ConfusionFusion(data.categories).fit(data.A, data.B, data.C))
        class Float32Head:
            def predict_proba(self, concepts):
                return np.tile(np.array([0.49999991, 0.49999991], dtype=np.float32), (len(concepts), 1))
        table = evaluate_actions(data, lookup, Float32Head())
        self.assertEqual(table.probabilities.dtype, np.dtype("float64"))
        np.testing.assert_allclose(table.probabilities.sum(axis=2), 1, rtol=0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()

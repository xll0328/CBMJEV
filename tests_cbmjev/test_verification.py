"""Synthetic checks for information boundaries and signed verification value."""
from dataclasses import FrozenInstanceError
import inspect
import unittest

import numpy as np

from cbmjev.verification import (
    STOP, FUSION_METHODS, ConfusionFusion, VerificationObservation, choose_action,
    expected_action_gains, expected_signed_gain, joint_from_conditionals,
    legal_actions, observe_b, one_hot_initial, product_of_marginals,
    repair_damage_metrics, signed_gains, singleton_predictions,
)


def fit_binary(method="joint"):
    # A carries no information; B is a balanced, perfect concept source.
    return ConfusionFusion((2, 2), method=method, alpha=0.1).fit(
        [[0, 0]] * 4, [[0, 0], [0, 1], [1, 0], [1, 1]],
        [[0, 0], [0, 1], [1, 0], [1, 1]])


class ObservationTests(unittest.TestCase):
    def test_initial_state_is_hard_immutable_and_copied(self):
        initial, counts = [2, 4], [3, 5]
        state = VerificationObservation(initial, counts)
        initial[0], counts[0] = 0, 7
        self.assertEqual(state.initial_a, (2, 4))
        self.assertEqual(state.category_counts, (3, 5))
        self.assertEqual(state.observed_b, (None, None))
        self.assertEqual(legal_actions(state), (STOP, 0, 1))
        with self.assertRaises(FrozenInstanceError):
            state.initial_a = (0, 0)
        np.testing.assert_array_equal(one_hot_initial(state), [0, 0, 1, 0, 0, 0, 0, 1])

    def test_missing_is_not_a_semantic_category(self):
        state = VerificationObservation((2,), (3,))
        after = observe_b(state, 0, 2)
        self.assertEqual(after.observed_b, (2,))
        self.assertEqual(state.observed_b, (None,))
        self.assertEqual(legal_actions(after), (STOP,))
        for invalid in (None, -1, 3, 2.0, True, "2", [0.1, 0.2, 0.7]):
            with self.subTest(answer=invalid), self.assertRaises(ValueError):
                observe_b(state, 0, invalid)
        with self.assertRaisesRegex(ValueError, "twice"):
            observe_b(after, 0, 1)

    def test_malformed_initial_states_and_indices_rejected(self):
        for initial, counts, observed in [
            ((None,), (3,), None), ((-1,), (3,), None), ((3,), (3,), None),
            ((0.0,), (3,), None), ((False,), (3,), None), ((0,), (True,), None),
            ((0,), (3.0,), None), ((0,), (1,), None), ((), (), None),
            ((0,), (3, 3), None), ((0,), (3,), (0, 1)), ((0,), (3,), (-1,)),
        ]:
            with self.subTest(initial=initial, counts=counts), self.assertRaises(ValueError):
                VerificationObservation(initial, counts, observed)
        state = VerificationObservation((0,), (3,))
        for index in (-1, 1, True, 0.0):
            with self.subTest(index=index), self.assertRaises(ValueError):
                observe_b(state, index, 0)

    def test_all_fused_reference_can_be_assembled_without_repeated_queries(self):
        state = VerificationObservation((0, 4, 2), (2, 5, 3))
        for j, answer in enumerate((1, 3, 0)):
            state = observe_b(state, j, answer)
        self.assertEqual(state.observed_b, (1, 3, 0))
        self.assertEqual(legal_actions(state), (STOP,))


class FusionTests(unittest.TestCase):
    def test_initial_fusion_is_shared_across_all_methods(self):
        state = VerificationObservation((0, 0), (2, 2))
        models = [fit_binary(method) for method in FUSION_METHODS]
        for model in models:
            self.assertEqual(model.fuse(state), models[0].fuse(state))
            for actual, expected in zip(model.probabilities(state), models[0].probabilities(state)):
                np.testing.assert_allclose(actual, expected)
        queried = observe_b(state, 0, 1)
        self.assertEqual(models[0].fuse(queried), models[0].fuse(state))
        for model in models[1:]:
            self.assertEqual(model.fuse(queried), (1, 0))
            np.testing.assert_allclose(model.probabilities(queried)[1],
                                       model.probabilities(state)[1])

    def test_calibrated_a_is_explicitly_different_from_raw_a(self):
        model = ConfusionFusion((2,), method="a_only").fit(
            [[0]] * 4 + [[1]] * 4, [[0]] * 8, [[1]] * 4 + [[0]] * 4)
        state = VerificationObservation((0,), (2,))
        self.assertEqual(state.initial_a, (0,))
        self.assertEqual(model.fuse(state), (1,))

    def test_missing_labels_excluded_but_category_two_counted(self):
        model = ConfusionFusion((3,), alpha=1).fit(
            [[2], [0], [1]], [[2], [2], [None]], [[2], [None], [1]])
        self.assertEqual(model.labeled_counts, (2,))
        self.assertEqual(model.paired_counts, (1,))
        self.assertEqual(model.preferred_sources, ("a",))
        p = model.probabilities(VerificationObservation((2,), (3,)))[0]
        self.assertEqual(int(p.argmax()), 2)
        self.assertTrue(np.isfinite(p).all())
        self.assertAlmostEqual(p.sum(), 1)

    def test_reliability_compares_only_paired_labels_and_ties_favor_a(self):
        # The many unpaired perfect A observations must not beat B's paired accuracy.
        model = ConfusionFusion((2,), method="reliability").fit(
            [[0]] * 10 + [[0], [1]], [[None]] * 10 + [[1], [0]],
            [[0]] * 10 + [[1], [0]])
        self.assertEqual(model.preferred_sources, ("b",))
        tied = ConfusionFusion((2,), method="reliability").fit(
            [[0], [1]], [[0], [1]], [[0], [1]])
        self.assertEqual(tied.preferred_sources, ("a",))

    def test_joint_fusion_can_use_correlated_source_errors(self):
        # C = A XOR B; each source alone is uninformative about C.
        a, b, c = [[0], [0], [1], [1]], [[0], [1], [0], [1]], [[0], [1], [1], [0]]
        independent = ConfusionFusion((2,), method="independent", alpha=0.1).fit(a, b, c)
        joint = ConfusionFusion((2,), method="joint", alpha=0.1).fit(a, b, c)
        state = observe_b(VerificationObservation((0,), (2,)), 0, 1)
        self.assertEqual(independent.fuse(state), (0,))
        self.assertEqual(joint.fuse(state), (1,))

    def test_generic_21_by_5_concepts_and_heterogeneous_counts(self):
        for counts in ((5,) * 21, (2, 3, 5)):
            labels = [[row % size for size in counts] for row in range(15)]
            state = VerificationObservation(tuple(size - 1 for size in counts), counts)
            for method in FUSION_METHODS:
                model = ConfusionFusion(counts, method=method, alpha=0.1).fit(labels, labels, labels)
                self.assertEqual(model.fuse(state), state.initial_a)
                self.assertEqual(tuple(len(p) for p in model.probabilities(state)), counts)
                full = state
                for j, answer in enumerate(state.initial_a):
                    full = observe_b(full, j, answer)
                self.assertEqual(model.fuse(full), state.initial_a)

    def test_no_task_label_fit_parameter_and_strict_validation(self):
        self.assertEqual(tuple(inspect.signature(ConfusionFusion.fit).parameters),
                         ("self", "initial_a", "source_b", "concept_labels"))
        model = ConfusionFusion((3,))
        for a, b, c in [([], [], []), ([[0]], [[0], [1]], [[0]]),
                        ([[0.0]], [[0]], [[0]]), ([[0]], [[False]], [[0]]),
                        ([[0]], [[0]], [[-1]]), ([[0]], [[0]], [[np.nan]]),
                        ([[0]], [[0]], [[None]]), ([[0]], [[0]], [[3]])]:
            with self.subTest(a=a, b=b, c=c), self.assertRaises(ValueError):
                model.fit(a, b, c)
        for alpha in (0, -1, np.nan, np.inf, True):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                ConfusionFusion((3,), alpha=alpha)
        with self.assertRaises(ValueError):
            ConfusionFusion((3,), method="unknown")
        with self.assertRaisesRegex(ValueError, "fitted"):
            model.fuse(VerificationObservation((0,), (3,)))
        model.fit([[0]], [[0]], [[0]])
        with self.assertRaisesRegex(ValueError, "differ"):
            model.fuse(VerificationObservation((0,), (2,)))

    def test_two_world_prefix_invariance_and_reads_only_on_reveal(self):
        class Environment:
            def __init__(self, answers):
                self.answers, self.reads = answers, []

            def reveal(self, state, j):
                self.reads.append(j)
                return observe_b(state, j, self.answers[j])

        fusion = fit_binary()
        worlds = (Environment((1, 0)), Environment((1, 1)))
        states = [VerificationObservation((0, 0), (2, 2)) for _ in worlds]
        q = joint_from_conditionals([0.5, 0.5], [np.eye(2), np.eye(2)])

        def infer(state):
            before, after = singleton_predictions(state, fusion, lambda hard: hard[0])
            gains = expected_action_gains(q, before, after)
            return fusion.fuse(state), before, choose_action(state, gains, [1, 1])

        self.assertEqual(infer(states[0]), infer(states[1]))
        self.assertEqual(infer(states[0])[2], 0)
        self.assertEqual([w.reads for w in worlds], [[], []])
        states = [world.reveal(state, 0) for world, state in zip(worlds, states)]
        self.assertEqual(states[0], states[1])
        self.assertEqual(fusion.fuse(states[0]), fusion.fuse(states[1]))
        self.assertEqual([w.reads for w in worlds], [[0], [0]])
        for state in states:
            self.assertEqual(choose_action(state, [1, 1], [0, 0]), STOP)
        # A different *revealed* answer is allowed to alter fusion and prediction.
        alternate = observe_b(VerificationObservation((0, 0), (2, 2)), 0, 0)
        self.assertNotEqual(fusion.fuse(states[0]), fusion.fuse(alternate))


class DecisionValueTests(unittest.TestCase):
    def test_negative_and_positive_values_with_identical_marginals(self):
        positive, negative = np.eye(2) / 2, np.fliplr(np.eye(2)) / 2
        self.assertEqual(expected_signed_gain(positive, 0, [0, 1]), 0.5)
        self.assertEqual(expected_signed_gain(negative, 0, [0, 1]), -0.5)
        for joint in (positive, negative):
            product = product_of_marginals(joint)
            np.testing.assert_allclose(product.sum(axis=0), joint.sum(axis=0))
            np.testing.assert_allclose(product.sum(axis=1), joint.sum(axis=1))
            self.assertEqual(expected_signed_gain(product, 0, [0, 1]), 0)

    def test_shared_y_joint_construction_and_preserved_marginals(self):
        rng = np.random.default_rng(47)
        q_y = rng.dirichlet(np.ones(13))
        r = [rng.dirichlet(np.ones(5), size=13) for _ in range(21)]
        joints = joint_from_conditionals(q_y, r)
        after = [rng.integers(13, size=5) for _ in range(21)]
        values = expected_action_gains(joints, 3, after)
        ablated = expected_action_gains(joints, 3, after, product_ablation=True)
        self.assertEqual(values.shape, (21,))
        for q, v, pv in zip(joints, values, ablated):
            product = product_of_marginals(q)
            np.testing.assert_allclose(q.sum(axis=1), q_y)
            np.testing.assert_allclose(product.sum(axis=1), q_y)
            np.testing.assert_allclose(product.sum(axis=0), q.sum(axis=0))
            self.assertLessEqual(abs(v - pv), 0.5 * np.abs(q - product).sum() + 1e-12)

    def test_single_before_prediction_and_hard_only_task_inputs(self):
        state = VerificationObservation((0, 0), (2, 2))
        calls = []

        def predict(hard):
            self.assertIsInstance(hard, tuple)
            self.assertTrue(all(type(v) is int for v in hard))
            calls.append(hard)
            return int(hard[0] == 1)

        before, after = singleton_predictions(state, fit_binary(), predict)
        self.assertEqual(before, 0)
        self.assertEqual(len(calls), 1 + sum(state.category_counts))
        self.assertEqual(calls[0], (0, 0))
        self.assertEqual(after, ((0, 1), (0, 0)))
        q = joint_from_conditionals([0.5, 0.5], [np.eye(2), np.eye(2)])
        np.testing.assert_allclose(expected_action_gains(q, before, after), [0.5, 0])
        with self.assertRaisesRegex(ValueError, "initial state"):
            singleton_predictions(observe_b(state, 0, 1), fit_binary(), predict)
        with self.assertRaisesRegex(ValueError, "integer"):
            singleton_predictions(state, fit_binary(), lambda hard: 0.5)

    def test_stop_ties_signed_values_and_zero_cost_positive_edge(self):
        state = VerificationObservation((0, 0), (2, 2))
        self.assertEqual(choose_action(state, [-0.3, -0.1], [0, 0]), STOP)
        self.assertEqual(choose_action(state, [0, 0], [0, 0]), STOP)
        self.assertEqual(choose_action(state, [0.5, 0.5], [1, 1]), 0)
        self.assertEqual(choose_action(state, [0.5, 0.5], [1, 1], cost_weight=0.5), STOP)
        self.assertEqual(choose_action(state, [0.1, 0.5], [0, 1], cost_weight=1e100), 0)
        self.assertEqual(choose_action(state, [0.1, 0.5], [0, 1], cost_weight=0), 1)
        for gains, costs, weight in [([1], [0], 0), ([np.nan, 0], [0, 0], 0),
                                      ([0, 0], [-1, 0], 0), ([0, 0], [np.inf, 0], 0),
                                      ([0, 0], [0, 0], -1)]:
            with self.subTest(gains=gains, costs=costs), self.assertRaises(ValueError):
                choose_action(state, gains, costs, cost_weight=weight)

    def test_bad_distributions_shapes_and_marginals_rejected(self):
        for q in ([[0.4, 0], [0, 0.4]], [[1.1, -0.1], [0, 0]],
                  [[np.nan, 0], [0, 1]], [0.5, 0.5]):
            with self.subTest(q=q), self.assertRaises(ValueError):
                expected_signed_gain(q, 0, [0, 1])
        with self.assertRaisesRegex(ValueError, "normalized"):
            joint_from_conditionals([0.5, 0.5], [[[0.1, 0.1], [0.5, 0.5]]])
        with self.assertRaisesRegex(ValueError, "label space"):
            joint_from_conditionals([0.5, 0.5], [[[1.0]]])
        with self.assertRaisesRegex(ValueError, "same Y marginal"):
            expected_action_gains([np.eye(2) / 2, np.array([[0.75, 0], [0, 0.25]])],
                                  0, [[0, 1], [0, 1]])
        for before, after in [(0.0, [0, 1]), (2, [0, 1]), (0, [0]), (0, [0, None])]:
            with self.subTest(before=before, after=after), self.assertRaises(ValueError):
                expected_signed_gain(np.eye(2) / 2, before, after)


class EventMetricsTests(unittest.TestCase):
    def test_signed_gain_targets_retain_negative_values(self):
        before, after, gold = [0, 1, 0, 1], [1, 0, 0, 0], [1, 1, 1, 0]
        np.testing.assert_array_equal(signed_gains(before, after, gold), [1, -1, 0, 1])
        metrics = repair_damage_metrics(before, after, gold)
        self.assertEqual(metrics["samples"], 4)
        self.assertEqual(metrics["task_counts"], {"repair": 2, "damage": 1, "unchanged": 1})
        self.assertEqual(metrics["mean_signed_gain"],
                         metrics["task_repair_rate"] - metrics["task_damage_rate"])

    def test_semantic_cross_table_excludes_missing_and_counts_not_mentioned(self):
        result = repair_damage_metrics(
            [0, 1, 0, 0, 0], [1, 0, 0, 1, 1], [1, 1, 0, 1, 1],
            concept_before=[0, 2, 0, 0, 0], concept_after=[1, 0, 1, 2, 1],
            concept_gold=[1, 2, 2, None, 0])
        self.assertEqual(result["concept_labeled"], 4)
        self.assertEqual(result["concept_missing"], 1)
        self.assertEqual(result["concept_counts"],
                         {"repair": 1, "damage": 2, "unchanged": 1, "missing": 1})
        self.assertEqual(result["concept_changed_still_wrong"], 1)
        self.assertEqual(result["task_by_concept"]["repair"]["damage"], 1)
        self.assertEqual(result["task_by_concept"]["repair"]["missing"], 1)
        self.assertEqual(sum(sum(row.values()) for row in result["task_by_concept"].values()), 5)
        self.assertEqual(result["concept_damage_rate"], 0.5)

    def test_missing_is_not_unchanged_and_bad_metric_inputs_rejected(self):
        metrics = repair_damage_metrics([0], [0], [0], concept_before=[2],
                                        concept_after=[2], concept_gold=[None])
        self.assertEqual(metrics["concept_counts"]["unchanged"], 0)
        self.assertIsNone(metrics["concept_repair_rate"])
        with self.assertRaises(ValueError):
            repair_damage_metrics([0], [0, 1], [0])
        with self.assertRaises(ValueError):
            repair_damage_metrics([0], [0], [0], concept_gold=[None])
        for labels in ([False], [0.0], [-1], [None]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                signed_gains([0], [0], labels)


if __name__ == "__main__":
    unittest.main()

"""Small deterministic checks for frozen E1 control policies and selection."""
import unittest

from cbmjev.contracts import Concept, DeclaredCost, QueryGroup, Schema
from scripts.evaluate_cebab_e1_controls import (
    METHODS, PENALTIES, THRESHOLDS, Scores, choose, external_j, paired,
    replay, select_configuration,
)


class Head:
    def probabilities(self, observed):
        return (0.6, 0.4) if observed[0] == -1 else (0.1, 0.9)


class Controller:
    objective = "value"

    def predict(self, observed, actions):
        return tuple(1.0 if action == (0,) else 0.2 for action in actions)


class E1ControlsTests(unittest.TestCase):
    def setUp(self):
        self.schema = Schema("cebab", 2,
            tuple(Concept(str(i), str(i), ("no", "yes")) for i in range(4)),
            tuple(QueryGroup(str(i), (i,)) for i in range(4)))
        self.scores = Scores(self.schema, Head(), Controller())
        self.cost = DeclaredCost(per_group=1)
        self.order = (1, 0, 2, 3)
        self.row = {"sample_id": "a", "group_id": "family:a", "split": "validation",
                    "z": [1, 0, 0, 0], "y": 1}

    def test_same_information_boundary_and_gate(self):
        state = self.schema.empty_state()
        action, _, reason = choose(state, self.schema, self.scores, self.cost,
                                   self.order, "uncertainty_static", 0.55)
        self.assertEqual((action, reason), ((), "confidence_gate"))
        action, _, _ = choose(state, self.schema, self.scores, self.cost,
                              self.order, "uncertainty_static", 0.8)
        self.assertEqual(action, (1,))
        action, _, _ = choose(state, self.schema, self.scores, self.cost,
                              self.order, "uncertainty_value_order", 0.8)
        self.assertEqual(action, (0,))
        trace = replay(self.row, self.schema, self.scores, self.cost,
                       self.order, "uncertainty_value_order", 0.8)
        self.assertEqual(trace["queried_groups"], [0])
        self.assertEqual(trace["prediction"], 1)

    def test_entropy_cap_is_explicit_sensitivity_not_new_target(self):
        state = self.schema.empty_state()
        raw, _, _ = choose(state, self.schema, self.scores, self.cost,
                           self.order, "signed_value", 0.8)
        capped, _, _ = choose(state, self.schema, self.scores, self.cost,
                              self.order, "entropy_cap_sensitivity", 0.8)
        self.assertEqual(raw, (0,))
        self.assertEqual(capped, ())

    def test_policy_fit_selection_external_j(self):
        fit = {}
        for method in METHODS:
            grid = THRESHOLDS if method.startswith("uncertainty_") else PENALTIES
            for p in grid:
                fit[(method, p)] = {"error": 0.3, "mean_queried_groups": 2}
        fit[("signed_value", 0.03)] = {"error": 0.2, "mean_queried_groups": 3}
        self.assertEqual(select_configuration(fit, "signed_value", 0, 4), 0.03)
        self.assertEqual(select_configuration(fit, "signed_value", 0.4, 4), 0.0)
        self.assertAlmostEqual(external_j(fit[("signed_value", 0.03)], 0.4, 4), 0.5)
        with self.assertRaisesRegex(ValueError, "incomplete policy_fit"):
            select_configuration({("signed_value", 0.03): fit[("signed_value", 0.03)]},
                                 "signed_value", 0, 4)

    def test_paired_effects_not_just_query_changes(self):
        a = replay(self.row, self.schema, self.scores, self.cost,
                   self.order, "uncertainty_static", 0.55)
        b = replay(self.row, self.schema, self.scores, self.cost,
                   self.order, "signed_value", 0.0)
        comparison = paired([a], [b], 0.2, self.schema.num_groups)
        self.assertEqual(comparison["num_samples"], 1)
        self.assertEqual(comparison["path_different"], 1)
        self.assertEqual(comparison["query_set_different"], 1)
        self.assertEqual(comparison["a_wrong_b_correct"], 1)
        self.assertAlmostEqual(comparison["mean_b_minus_a_j"], -0.8)
        with self.assertRaisesRegex(ValueError, "identical nonempty"):
            paired([a], [], 0.2, self.schema.num_groups)


if __name__ == "__main__":
    unittest.main()

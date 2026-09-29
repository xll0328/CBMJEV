"""Focused invariants for the same-information CEBaB predictor control."""
import math
import unittest

from cbmjev.contracts import Concept, QueryGroup, Schema
from scripts.evaluate_cebab_predictor_sufficiency import (
    MASKS, MASK_WEIGHTS, family_split, fit_empirical, make_policy,
    evaluate_policy, text_reference)


class ToyHead:
    def probabilities(self, state):
        return (0.01, 0.99) if state[1] == 1 else (0.99, 0.01)

    def probabilities_many(self, states):
        return tuple(self.probabilities(state) for state in states)

    def predict(self, state):
        return 1 if state[1] == 1 else 0


class PredictorSufficiencyTests(unittest.TestCase):
    def setUp(self):
        self.schema = Schema("cebab", 2, tuple(
            Concept(f"c{i}", f"concept {i}", ("no", "yes")) for i in range(4)),
            tuple(QueryGroup(f"g{i}", (i,)) for i in range(4)))
        self.rows = [
            {"sample_id": f"s{i}", "group_id": f"family{i}",
             "split": "policy_fit", "z": [0, i % 2, 0, 0], "y": i % 2}
            for i in range(10)]

    def test_all_masks_preserve_original_cardinality_uniform_objective(self):
        self.assertAlmostEqual(sum(MASK_WEIGHTS), 1.0)
        for cardinality in range(5):
            self.assertAlmostEqual(
                sum(w for mask, w in zip(MASKS, MASK_WEIGHTS)
                    if sum(mask) == cardinality), 0.2)

    def test_inner_split_is_family_disjoint(self):
        fit, tune = family_split(self.rows)
        self.assertFalse({r["group_id"] for r in fit} & {r["group_id"] for r in tune})
        self.assertEqual(len(fit) + len(tune), len(self.rows))

    def test_fixed_order_restricts_every_successor(self):
        head = ToyHead()
        risk, branches = fit_empirical(self.rows, head, self.schema)
        fixed = make_policy(risk, branches, cost=0.01, order=(0, 1, 2, 3))
        dynamic = make_policy(risk, branches, cost=0.01)
        root = self.schema.empty_state()
        self.assertEqual(fixed(root)[1], 0)
        self.assertEqual(dynamic(root)[1], 1)
        self.assertEqual(fixed((0, -1, -1, -1))[1], 1)
        self.assertAlmostEqual(fixed(root)[0], 0.02)
        self.assertAlmostEqual(dynamic(root)[0], 0.01)
        fitted = evaluate_policy(self.rows, head, self.schema, fixed, 0.01)
        self.assertAlmostEqual(fitted["J"], fixed(root)[0])
        self.assertTrue(math.isfinite(fitted["nll"]))

    def test_text_reference_only_uses_head_fit_and_validation(self):
        records = [
            {"split": "head_fit", "text": "good food good service", "y": 1},
            {"split": "head_fit", "text": "good service good food", "y": 1},
            {"split": "head_fit", "text": "bad food bad service", "y": 0},
            {"split": "head_fit", "text": "bad service bad food", "y": 0},
            {"split": "validation", "text": "good food", "y": 1},
            {"split": "test", "text": "test secret with different label", "y": 0},
        ]
        report = text_reference(records, 2)
        self.assertEqual(report["fit"]["n"], 4)
        self.assertEqual(report["validation"]["n"], 1)


if __name__ == "__main__":
    unittest.main()

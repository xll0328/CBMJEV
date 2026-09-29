"""Focused correctness and information-boundary checks for DIME adaptation."""
import math
import unittest

import torch

from cbmjev.contracts import Concept, DeclaredCost, QueryGroup, Schema
from scripts.train_evaluate_cebab_dime_adapter import (
    PENALTIES, choose_penalty, external_j, fit_value_net,
    make_training_examples, replay, select_action,
)


class Head:
    def probabilities_many(self, states):
        return tuple(self.probabilities(state) for state in states)

    def probabilities(self, state):
        return (0.2, 0.8) if state[0] == 1 else (0.8, 0.2)


class Values:
    def predict(self, state):
        return (0.5, -0.2, 0.1, 0.0)


class DimeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.schema = Schema("cebab", 2,
            tuple(Concept(str(i), str(i), ("no", "yes")) for i in range(4)),
            tuple(QueryGroup(str(i), (i,)) for i in range(4)))
        self.row = {"sample_id": "a", "group_id": "family:a", "split": "policy_fit",
                    "z": [1, 0, 0, 0], "y": 1}
        self.cost = DeclaredCost(per_group=1)

    def test_signed_ce_target_and_training_role(self):
        features, target, legal = make_training_examples([self.row], self.schema, Head())
        self.assertEqual(features.shape[0], 16)
        self.assertEqual(target.shape, (16, 4))
        self.assertEqual(int(legal.sum()), 32)
        self.assertAlmostEqual(float(target[0, 0]), math.log(4), places=5)
        self.assertAlmostEqual(float(target[0, 1]), 0.0)
        self.assertFalse(bool(legal[1, 0]))
        self.assertEqual(float(target[1, 0]), 0.0)
        with self.assertRaisesRegex(ValueError, "policy_fit"):
            make_training_examples([{**self.row, "split": "validation"}],
                                   self.schema, Head())

    def test_no_future_response_or_label_at_inference_and_no_requery(self):
        empty = self.schema.empty_state()
        self.assertEqual(select_action(empty, self.schema, Values(), self.cost, 0)[0], (0,))
        state = (1, -1, -1, -1)
        self.assertEqual(select_action(state, self.schema, Values(), self.cost, 0)[0], (2,))
        self.assertEqual(select_action(state, self.schema, Values(), self.cost, 0.2)[0], ())
        trace = replay({**self.row, "split": "validation"}, self.schema,
                       Head(), Values(), self.cost, 0.2)
        self.assertEqual(trace["queried_groups"], [0])
        self.assertEqual(trace["prediction"], 1)
        self.assertEqual(trace["split"], "validation")

    def test_penalty_selection_uses_external_j_and_complete_fit_grid(self):
        reports = {p: {"error": 0.3, "mean_queried_groups": 2} for p in PENALTIES}
        reports[0.03] = {"error": 0.2, "mean_queried_groups": 3}
        self.assertEqual(choose_penalty(reports, 0, 4), 0.03)
        self.assertEqual(choose_penalty(reports, 0.4, 4), 0)
        self.assertAlmostEqual(external_j(reports[0.03], 0.4, 4), 0.5)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            choose_penalty({0.0: reports[0.0]}, 0, 4)

    def test_small_fit_is_finite_and_reproducible(self):
        features, targets, legal = make_training_examples([self.row], self.schema, Head())
        first, losses = fit_value_net(features, targets, legal, seed=3, hidden=8,
                                      epochs=2, batch_size=8)
        second, other = fit_value_net(features, targets, legal, seed=3, hidden=8,
                                      epochs=2, batch_size=8)
        self.assertEqual(losses, other)
        self.assertTrue(all(math.isfinite(v) for v in losses))
        for key, tensor in first.state_dict().items():
            self.assertTrue(torch.equal(tensor, second.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()

"""Unit and tiny fit checks for the concept-only same-target adapter."""
import unittest

from cbmjev.contracts import Concept, DeclaredCost, QueryGroup, Schema
from scripts.train_evaluate_cebab_lavoir_adapter import (
    AdapterScores, VARIANTS, choose, fit_pair, gini, paired, replay, targets,
)


class Head:
    def probabilities(self, state):
        p1 = 0.9 if state[0] == 1 else 0.1 if state[1] == 1 else 0.5
        return (1 - p1, p1)

    def probabilities_many(self, states):
        return tuple(self.probabilities(s) for s in states)


class FakeModel:
    def __init__(self, values):
        self.values = values

    def predict(self, state, actions):
        return tuple(self.values[action[0]] for action in actions)


class LavoirAdapterTests(unittest.TestCase):
    def setUp(self):
        self.schema = Schema("cebab", 2,
            tuple(Concept(str(i), str(i), ("no", "yes")) for i in range(4)),
            tuple(QueryGroup(str(i), (i,)) for i in range(4)))
        self.head = Head()
        self.row = {"sample_id": "a", "group_id": "family:a", "split": "validation",
                    "z": [1, 1, 0, 0], "y": 1}

    def test_gini_and_signed_gold_probability_target(self):
        self.assertAlmostEqual(gini((0.5, 0.5)), 0.5)
        self.assertAlmostEqual(gini((0.9, 0.1)), 0.18)
        with self.assertRaisesRegex(ValueError, "sum"):
            gini((0.5, 0.4))
        examples = [((-1, -1, -1, -1), (0,), (1, -1, -1, -1), 1),
                    ((-1, -1, -1, -1), (1,), (-1, 1, -1, -1), 1),
                    ((-1, 1, -1, -1), (0,), (1, 1, -1, -1), 1)]
        values, caps = targets(self.head, examples, self.schema)
        self.assertAlmostEqual(values[0], 0.4)
        self.assertAlmostEqual(values[1], -0.4)
        self.assertAlmostEqual(values[2], 0.8)
        self.assertAlmostEqual(caps[2], 0.18)
        self.assertGreater(values[2], caps[2])

    def test_cap_formula_and_stop(self):
        models = {"uncapped": FakeModel((0.8, 0.1, 0.1, 0.1)),
                  "gini_capped": FakeModel((0.8, 0.1, 0.1, 0.1))}
        scores = AdapterScores(self.schema, self.head, models)
        initial = self.schema.empty_state()
        uncapped = scores.values(initial, ((0,),), "uncapped")[0]
        capped = scores.values(initial, ((0,),), "gini_capped")[0]
        self.assertAlmostEqual(uncapped, 0.8)
        self.assertAlmostEqual(capped, 0.5 * 0.68997448, places=6)
        cost = DeclaredCost(per_group=1)
        self.assertEqual(choose(initial, self.schema, scores, cost, "uncapped", 0.4)[0], (0,))
        self.assertEqual(choose(initial, self.schema, scores, cost, "gini_capped", 0.4)[0], ())
        a = replay(self.row, self.schema, scores, cost, "uncapped", 0.4)
        b = replay(self.row, self.schema, scores, cost, "gini_capped", 0.4)
        result = paired(a=[a], b=[b], lam=0.1, groups=4)
        self.assertEqual(result["num_samples"], 1)
        self.assertEqual(result["path_different"], 1)

    def test_same_capacity_training_pilot_and_no_test_role(self):
        training = [dict(self.row, sample_id="p1", group_id="family:p1",
                         split="policy_fit"),
                    dict(self.row, sample_id="p2", group_id="family:p2",
                         split="policy_fit", z=[0, 1, 1, 0], y=0),
                    dict(self.row, sample_id="test-secret", group_id="family:test",
                         split="test", z=[1, 1, 1, 1], y=1)]
        config = {"hidden": 8, "policy_epochs": 2, "batch_size": 4,
                  "masks_per_sample": 3, "learning_rate": 0.001,
                  "dropout": 0.0, "weight_decay": 0.0}
        pair, report = fit_pair(training, self.schema, self.head, config, 40,
                               epochs=1)
        self.assertEqual(set(pair), set(VARIANTS))
        self.assertEqual(report["policy_fit_rows"], 2)
        self.assertEqual(report["epochs"], 1)
        self.assertEqual(report["target_signs_first_epoch"]["examples_first_epoch"],
                         report["examples_per_epoch"])
        self.assertTrue(all(len(report["training_mse"][v]) == 1 for v in VARIANTS))


if __name__ == "__main__":
    unittest.main()

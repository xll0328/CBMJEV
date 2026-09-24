import unittest

import torch

from cbmjev.contracts import Concept, QueryGroup, Schema
from cbmjev.learning import MaskedHead
from scripts.evaluate_oof_multistep_conditional_risk import (
    choose_next, fit_fold, sampled_events, state_action_features)


class FixedScore(torch.nn.Module):
    def forward(self, features):
        # The final 17 coordinates are the action encoding including STOP.
        return features[:, -17:][:, 2]


class OofMultistepRiskTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.schema = Schema("tiny", 2,
            tuple(Concept(f"c{i}", f"c{i}", ("no", "yes")) for i in range(16)),
            tuple(QueryGroup(f"g{i}", (i,)) for i in range(16)))

    def test_training_visibility_and_action_target_separation(self):
        rows = [dict(sample_id="a", group_id="a", z=[0] * 16, y=0),
                dict(sample_id="b", group_id="b", z=[1] * 16, y=1)]
        events = sampled_events(rows, self.schema, tuple(range(16)), 7, 4)
        self.assertTrue(events)
        for before, action, after, label in events:
            self.assertEqual(before[action], -1)
            self.assertGreaterEqual(after[action], 0)
            self.assertEqual(sum(v >= 0 for v in after), sum(v >= 0 for v in before) + 1)
            self.assertIn(label, (0, 1))

    def test_model_features_cannot_access_hidden_answer_or_label(self):
        state = (1,) + (-1,) * 15
        feature = state_action_features([state], [2], self.schema, "cpu")
        self.assertTrue(torch.equal(feature,
            state_action_features([state], [2], self.schema, "cpu")))
        self.assertNotEqual(float(feature[0, -17 + 2]), 0)
        with self.assertRaises(ValueError):
            state_action_features([state], [0], self.schema, "cpu")
        index, available = choose_next(state, [FixedScore()], self.schema, "cpu")
        self.assertEqual(available[index], 1)

    def test_tiny_fold_fit_and_multistep_rollin(self):
        head = MaskedHead(self.schema,
            {"device": "cpu", "hidden": 8, "dropout": 0.0, "class_weighting": "none"})
        rows = [dict(sample_id=str(i), group_id=str(i), z=[i % 2] * 16,
                     y=i % 2) for i in range(4)]
        model, report = fit_fold(rows, head, self.schema, tuple(range(16)),
            seed=3, hidden=8, epochs=1, batch_size=16,
            candidates_per_state=2, device="cpu")
        self.assertEqual(report["training_rows"], 4)
        self.assertGreater(report["events"], 0)
        self.assertTrue(torch.isfinite(torch.tensor(report["history"][-1]["mse"])))
        state = self.schema.empty_state()
        chosen = []
        for _ in range(16):
            index, available = choose_next(state, [model], self.schema, "cpu")
            action = available[index]
            chosen.append(action)
            state = tuple(0 if atom == action else value
                          for atom, value in enumerate(state))
        self.assertEqual(set(chosen), set(range(16)))


if __name__ == "__main__":
    unittest.main()

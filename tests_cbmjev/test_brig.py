import inspect
import math
import unittest

import torch

from cbmjev.brig import GroupQ, fit_brig
from cbmjev.contracts import Concept, QueryGroup, Schema


class ComplementHead:
    def probabilities_many(self, states):
        # A distractor gives a small immediate gain; B+C unlock a large gain.
        return [((.99 if s[1] >= 0 and s[2] >= 0 else .7 if s[0] >= 0 else .5),
                 (.01 if s[1] >= 0 and s[2] >= 0 else .3 if s[0] >= 0 else .5)) for s in states]


class BRiGTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.schema = Schema("tiny", 2, tuple(Concept(str(i), "feature", ("0", "1")) for i in range(3)),
                             tuple(QueryGroup(str(i), (i,)) for i in range(3)))
        self.rows = [dict(sample_id=str(i), group_id=str(i), split="policy_fit", z=[0, 0, 0], y=0)
                     for i in range(8)]

    def test_delayed_complementarity_and_budget(self):
        policy, report = fit_brig(self.rows, ComplementHead(), self.schema,
                                 dict(seed=7, max_budget=2, epochs=140, batch_size=8,
                                      hidden=32, learning_rate=.01),
                                 excluded_head_group_ids=[str(i) for i in range(8)])
        self.assertEqual(policy.choose((-1, -1, -1), 1), (0,))
        first = policy.choose((-1, -1, -1), 2)[0]
        self.assertIn(first, (1, 2))
        state = list(self.schema.empty_state())
        state[first] = 0
        second = policy.choose(tuple(state), 1)[0]
        self.assertEqual({first, second}, {1, 2})
        self.assertEqual(policy.choose(tuple(state), 0), ())
        with self.assertRaises(ValueError):
            policy.choose((0, 0, -1), 2)
        self.assertTrue(all(math.isfinite(x["mse"]) for x in report["history"]))
        self.assertTrue(any(x["rollout_targets"] > 0 for x in report["history"]))
        self.assertEqual(report["epochs_effective"], 140)

    def test_model_api_and_reacquisition(self):
        self.assertEqual(tuple(inspect.signature(GroupQ.forward).parameters),
                         ("self", "observed_states", "candidates", "remaining_budget"))
        model = GroupQ(self.schema, 8)
        with self.assertRaises(ValueError):
            model([(0, -1, -1)], [0], 1)
        with self.assertRaises(ValueError):
            model([(-1, -1, -1)], [True], 1)

    def test_bellman_without_rollout_auxiliary(self):
        policy, report = fit_brig(self.rows, ComplementHead(), self.schema,
                                 dict(seed=7, max_budget=2, epochs=140, batch_size=8,
                                      hidden=32, learning_rate=.01, empty_rollout=False),
                                 excluded_head_group_ids=[str(i) for i in range(8)])
        self.assertEqual(policy.choose(self.schema.empty_state(), 1), (0,))
        self.assertIn(policy.choose(self.schema.empty_state(), 2)[0], (1, 2))
        self.assertTrue(all(x["rollout_targets"] == 0 for x in report["history"]))

    def test_separation_and_determinism(self):
        with self.assertRaisesRegex(ValueError, "exclude"):
            fit_brig(self.rows, ComplementHead(), self.schema, excluded_head_group_ids=[])
        config = dict(max_budget=1, epochs=1, hidden=8)
        # Poisoned held-out row must not be inspected by the target producer.
        rows = self.rows + [dict(split="validation", y=object(), z=object())]
        p1, r1 = fit_brig(rows, ComplementHead(), self.schema, config,
                          excluded_head_group_ids=[str(i) for i in range(8)])
        p2, r2 = fit_brig(rows, ComplementHead(), self.schema, config,
                          excluded_head_group_ids=[str(i) for i in range(8)])
        self.assertEqual(r1["event_sha256"], r2["event_sha256"])
        self.assertEqual(r1["history"], r2["history"])
        self.assertEqual(r1["epochs_effective"], 8)
        self.assertEqual(p1.predict_budget(self.schema.empty_state(), 1),
                         p2.predict_budget(self.schema.empty_state(), 1))


if __name__ == "__main__":
    unittest.main()

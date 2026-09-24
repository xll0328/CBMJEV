import unittest
from unittest.mock import patch

import torch

from cbmjev.brig import GroupQ
from cbmjev.contracts import Concept, QueryGroup, Schema
from scripts.brig_fast_fit import fit_brig_fast
from scripts.brig_q_rle import GroupQRunLength


class RunLengthQTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.schema = Schema("rle", 2,
            tuple(Concept(str(i), "feature", ("0", "1")) for i in range(5)),
            tuple(QueryGroup(str(i), (i,)) for i in range(5)))

    def test_forward_and_gradient_exact_parity(self):
        a, b, c = (-1,) * 5, (0, -1, -1, -1, -1), (1, -1, -1, -1, -1)
        states = [a, a, a, b, b, c, c, b, b]
        actions = [0, 1, 2, 1, 2, 1, 3, 2, 4]
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                reference = GroupQ(self.schema, 12).to(device)
                candidate = GroupQRunLength(self.schema, 12).to(device)
                candidate.load_state_dict(reference.state_dict())
                left, right = reference(states, actions, 2), candidate(states, actions, 2)
                self.assertTrue(torch.equal(left, right))
                left.square().sum().backward()
                right.square().sum().backward()
                for old, new in zip(reference.parameters(), candidate.parameters()):
                    torch.testing.assert_close(old.grad, new.grad, atol=0, rtol=0)

    def test_invalid_inputs_still_rejected(self):
        head = GroupQRunLength(self.schema, 12)
        empty = self.schema.empty_state()
        for states, actions, budget in (
                ([empty], [True], 1), ([empty], [5], 1),
                ([empty], [0], 0), ([empty], [0], 6),
                ([(0, -1, -1, -1, -1)], [0], 1),
                ([empty], [], 1), ([(0, -1, -1)], [0], 1)):
            with self.subTest(states=states, actions=actions, budget=budget):
                with self.assertRaises(ValueError):
                    head(states, actions, budget)

    def test_full_batched_fit_event_and_weight_parity(self):
        class Head:
            def probabilities_many(self, states):
                return [(.8, .2) if state[0] >= 0 else (.4, .6) for state in states]

        rows = [dict(sample_id=str(i), group_id=str(i), split="policy_fit",
                     z=[i % 2] * 5, y=i % 2) for i in range(8)]
        config = dict(seed=17, device="cpu", hidden=12, max_budget=3,
                      epochs=8, batch_size=4, learning_rate=.001)
        old, old_report = fit_brig_fast(rows, Head(), self.schema, config,
            excluded_head_group_ids=[str(i) for i in range(8)])
        with patch("scripts.brig_fast_fit.GroupQ", GroupQRunLength):
            new, new_report = fit_brig_fast(rows, Head(), self.schema, config,
                excluded_head_group_ids=[str(i) for i in range(8)])
        self.assertEqual(old_report["event_sha256"], new_report["event_sha256"])
        self.assertEqual(old_report["history"], new_report["history"])
        for budget in old.models:
            for key, value in old.models[budget].state_dict().items():
                self.assertTrue(torch.equal(value, new.models[budget].state_dict()[key]))


if __name__ == "__main__":
    unittest.main()

import copy
import unittest

import torch

from cbmjev.brig import GroupQ
from cbmjev.contracts import Concept, QueryGroup, Schema
from scripts.brig_accelerated_ops import accelerated_groupq_forward


class AcceleratedGroupQTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.schema = Schema(
            "synthetic", 3,
            tuple(Concept(f"c{i}", f"concept {i}", ("no", "yes")) for i in range(6)),
            (QueryGroup("g0", (0, 1)), QueryGroup("g1", (2, 3)),
             QueryGroup("g2", (4, 5))),
        )

    def test_forward_values_and_gradients_match_reference(self):
        reference = GroupQ(self.schema, hidden=12)
        accelerated = copy.deepcopy(reference)
        states = [
            (-1, -1, -1, -1, -1, -1),
            (0, 1, -1, -1, -1, -1),
            (-1, -1, 2, 0, -1, -1),
            (1, 0, -1, -1, 1, 0),
        ] * 8
        candidates = [2, 1, 0, 1] * 8

        expected = reference(states, candidates, 1)
        actual = accelerated_groupq_forward(accelerated, states, candidates, 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        expected.square().mean().backward()
        actual.square().mean().backward()
        for left, right in zip(reference.parameters(), accelerated.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_validation_contract_matches_reference(self):
        model = GroupQ(self.schema, hidden=8)
        invalid_cases = [
            ([(0, -1, -1, -1, -1, -1)], [1], 1),  # partial group
            ([(0, 1, -1, -1, -1, -1)], [0], 1),   # reacquisition
            ([(-1, -1, -1, -1, -1, -1)], [3], 1), # invalid action
            ([(-1, -1, -1, -1, -1, -1)], [0], 4), # invalid budget
        ]
        for states, actions, budget in invalid_cases:
            with self.subTest(states=states, actions=actions, budget=budget):
                with self.assertRaises(ValueError):
                    model(states, actions, budget)
                with self.assertRaises(ValueError):
                    accelerated_groupq_forward(model, states, actions, budget)


if __name__ == "__main__":
    unittest.main()

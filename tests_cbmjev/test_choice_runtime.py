import unittest
from unittest.mock import Mock

import torch

from cbmjev.choice_features import choice_feature_width
from cbmjev.choice_runtime import StructuredChoiceController
from cbmjev.contracts import Concept, QueryGroup, Schema, DeclaredCost
from cbmjev.nano_choice import NanoChoiceHead
from cbmjev.runtime import ReplayEnvironment, choose_action, run_episode


class ChoiceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.schema = Schema("synthetic", 2,
            (Concept("a", "a", ("no", "yes")), Concept("b", "b", ("no", "yes"))),
            (QueryGroup("a", (0,)), QueryGroup("b", (1,))))

    def test_preference_argmax_no_double_cost_and_stop_ties(self):
        controller = Mock(objective="choice")
        controller.predict_logits.return_value = (0., 1., .5)
        action, scores = choose_action((-1, -1), ((), (0,), (1,)),
            method="structured_choice", controller=controller, remaining_groups=2,
            cost=DeclaredCost(per_group=100), cost_weight=100)
        self.assertEqual(action, (0,))
        self.assertEqual(scores["0"], 1.)
        controller.predict_logits.return_value = (1., 1., 1.)
        self.assertEqual(choose_action((-1, -1), ((), (0,), (1,)),
            method="structured_choice", controller=controller, remaining_groups=2)[0], ())
        controller.predict_logits.return_value = (float("nan"), 1., 0.)
        with self.assertRaises(ValueError):
            choose_action((-1, -1), ((), (0,), (1,)), method="structured_choice",
                          controller=controller, remaining_groups=2)

    def test_episode_enforces_budget_and_singleton_legal_set(self):
        controller = Mock(objective="choice")
        def scores(observed, actions, **kwargs):
            self.assertTrue(all(len(a) <= 1 for a in actions))
            return tuple(0. if not a else 1. for a in actions)
        controller.predict_logits.side_effect = scores
        task_head = Mock()
        task_head.probabilities.return_value = (.3, .7)
        trace = run_episode(ReplayEnvironment((1, 0), self.schema), self.schema,
            task_head, method="structured_choice", controller=controller, max_groups=1)
        self.assertEqual(trace["queried_groups"], [0])
        self.assertEqual(trace["calls"], 1)
        self.assertEqual(trace["final_state"], [1, -1])
        self.assertEqual(trace["steps"][-1]["decision"], "STOP")
        self.assertNotIn("y", trace)

    def test_actual_frozen_head_and_hidden_response_independence(self):
        head = NanoChoiceHead(choice_feature_width(self.schema)).eval().requires_grad_(False)
        controller = StructuredChoiceController(self.schema, head)
        a = controller.predict_logits((-1, -1), ((), (0,), (1,)),
            remaining_groups=2, cost=DeclaredCost(), cost_weight=.1)
        b = controller.predict_logits((-1, -1), ((), (0,), (1,)),
            remaining_groups=2, cost=DeclaredCost(), cost_weight=.1)
        self.assertEqual(a, b)
        with self.assertRaisesRegex(ValueError, "singleton"):
            controller.predict_logits((-1, -1), ((), (0, 1)),
                remaining_groups=2, cost=DeclaredCost(), cost_weight=.1)
        with self.assertRaisesRegex(ValueError, "frozen"):
            StructuredChoiceController(self.schema, head.train())


if __name__ == "__main__":
    unittest.main()

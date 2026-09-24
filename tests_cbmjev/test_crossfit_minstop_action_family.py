import unittest

from cbmjev.contracts import DeclaredCost
from cbmjev.runtime import ReplayEnvironment, run_episode
from scripts.evaluate_crossfit_minstop_grid import (choose_value_minstop,
                                                    run_episode_minstop)
from tests_cbmjev.test_contracts_runtime import Head, fixture_schema


class ValueController:
    objective = "value"
    pairs = ((0, 1),)

    def predict(self, observed, actions):
        acquired = sum(value != -1 for value in observed)
        return tuple((.9 if not action else .2) if acquired else
                     (0. if not action else .6 if len(action) == 1 else .8)
                     for action in actions)


class MinStopActionFamilyTests(unittest.TestCase):
    def setUp(self):
        self.schema = fixture_schema()
        self.controller = ValueController()
        self.cost = DeclaredCost()
        self.z = (1, 0, 2)

    def diagnostic(self, family, min_groups):
        return run_episode_minstop(ReplayEnvironment(self.z, self.schema),
            self.schema, Head(), controller=self.controller, cost=self.cost,
            cost_weight=.03, max_groups=2, min_groups=min_groups,
            max_cost=None, include_pairs=True, include_all=True,
            pairs=self.controller.pairs, device="cpu", action_family=family)

    def original(self, method):
        return run_episode(ReplayEnvironment(self.z, self.schema),
            self.schema, Head(), method=method, controller=self.controller,
            cost=self.cost, cost_weight=.03, max_groups=2,
            include_pairs=True, include_all=True, pairs=self.controller.pairs)

    def test_min_zero_matches_original_action_family_and_prediction(self):
        for family, method in (("singleton", "value_singleton"),
                               ("configured", "value")):
            with self.subTest(family=family):
                actual, expected = self.diagnostic(family, 0), self.original(method)
                for key in ("prediction", "probabilities", "final_state",
                            "queried_groups", "queried_atoms", "calls", "declared_cost"):
                    self.assertEqual(actual[key], expected[key])
                self.assertEqual([step["action"] for step in actual["steps"]],
                                 [step["action"] for step in expected["steps"]])
                self.assertEqual([step["scores"] for step in actual["steps"]],
                                 [step["scores"] for step in expected["steps"]])
        self.assertEqual(self.diagnostic("configured", 0)["calls"], 1)
        self.assertEqual(self.diagnostic("singleton", 0)["calls"], 1)

    def test_forced_stop_changes_only_stop_permission_and_keeps_pair_legal(self):
        configured = self.diagnostic("configured", 2)
        singleton = self.diagnostic("singleton", 2)
        self.assertEqual(configured["steps"][0]["action"], [0, 1])
        self.assertEqual(configured["queried_groups"], [0, 1])
        self.assertEqual(configured["calls"], 1)
        self.assertEqual(singleton["queried_groups"], [0, 1])
        self.assertEqual(singleton["calls"], 2)
        with self.assertRaisesRegex(ValueError, "unknown min-stop action family"):
            choose_value_minstop(self.schema.empty_state(), ((), (0,)),
                schema=self.schema, controller=self.controller, cost=self.cost,
                cost_weight=.03, min_groups=1, action_family="unknown")

    def test_infeasible_floor_fails_instead_of_silent_stop(self):
        with self.assertRaisesRegex(ValueError, "min-stop floor infeasible"):
            choose_value_minstop(self.schema.empty_state(), ((),),
                schema=self.schema, controller=self.controller, cost=self.cost,
                cost_weight=.03, min_groups=1, action_family="configured")


if __name__ == "__main__":
    unittest.main()

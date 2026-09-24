import unittest

from cbmjev.config import resolve_config
from cbmjev.runtime import ReplayEnvironment, run_episode
from tests_cbmjev.test_crossfit_training import schema_fixture


class Head:
    def probabilities(self, observed):
        return (0.5, 0.5)


class Controller:
    objective = "value"

    def __init__(self):
        self.seen = []

    def predict(self, observed, actions):
        self.seen.append((observed, actions))
        return tuple(0.0 if not a else (-1.0 if observed[0] == 0 else 0.2 + sum(a))
                     for a in actions)


class StoppingControlTests(unittest.TestCase):
    def episode(self, method, answers=(0, 1, 1), **options):
        schema = schema_fixture()
        controller = Controller()
        trace = run_episode(ReplayEnvironment(answers, schema), schema, Head(),
                            controller=controller, method=method, **options)
        return trace, controller

    def test_static_order_is_fixed_but_stopping_depends_on_observed_evidence(self):
        stopped, controller = self.episode("static_value")
        continuing, _ = self.episode("static_value", answers=(1, 1, 1))
        self.assertEqual([s["action"] for s in stopped["steps"]], [[0], []])
        self.assertEqual([s["action"] for s in continuing["steps"]], [[0], [1], [2], []])
        self.assertEqual(controller.seen[0][0], (-1, -1, -1))
        self.assertEqual(set(controller.seen[0][1]), {(), (0,)})
        self.assertEqual(controller.seen[1][0], (0, -1, -1))

    def test_static_uses_supplied_training_order(self):
        trace, _ = self.episode("static_value", order=(2, 0, 1))
        self.assertEqual([s["action"] for s in trace["steps"]], [[2], [0], []])

    def test_dynamic_singleton_disables_batching_not_selection(self):
        trace, controller = self.episode("value_singleton", max_groups=1)
        self.assertEqual(trace["steps"][0]["action"], [2])
        self.assertTrue(all(len(a) <= 1 for _, actions in controller.seen for a in actions))

    def test_budget_and_cost_stopping(self):
        for method in ("static_value", "value_singleton"):
            with self.subTest(method=method):
                trace, _ = self.episode(method, max_groups=0)
                self.assertEqual(trace["queried_groups"], [])
        trace, _ = self.episode("static_value", cost_weight=0.2)
        self.assertEqual(trace["queried_groups"], [])  # STOP wins a value/cost tie.

    def test_requires_value_controller_configuration(self):
        resolve_config({"learning": {"objective": "value"},
                        "evaluation": {"methods": ["static_value", "value_singleton"]}})
        for method in ("static_value", "value_singleton"):
            with self.assertRaises(ValueError):
                resolve_config({"learning": {"objective": "risk"},
                                "evaluation": {"methods": [method]}})


if __name__ == "__main__":
    unittest.main()

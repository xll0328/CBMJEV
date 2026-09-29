"""Focused guards for matched-loss empirical ordered DP evaluation."""
import unittest

from cbmjev.baselines import EmpiricalLookaheadPolicy
from cbmjev.contracts import Concept, DeclaredCost, QueryGroup, Schema
from scripts.evaluate_cebab_ordered_empirical_dp import (
    CachedDecisions, empirical_objective, evaluate, paired_comparison, replay,
    select_fixed_order)


class BinaryHead:
    def probabilities(self, state):
        return (0.1, 0.9) if state[1] == 1 else (0.9, 0.1)

    def predict(self, state):
        return max(range(2), key=lambda index: self.probabilities(state)[index])


class OrderedEmpiricalDPTests(unittest.TestCase):
    def setUp(self):
        self.schema = Schema("cebab", 2, (
            Concept("c0", "first", ("no", "yes")),
            Concept("c1", "second", ("no", "yes"))),
            (QueryGroup("g0", (0,)), QueryGroup("g1", (1,))))
        self.head = BinaryHead()
        self.rows = [
            {"sample_id": "p0", "group_id": "family0", "split": "policy_fit", "z": [0, 0], "y": 0},
            {"sample_id": "p1", "group_id": "family1", "split": "policy_fit", "z": [0, 1], "y": 1},
            {"sample_id": "p2", "group_id": "family2", "split": "policy_fit", "z": [1, 1], "y": 1},
            {"sample_id": "v0", "group_id": "family3", "split": "validation", "z": [1, 0], "y": 0},
        ]
        self.cost = DeclaredCost(per_group=1.0)

    def test_replay_order_is_respected_and_runtime_is_history_only(self):
        policy = EmpiricalLookaheadPolicy.fit(self.rows, self.head, self.schema,
            depth=None, max_groups=2, include_pairs=False, include_all=False, order=(0, 1))
        decisions = CachedDecisions(policy, self.cost, 0.01, None)
        trace = replay(self.rows[-1], self.head, policy, decisions, "ordered")
        path = [step["action"] for step in trace["steps"] if step["decision"] == "ACQUIRE"]
        self.assertEqual(path, [[0], [1]])
        self.assertEqual(trace["queried_groups"], [0, 1])
        self.assertEqual(trace["prediction"], 0)
        self.assertEqual(trace["declared_cost"], 2.0)
        self.assertEqual(decisions.choose(self.schema.empty_state(), 0), ((0,), 2))

    def test_dynamic_and_ordered_have_same_terminal_and_cost_objective(self):
        fixed = EmpiricalLookaheadPolicy.fit(self.rows, self.head, self.schema,
            depth=None, max_groups=2, include_pairs=False, include_all=False, order=(0, 1))
        dynamic = EmpiricalLookaheadPolicy.fit(self.rows, self.head, self.schema,
            depth=None, max_groups=2, include_pairs=False, include_all=False)
        validation = self.rows[-1:]
        left, left_report, _ = evaluate(validation, self.head, fixed, family="fixed",
                                       cost=self.cost, weight=0.01, max_cost=None)
        right, right_report, _ = evaluate(validation, self.head, dynamic, family="dynamic",
                                         cost=self.cost, weight=0.01, max_cost=None)
        self.assertAlmostEqual(empirical_objective(left_report, 0.01),
                               left_report["error"] + 0.01 * left_report["mean_declared_cost"])
        self.assertAlmostEqual(empirical_objective(right_report, 0.01),
                               right_report["error"] + 0.01 * right_report["mean_declared_cost"])
        paired = paired_comparison(left, right)
        self.assertEqual(paired["num_samples"], 1)

    def test_training_only_selection_requires_all_orders(self):
        with self.assertRaisesRegex(ValueError, "24"):
            select_fixed_order([], 0.03)
        reports = [{"order": list(order), "report": {"error": index / 100,
                   "mean_declared_cost": 1.0}} for index, order in enumerate(
                   __import__("itertools").permutations(range(4)))]
        self.assertEqual(select_fixed_order(reports, 0.03)["order"], [0, 1, 2, 3])

    def test_paired_requires_identical_validation_population(self):
        with self.assertRaisesRegex(ValueError, "matching"):
            paired_comparison([{"sample_id": "a"}], [{"sample_id": "b"}])


if __name__ == "__main__":
    unittest.main()

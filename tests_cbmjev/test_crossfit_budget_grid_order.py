import copy
import tempfile
import unittest
from pathlib import Path

from cbmjev.contracts import Concept, QueryGroup, Schema
from cbmjev.runtime import ReplayEnvironment, run_episode
from cbmjev.io import file_hash
from scripts.evaluate_crossfit_budget_grid import (
    order_for_method, verify_expected_grid_binding, verify_plan_files_unchanged,
)


class ConstantHead:
    def probabilities(self, state):
        return (0.6, 0.4)


class CrossfitBudgetGridOrderTests(unittest.TestCase):
    def test_plan_hashes_are_checked_with_stable_evaluator_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            planned = Path(temp_dir)
            (planned / "plan.json").write_text("plan-v1", encoding="utf-8")
            (planned / "fold_manifest.jsonl").write_text("fold-v1\n", encoding="utf-8")
            binding = {"plan_binding": {
                "plan_sha256": file_hash(planned / "plan.json"),
                "fold_manifest_sha256": file_hash(planned / "fold_manifest.jsonl"),
            }}
            verify_plan_files_unchanged(binding, planned)
            (planned / "plan.json").write_text("plan-v2", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outer plan changed"):
                verify_plan_files_unchanged(binding, planned)

    def test_canonical_rerun_requires_same_system_and_old_alias(self):
        keys = ("merged_receipt_sha256", "responder_receipt_sha256",
                "cache_manifest_sha256", "outer_sources", "plan_binding",
                "head_component_sha256", "controller_component_sha256",
                "responder_checkpoint_sha256")
        current = {key: key + ":same" for key in keys}
        old = dict(current, static_order={"order": [1, 0]})
        metric = {"group_metrics": {"image:1": {"error": 0.0}}, "accuracy": 1.0}
        expected = {"seed": 60, "split": "validation", "mode": "offline_replay",
                    "source_binding": old,
                    "policies": {"fixed_K1": dict(metric, method="fixed"),
                                 "static_K1": dict(metric, method="static")}}
        verified = verify_expected_grid_binding(expected, current, 60)
        self.assertEqual(verified["shared_aliased_budgets"], [1])
        wrong = dict(current, responder_receipt_sha256="different")
        with self.assertRaisesRegex(ValueError, "does not match"):
            verify_expected_grid_binding(expected, wrong, 60)
        with self.assertRaisesRegex(ValueError, "seed differs"):
            verify_expected_grid_binding(expected, current, 61)
        test_grid = copy.deepcopy(expected)
        test_grid["split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation offline replay"):
            verify_expected_grid_binding(test_grid, current, 60)
        altered = copy.deepcopy(expected)
        altered["policies"]["static_K1"]["accuracy"] = 0.0
        with self.assertRaisesRegex(ValueError, "alias not verified"):
            verify_expected_grid_binding(altered, current, 60)
        with self.assertRaisesRegex(ValueError, "must not"):
            verify_expected_grid_binding(expected, old, 60)

    def test_fixed_and_static_use_distinct_orders(self):
        schema = Schema("tiny", 2,
            tuple(Concept(f"c{i}", f"c{i}", ("no", "yes")) for i in range(3)),
            tuple(QueryGroup(f"g{i}", (i,)) for i in range(3)))
        static_order = [2, 1, 0]
        self.assertIsNone(order_for_method("fixed", static_order))
        self.assertIsNone(order_for_method("value", static_order))
        self.assertEqual(order_for_method("static", static_order), static_order)
        self.assertEqual(order_for_method("static_value", static_order), static_order)
        for method, expected in (("fixed", 0), ("static", 2)):
            trace = run_episode(ReplayEnvironment((0, 0, 0), schema), schema,
                ConstantHead(), method=method, max_groups=1,
                order=order_for_method(method, static_order))
            self.assertEqual(trace["queried_groups"], [expected])


if __name__ == "__main__":
    unittest.main()

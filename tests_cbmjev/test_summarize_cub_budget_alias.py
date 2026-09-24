import json
import tempfile
import unittest
from pathlib import Path

from scripts.figures.summarize_cub_budget_grids import verify_fixed_static_alias


class BudgetAliasDisplayTest(unittest.TestCase):
    def test_requires_identical_reports_and_noncanonical_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            report = {
                "group_metrics": {"image:1": {"error": 0.0}},
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "mean_declared_cost": 1.0,
                "mean_queried_groups": 1.0,
                "mean_calls": 1.0,
                "num_samples": 1,
            }
            metrics = {
                "source_binding": {"static_order": {"order": [1, 0]}},
                "policies": {"fixed_K1": report, "static_K1": dict(report)},
            }
            path.write_text(json.dumps(metrics), encoding="utf-8")
            verify_fixed_static_alias(path)

            metrics["policies"]["fixed_K1"]["accuracy"] = 0.0
            path.write_text(json.dumps(metrics), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "accuracy differs"):
                verify_fixed_static_alias(path)

            metrics["policies"]["fixed_K1"]["accuracy"] = 1.0
            metrics["source_binding"]["static_order"]["order"] = [0, 1]
            path.write_text(json.dumps(metrics), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "noncanonical"):
                verify_fixed_static_alias(path)


if __name__ == "__main__":
    unittest.main()

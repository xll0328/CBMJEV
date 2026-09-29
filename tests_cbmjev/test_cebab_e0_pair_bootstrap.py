"""E0 family-bootstrap input and pairing checks."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_cebab_e0_pair_bootstrap import compare


class PairBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "traces.jsonl"
        self.report = {
            "format": "cbmjev-cebab-order-stop-factorial-v1", "test_evaluated": False,
            "seed": 40, "selection": {"selected_order_by_cell": {"stop_capK4": [0, 1, 2, 3]}},
            "population": {"validation_samples": 2},
            "configuration": {"cost_weight": .1}}
        rows = []
        for family in ("fixed_order_0123_stop_capK4", "dynamic_singleton_stop_capK4"):
            for index in range(2):
                rows.append({"family": family, "split": "validation",
                             "sample_id": str(index), "group_id": f"g{index}", "y": 1,
                             "prediction": (1 if family.startswith("fixed") else index),
                             "declared_cost": 2 if family.startswith("fixed") else 1})
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_paired_family_estimates(self):
        result = compare(self.report, self.path, resamples=100, seed=5)
        self.assertEqual(result["families"], 2)
        self.assertAlmostEqual(result["estimates"]["error"]["dynamic_minus_static_equal_family_mean"], .5)
        self.assertAlmostEqual(result["estimates"]["declared_cost"]["dynamic_minus_static_equal_family_mean"], -1)
        self.assertAlmostEqual(result["estimates"]["J"]["dynamic_minus_static_equal_family_mean"], .4)

    def test_test_flag_and_missing_partner_rejected(self):
        self.report["test_evaluated"] = True
        with self.assertRaisesRegex(ValueError, "non-test"):
            compare(self.report, self.path, resamples=2)
        self.report["test_evaluated"] = False
        self.path.write_text(self.path.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "unpaired"):
            compare(self.report, self.path, resamples=2)


if __name__ == "__main__":
    unittest.main()

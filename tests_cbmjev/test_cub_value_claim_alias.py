import unittest

from scripts.analyze_cub_value_claim import analyze as analyze_summary
from scripts.analyze_cub_value_seed_deltas import analyze as analyze_seeds


class CubValueClaimAliasTests(unittest.TestCase):
    def test_explicit_fitted_static_baseline_avoids_aliased_fixed_label(self):
        summary = {"seeds": [60, 61], "num_seeds": 2, "policies": {
            "value_K1": {"method": "value", "budget_groups": 1,
                         "accuracy_mean": .2, "num_seeds": 2},
            "fixed_K1": {"method": "fixed", "budget_groups": 1,
                         "accuracy_mean": .3, "num_seeds": 2},
            "static_K1": {"method": "static", "budget_groups": 1,
                          "accuracy_mean": .3, "num_seeds": 2},
        }}
        result = analyze_summary(summary, "accuracy", .005, ("static", "static_value"))
        self.assertEqual(result["status"], "no_observed_matched_budget_gain")
        self.assertEqual(result["baseline_methods"], ["static", "static_value"])
        self.assertEqual(result["comparisons"][0]["baseline_method"], "static")
        self.assertAlmostEqual(result["comparisons"][0]["delta_vs_best_baseline"], -.1)
        self.assertIn("equivalence is not established", result["rationale"])

    def test_seed_pairs_keep_fitted_static_identity(self):
        rows = []
        for seed in (60, 61):
            for method, value in (("value", .2), ("fixed", .3), ("static", .3)):
                rows.append({"seed": seed, "budget_groups": 1,
                             "method": method, "policy_id": method + "_K1",
                             "accuracy": value})
        result = analyze_seeds(rows, "accuracy", .005, ("static", "static_value"))
        self.assertEqual(result["baseline_methods"], ["static", "static_value"])
        self.assertEqual({item["baseline_method"] for item in
                          result["budget_reports"][0]["paired_deltas"]}, {"static"})

    def test_rejects_invalid_baseline_scope(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            analyze_summary({"policies": {}}, "accuracy", .005, ())
        with self.assertRaisesRegex(ValueError, "nonempty"):
            analyze_seeds([], "accuracy", .005, ("random",))

    def test_duplicate_fixed_static_requires_static_only_analysis(self):
        summary = {"policies": {}}
        rows = []
        for budget in (1, 2):
            for method in ("fixed", "static"):
                summary["policies"][f"{method}_K{budget}"] = {
                    "method": method, "budget_groups": budget,
                    "accuracy_mean": .3, "macro_f1_mean": .2,
                    "mean_queried_groups_mean": budget,
                    "mean_calls_mean": budget, "mean_declared_cost_mean": budget,
                    "num_seeds": 1}
                rows.append({"seed": 60, "budget_groups": budget,
                             "method": method, "policy_id": f"{method}_K{budget}",
                             "accuracy": .3, "macro_f1": .2,
                             "mean_queried_groups": budget,
                             "mean_calls": budget, "mean_declared_cost": budget})
        with self.assertRaisesRegex(ValueError, "indistinguishable"):
            analyze_summary(summary, "accuracy", .005, ("static", "fixed"))
        with self.assertRaisesRegex(ValueError, "indistinguishable"):
            analyze_seeds(rows, "accuracy", .005, ("static", "fixed"))


if __name__ == "__main__":
    unittest.main()

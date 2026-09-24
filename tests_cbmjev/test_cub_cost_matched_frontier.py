import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_cub_cost_matched_frontier import analyze_one, summarize, summarize_grid


def fixture(seed=60, cost=10.0, accuracy=0.26):
    def report(method, k, acc):
        return {"method": method, "split": "validation", "num_samples": 10,
                "mean_declared_cost": float(k), "mean_queried_groups": float(k),
                "accuracy": acc}
    policies = {f"fixed_K{k}": report("fixed", k, acc) for k, acc in
                ((4, .2), (8, .24), (16, .32))}
    policies["value_K16"] = report("value", cost, accuracy)
    return {"split": "validation", "mode": "offline_replay", "seed": seed,
            "source_binding": {"fixture": True}, "policies": policies}


class CostMatchedFrontierTests(unittest.TestCase):
    def test_interpolates_expected_accuracy_at_equal_mean_cost(self):
        row = analyze_one(fixture(), baseline_method="fixed")
        self.assertEqual(row["low_static"]["policy_id"], "fixed_K8")
        self.assertEqual(row["high_static"]["policy_id"], "fixed_K16")
        self.assertAlmostEqual(row["high_static_probability"], .25)
        self.assertAlmostEqual(row["static_mixture_expected_declared_cost"], 10)
        self.assertAlmostEqual(row["static_mixture_expected_accuracy"], .26)
        self.assertAlmostEqual(row["adaptive_minus_static_mixture_accuracy"], 0)

    def test_rejects_extrapolation_or_unpaired_sample_counts(self):
        with self.assertRaisesRegex(ValueError, "outside baseline cost"):
            analyze_one(fixture(cost=20), baseline_method="fixed")
        altered = fixture()
        altered["policies"]["fixed_K8"]["num_samples"] = 9
        with self.assertRaisesRegex(ValueError, "sample count"):
            analyze_one(altered, baseline_method="fixed")

    def test_flags_historical_fixed_static_alias_without_relabeling_results(self):
        metrics = fixture()
        metrics["source_binding"]["static_order"] = {"order": [2, 0, 1]}
        for k in (4, 8, 16):
            metrics["policies"][f"fixed_K{k}"]["group_metrics"] = {
                "image": {"num_samples": 10, "error": .2, "declared_cost": float(k)}}
            metrics["policies"][f"static_K{k}"] = json.loads(json.dumps(
                metrics["policies"][f"fixed_K{k}"]))
        row = analyze_one(metrics, baseline_method="fixed")
        self.assertTrue(row["fixed_static_group_metrics_identical_all_shared_budgets"])
        self.assertTrue(row["static_order_noncanonical"])
        self.assertEqual(row["shared_fixed_static_budgets"], [4, 8, 16])

    def test_seed_summary_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"metrics_{index}.json" for index in range(2)]
            for path in paths:
                path.write_text(json.dumps(fixture()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate seed"):
                summarize(paths, baseline_method="fixed")

    def test_full_grid_reads_sources_once_and_uses_fitted_static(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for seed in (60, 61):
                metrics = fixture(seed=seed, cost=10, accuracy=.26)
                for k in (4, 8, 16):
                    metrics["policies"][f"static_K{k}"] = dict(
                        metrics["policies"][f"fixed_K{k}"], method="static")
                metrics["policies"]["value_singleton_K16"] = dict(
                    metrics["policies"]["value_K16"],
                    method="value_singleton", accuracy=.28)
                path = Path(directory) / f"seed{seed}.json"
                path.write_text(json.dumps(metrics), encoding="utf-8")
                paths.append(path)
            report = summarize_grid(paths, adaptive_policies=(
                "value_K16", "value_singleton_K16"))
            self.assertEqual(analyze_one(metrics)["low_static"]["policy_id"], "static_K8")
            self.assertEqual(summarize([paths[0]])["baseline_method"], "static")
            self.assertEqual(report["baseline_method"], "static")
            self.assertEqual(len(report["sources"]), 2)
            self.assertEqual(report["policies"]["value_K16"]["num_seeds"], 2)
            self.assertAlmostEqual(report["policies"]["value_K16"]["mean_accuracy_delta"], 0)
            self.assertAlmostEqual(report["policies"]["value_singleton_K16"]["mean_accuracy_delta"], .02)
            self.assertIsNone(report["policies"]["value_K16"]["rows"][0]["paired_group_bootstrap"])
            with self.assertRaisesRegex(ValueError, "fitted static"):
                summarize_grid(paths, adaptive_policies=("value_K16",),
                               baseline_method="fixed")
            with self.assertRaisesRegex(ValueError, "adaptive policies"):
                summarize_grid(paths, adaptive_policies=("static_K16",))

    def test_paired_group_bootstrap_uses_matched_groups_and_recomputed_cost(self):
        metrics = fixture(cost=2, accuracy=.5)
        metrics["policies"] = {}
        for name, cost, errors in (
                ("fixed_K0", 0, [1, 1, 1, 0]),
                ("fixed_K4", 4, [1, 0, 0, 0]),
                ("value_K16", 2, [1, 0, 1, 0])):
            metrics["policies"][name] = {
                "split": "validation", "num_samples": 4,
                "mean_declared_cost": float(cost),
                "mean_queried_groups": float(cost),
                "accuracy": 1 - sum(errors) / 4,
                "group_metrics": {f"image:{i}": {"num_samples": 1,
                                   "error": error, "declared_cost": float(cost)}
                                  for i, error in enumerate(errors)}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps(metrics), encoding="utf-8")
            first = summarize([path], baseline_method="fixed")["rows"][0]["paired_group_bootstrap"]
            self.assertEqual(first, summarize([path], baseline_method="fixed")["rows"][0]["paired_group_bootstrap"])
            self.assertEqual(first["groups"], 4)
            self.assertEqual(first["repeats"], 2000)
            self.assertLessEqual(first["percentile_95_interval"][0],
                                 first["percentile_95_interval"][1])
            metrics["policies"]["fixed_K4"]["group_metrics"].pop("image:3")
            path.write_text(json.dumps(metrics), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different image groups"):
                summarize([path], baseline_method="fixed")


if __name__ == "__main__":
    unittest.main()

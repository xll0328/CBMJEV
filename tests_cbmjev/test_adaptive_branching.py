import unittest
import copy

from tools.analyze_adaptive_branching import (macro_f1_from_confusion, mix_confusions,
                                            percentile, validate_comparison_identity,
                                            select_static_frontier_pair, validate_matched_population)
from tools.compare_policy_traces import compare


class AdaptiveBranchingAnalysisTests(unittest.TestCase):
    def test_every_frontier_candidate_must_use_same_population(self):
        run = {"metrics": {"seed": 17, "source_revision": "r", "source_code_hash": "h"},
               "traces": {"s": {"group_id": "g", "y": 0}}}
        runs = [copy.deepcopy(run) for _ in range(6)]
        validate_matched_population(runs)
        runs[-1]["traces"]["s"]["y"] = 1
        with self.assertRaisesRegex(ValueError, "group/target"):
            validate_matched_population(runs)
        runs[-1]["traces"] = {}
        with self.assertRaisesRegex(ValueError, "same validation samples"):
            validate_matched_population(runs)

    def test_static_envelope_can_use_nonadjacent_budgets(self):
        candidates = [{"directory": str(q), "report": {"mean_queried_groups": q,
                      "accuracy": a}} for q, a in [(0, .2), (1, .3), (2, .4), (4, .9)]]
        low, high = select_static_frontier_pair(candidates, 1.5)
        self.assertEqual((low["directory"], high["directory"]), ("0", "4"))
        self.assertEqual(select_static_frontier_pair(candidates, 4)[1]["directory"], "4")
        with self.assertRaisesRegex(ValueError, "bracket"):
            select_static_frontier_pair(candidates, 5)

    def comparison_runs(self):
        fields = ("schema_hash", "models_receipt_sha256", "models_sha256", "cache_sha256",
                  "static_order_sha256", "responder_checkpoint_sha256", "head_component_sha256",
                  "controller_component_sha256", "training_source_code_hash")
        run = {"metrics": {field: "a" * 64 for field in fields},
               "report": {"method": "static"},
               "settings": {"static": {"config": {"learning": {"hidden": 128},
                             "cost": {"per_group": 1}, "policy": {"max_groups": 2}}}}}
        return [copy.deepcopy(run) for _ in range(3)]

    def test_same_artifacts_allow_different_budgets(self):
        runs = self.comparison_runs()
        runs[2]["settings"]["static"]["config"]["policy"]["max_groups"] = 3
        self.assertEqual(validate_comparison_identity(runs)["models_sha256"], "a" * 64)

    def test_trace_comparison_counts_changes_and_rejects_mismatches(self):
        left, right = self.comparison_runs()[:2]
        for run in (left, right):
            run["metrics"]["seed"] = 17
            run["hashes"] = {}
            run["report"].update(accuracy=1, macro_f1=1, mean_queried_groups=1, mean_calls=1)
            run["traces"] = {"s": {"group_id": "g", "y": 1, "prediction": 1,
                            "final_state": [1], "queried_groups": [0],
                            "steps": [{"action": [0]}]}}
        self.assertEqual(compare(left, right)["differing_samples"]["actions"], 0)
        right["traces"]["s"]["steps"] = []
        self.assertEqual(compare(left, right)["differing_samples"]["actions"], 1)
        right["traces"]["s"]["y"] = 0
        with self.assertRaisesRegex(ValueError, "targets/groups"):
            compare(left, right)
        right["traces"] = {}
        with self.assertRaisesRegex(ValueError, "sample sets"):
            compare(left, right)

    def test_optional_group_interval_preserves_direction_and_caveats(self):
        left, right = self.comparison_runs()[:2]
        for run in (left, right):
            run["metrics"]["seed"] = 17
            run["hashes"] = {}
            run["report"].update(accuracy=1, macro_f1=1, mean_queried_groups=1, mean_calls=1)
            run["traces"] = {"s": {"sample_id": "s", "group_id": "g", "y": 1,
                "prediction": 1, "method": "static", "split": "validation",
                "mode": "offline_replay", "final_state": [1], "queried_groups": [0],
                "queried_atoms": [0], "calls": 1, "declared_cost": 1,
                "steps": [{"action": [0]}]}}
        right["traces"]["s"]["prediction"] = 0
        result = compare(left, right, bootstrap_resamples=10)
        interval = result["paired_group_bootstrap"]
        self.assertEqual(interval["estimates"]["error"]["difference_a_minus_b"], -1)
        self.assertFalse(interval["training_seed_uncertainty_included"])
        self.assertNotIn("group_ids", interval)
        self.assertIn("no selection/multiplicity", result["inference_warning"])
        with self.assertRaises(ValueError):
            compare(left, right, bootstrap_resamples=True)

    def test_changed_artifact_is_not_matched_comparison(self):
        for field in self.comparison_runs()[0]["metrics"]:
            with self.subTest(field=field):
                runs = self.comparison_runs()
                runs[1]["metrics"][field] = "b" * 64
                with self.assertRaisesRegex(ValueError, "artifacts differ"):
                    validate_comparison_identity(runs)

    def test_missing_provenance_is_not_equal_evidence(self):
        runs = self.comparison_runs()
        for run in runs:
            del run["metrics"]["head_component_sha256"]
        with self.assertRaisesRegex(ValueError, "missing/invalid"):
            validate_comparison_identity(runs)

    def test_cost_and_learning_mismatch_rejected(self):
        for field in ("cost", "learning"):
            runs = self.comparison_runs()
            runs[2]["settings"]["static"]["config"][field] = {}
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                validate_comparison_identity(runs)

    def test_mixed_confusion_and_macro_f1(self):
        low = [[8, 2], [4, 6]]
        high = [[9, 1], [2, 8]]
        mixed = mix_confusions(low, high, .25)
        self.assertEqual(mixed, [[8.25, 1.75], [3.5, 6.5]])
        expected = ((2 * 8.25 / (2 * 8.25 + 3.5 + 1.75))
                    + (2 * 6.5 / (2 * 6.5 + 1.75 + 3.5))) / 2
        self.assertAlmostEqual(macro_f1_from_confusion(mixed), expected)

    def test_confusion_shape_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            mix_confusions([[1, 0], [0, 1]], [[1]], .5)

    def test_percentile_is_deterministic_nearest_rank(self):
        values = list(range(1, 101))
        self.assertEqual(percentile(values, .025), 3)
        self.assertEqual(percentile(values, .975), 98)


if __name__ == "__main__":
    unittest.main()

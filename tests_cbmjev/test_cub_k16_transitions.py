from pathlib import Path
import tempfile
import unittest

from cbmjev.io import write_json
from scripts.analyze_cub_k16_transitions import _MATCHED_SYSTEM_KEYS, analyze


def policy(groups):
    count = sum(row["num_samples"] for row in groups.values())
    return {"num_samples": count,
        "accuracy": 1 - sum(row["num_samples"] * row["error"] for row in groups.values()) / count,
        "mean_queried_groups": sum(row["num_samples"] * row["queried_groups"]
                                   for row in groups.values()) / count,
        "group_metrics": groups}


def group(count, error, queries):
    return {"num_samples": count, "error": error, "queried_groups": queries}


class K16TransitionTests(unittest.TestCase):
    def fixture(self, directory, seed, *, changed_fixed=False, changed_singleton=False):
        dynamic = {"one": group(1, 1, 8), "two": group(1, 0, 16),
                   "duplicate": group(2, .5, 10)}
        static = {"one": group(1, 0, 16), "two": group(1, 1, 16),
                  "duplicate": group(2, .5, 16)}
        if changed_fixed:
            static["one"]["queried_groups"] = 15
        if changed_singleton:
            dynamic["new"] = dynamic.pop("one")
            static["new"] = static.pop("one")
        path = Path(directory) / (str(seed) + ".json")
        write_json(path, {"seed": seed, "split": "validation", "mode": "offline_replay",
                          "policies": {"value_K16": policy(dynamic),
                                       "fixed_K16": policy(static)}})
        return path

    def test_singleton_transitions_exclude_ambiguous_duplicate_group(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [self.fixture(directory, seed) for seed in (60, 61)]
            report = analyze(paths)
            self.assertEqual(report["seeds"], [60, 61])
            self.assertEqual(report["complete_singleton_images_across_seeds"], 2)
            for seed in (60, 61):
                item = report["by_seed"][str(seed)]
                self.assertEqual(item["samples"], 4)
                self.assertEqual(item["excluded_multisample_images"], {"duplicate": 2})
                self.assertEqual(item["singleton_transitions"],
                                 {"adaptive_only": 1, "fixed_only": 1})
                self.assertEqual(item["fixed_only_by_adaptive_query_bucket"], {"5-8": 1})
            self.assertEqual(report["fixed_only_frequency_across_seeds"], {0: 1, 2: 1})

    def test_rejects_non_k16_and_cross_seed_image_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = self.fixture(directory, 60)
            bad_budget = self.fixture(directory, 61, changed_fixed=True)
            with self.assertRaisesRegex(ValueError, "exact K16"):
                analyze((valid, bad_budget))
            changed = self.fixture(directory, 62, changed_singleton=True)
            with self.assertRaisesRegex(ValueError, "differ across seeds"):
                analyze((valid, changed))

    def test_cross_file_comparison_requires_same_nested_system(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {key: "same" for key in _MATCHED_SYSTEM_KEYS}
            adaptive = root / "adaptive.json"
            fixed = root / "fixed.json"
            groups_a = {"one": group(1, 1, 8), "duplicate": group(2, .5, 10)}
            groups_f = {"one": group(1, 0, 16), "duplicate": group(2, .5, 16)}
            shared = {"seed": 60, "split": "validation", "mode": "offline_replay",
                      "source_binding": source}
            write_json(adaptive, {**shared, "policies": {"value_K16": policy(groups_a)}})
            write_json(fixed, {**shared, "policies": {"fixed_K16": policy(groups_f)}})
            report = analyze((adaptive,), comparator_paths=(fixed,))
            self.assertEqual(report["by_seed"]["60"]["singleton_transitions"],
                             {"fixed_only": 1})
            changed = root / "changed.json"
            write_json(changed, {**shared, "source_binding": {**source,
                "merged_receipt_sha256": "different"},
                "policies": {"fixed_K16": policy(groups_f)}})
            with self.assertRaisesRegex(ValueError, "same nested system"):
                analyze((adaptive,), comparator_paths=(changed,))


if __name__ == "__main__":
    unittest.main()

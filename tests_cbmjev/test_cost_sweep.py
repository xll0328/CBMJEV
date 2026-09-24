import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.summarize_cost_sweep import write_summary


class CostSweepSummaryTests(unittest.TestCase):
    def run_dir(self, root, name, seed, weight, accuracy):
        directory = root / name
        directory.mkdir()
        metrics = {
            "split": "validation", "mode": "offline_replay", "seed": seed,
            "evidence_status": "OFFLINE_REPLAY_NOT_LATENCY",
            "policies": {"value": {"accuracy": accuracy, "macro_f1": 0.4,
                "group_mean_risk": 1 - accuracy, "mean_queried_groups": 2.0,
                "mean_calls": 1.5, "mean_declared_cost": 2.0,
                "num_samples": 10, "num_groups": 5}},
        }
        settings = {"value": {"method": "value", "mode": "offline_replay",
            "config": {"policy": {"cost_weight": weight, "max_groups": 4}}}}
        (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (directory / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
        return directory

    def test_aggregates_distinct_training_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self.run_dir(root, "a", 1, 0.03, 0.6)
            second = self.run_dir(root, "b", 2, 0.03, 0.8)
            rows, aggregate = write_summary([first, second], root / "out")
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(aggregate[0]["accuracy_mean"], 0.7)
            self.assertEqual(aggregate[0]["max_groups"], 4)
            self.assertIsNotNone(aggregate[0]["accuracy_sd"])
            self.assertFalse(json.loads((root / "out/summary.json").read_text())["paper_claim"])

    def test_rejects_duplicate_seed_at_same_setting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self.run_dir(root, "a", 1, 0.03, 0.6)
            second = self.run_dir(root, "b", 1, 0.03, 0.8)
            with self.assertRaisesRegex(ValueError, "duplicate seed"):
                write_summary([first, second], root / "out")
            self.assertFalse((root / "out").exists())

    def test_source_bytes_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.run_dir(root, "a", 1, 0.03, 0.6)
            rows, _ = write_summary([source], root / "out")
            for filename, key in (("metrics.json", "source_metrics_sha256"),
                                  ("settings.json", "source_settings_sha256")):
                self.assertEqual(rows[0][key], hashlib.sha256((source / filename).read_bytes()).hexdigest())
            self.assertIsNone(rows[0]["schema_hash"])

    def test_mixed_population_or_schema_rejected_before_output_creation(self):
        for field in ("num_samples", "schema_hash"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                a = self.run_dir(root, "a", 1, 0.03, 0.6)
                b = self.run_dir(root, "b", 2, 0.03, 0.8)
                metrics = json.loads((b / "metrics.json").read_text())
                if field == "num_samples":
                    metrics["policies"]["value"][field] = 11
                else:
                    metrics[field] = "a" * 64
                (b / "metrics.json").write_text(json.dumps(metrics))
                with self.assertRaisesRegex(ValueError, "incomparable"):
                    write_summary([a, b], root / "out")
                self.assertFalse((root / "out").exists())

    def test_mixed_training_config_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.run_dir(root, "a", 1, 0.03, 0.6)
            b = self.run_dir(root, "b", 2, 0.03, 0.8)
            settings = json.loads((b / "settings.json").read_text())
            settings["value"]["config"]["learning"] = {"class_weighting": "inverse_frequency"}
            (b / "settings.json").write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError, "comparison_config"):
                write_summary([a, b], root / "out")

    def test_settings_mode_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.run_dir(root, "a", 1, 0.03, 0.6)
            settings = json.loads((a / "settings.json").read_text())
            settings["value"]["mode"] = "live"
            (a / "settings.json").write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError, "mode mismatch"):
                write_summary([a], root / "out")

    def test_mixed_lookahead_depth_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = [self.run_dir(root, str(seed), seed, 0.03, 0.6)
                       for seed in (1, 2)]
            for depth, source in enumerate(sources, start=1):
                settings = json.loads((source / "settings.json").read_text())
                settings["value"]["config"]["evaluation"] = {"lookahead_depth": depth}
                (source / "settings.json").write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError, "comparison_config"):
                write_summary(sources, root / "out")
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()

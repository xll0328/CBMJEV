import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_forced_trajectory_diversity import audit_saved_baselines


class ForcedTrajectoryBaselineProvenanceTests(unittest.TestCase):
    def test_alias_is_bound_to_saved_report_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "seed60"
            source.mkdir()
            metrics_path = source / "metrics.json"
            report_path = root / "trajectory.json"
            group = {"image:1": {"num_samples": 1, "error": 0.0,
                                  "declared_cost": 16.0}}
            metrics = {"seed": 60, "split": "validation", "mode": "offline_replay",
                       "source_binding": {"static_order": {"order": [1, 0]}},
                       "policies": {"fixed_K16": {"num_samples": 1, "accuracy": 1.0,
                                                   "group_metrics": group},
                                    "static_K16": {"num_samples": 1, "accuracy": 1.0,
                                                    "group_metrics": group}}}
            report = {"fixed_policy": "fixed_K16", "fixed_traces": str(source / "traces.jsonl"),
                      "samples": 1, "fixed_accuracy": 1.0}
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            report_path.write_text(json.dumps(report), encoding="utf-8")
            audited = audit_saved_baselines([report_path], [metrics_path])
            self.assertEqual(audited["status"], "HISTORICAL_FIXED_IS_FITTED_STATIC")
            self.assertEqual(audited["rows"][0]["shared_aliased_budgets"], [16])
            self.assertFalse(audited["rows"][0]["raw_fixed_trace_available"])
            self.assertEqual(len(audited["rows"][0]["budget_metrics"]["sha256"]), 64)
            report["fixed_traces"] = str(root / "other" / "traces.jsonl")
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not bound"):
                audit_saved_baselines([report_path], [metrics_path])
            report["fixed_traces"] = str(source / "traces.jsonl")
            report_path.write_text(json.dumps(report), encoding="utf-8")
            metrics["policies"]["static_K16"]["group_metrics"] = {}
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "alias"):
                audit_saved_baselines([report_path], [metrics_path])
            metrics["policies"]["static_K16"]["group_metrics"] = group
            metrics["policies"]["static_K16"]["accuracy"] = 0.0
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "alias"):
                audit_saved_baselines([report_path], [metrics_path])


if __name__ == "__main__":
    unittest.main()

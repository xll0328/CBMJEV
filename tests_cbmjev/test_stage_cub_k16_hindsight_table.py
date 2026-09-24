from pathlib import Path
import tempfile
import unittest

from cbmjev.contracts import stable_hash
from cbmjev.io import write_json
from scripts.figures.stage_cub_k16_hindsight_table import render_table, SEEDS


class StageHindsightTableTests(unittest.TestCase):
    def test_stages_only_complete_matched_reports(self):
        with tempfile.TemporaryDirectory() as root:
            paths = {}
            for seed in SEEDS:
                report = {
                    "format": "cbmjev-k16-last-query-hindsight-headroom-v1",
                    "evidence_status": "LABEL_CLAIRVOYANT_VALIDATION_DIAGNOSTIC_NOT_POLICY",
                    "seed": seed, "split": "validation", "test_evaluated": False,
                    "summary": {"samples": 594, "candidates": 13,
                                "fixed_accuracy": .30,
                                "label_clairvoyant_ce_selected_accuracy": .36,
                                "label_clairvoyant_any_correct_accuracy": .38,
                                "fixed_ce": 2.7, "label_clairvoyant_min_ce": 2.3},
                    "bindings": {"analysis_script_sha256": "same"}}
                report["report_hash"] = stable_hash(report)
                path = Path(root) / f"{seed}.json"
                write_json(path, report)
                paths[seed] = path
            table = render_table(paths)
            self.assertIn("Mean & 30.00 & 36.00 & 2.7000 & 2.3000", table)
            self.assertIn("NOT a deployable policy", table)
            bad = paths[63].read_text().replace('"same"', '"different"')
            paths[63].write_text(bad)
            with self.assertRaises(ValueError):
                render_table(paths)


if __name__ == "__main__":
    unittest.main()

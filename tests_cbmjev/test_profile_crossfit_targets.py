import tempfile
from pathlib import Path
import unittest

from tools.profile_crossfit_targets import profile


class TargetProfileTests(unittest.TestCase):
    def test_full_synthetic_path_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "profile"
            result = profile(out, rows=3, epochs=2, head_rows=8, head_epochs=1,
                             hidden=8, batch_size=16)
            self.assertTrue(result["full_logical_validation_completed"])
            self.assertEqual(result["record_count"], sum(result["records_per_epoch"]))
            self.assertEqual(len(result["records_per_epoch"]), 2)
            self.assertEqual(result["num_atoms"], 312)
            self.assertEqual(result["num_groups"], 28)
            self.assertEqual(result["config"]["device"], "cpu")
            self.assertGreater(result["target_storage_bytes"], 0)
            self.assertEqual(len(result["phases"]), 7)
            self.assertTrue((out / "profile.json").is_file())
            with self.assertRaises(ValueError):
                profile(out, rows=3)

    def test_invalid_shape_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "invalid"
            with self.assertRaises(ValueError):
                profile(out, rows=0)
            self.assertFalse(out.exists())

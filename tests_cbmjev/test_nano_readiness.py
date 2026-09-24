import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cbmjev.nanojev import NanoCandidateScorer
from tests_cbmjev.test_responders import FakeBackbone, FakeTokenizer
from tools.check_nano_backbone import main


class NanoReadinessTests(unittest.TestCase):
    def test_local_readiness_report_is_not_performance_evidence(self):
        scorer = NanoCandidateScorer(FakeBackbone(), FakeTokenizer(), max_length=512)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "readiness.json"
            with patch("tools.check_nano_backbone.load_local_nano", return_value=scorer), \
                    patch("sys.argv", ["check", "--backbone", folder, "--out", str(out)]):
                main()
                report = json.loads(out.read_text())
                self.assertFalse(report["paper_evidence"])
                self.assertEqual(set(report["batch_padding_checks"]), {"left", "right"})
                self.assertTrue(report["frozen_backbone_gradient_check"])
                with self.assertRaises(FileExistsError):
                    main()


if __name__ == "__main__":
    unittest.main()

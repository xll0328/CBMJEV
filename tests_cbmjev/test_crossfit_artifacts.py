import copy
import tempfile
import unittest
from pathlib import Path

import torch

from cbmjev.crossfit_artifacts import save_crossfit_head, load_crossfit_head
from cbmjev.contracts import stable_hash
from cbmjev.crossfit_training import fit_head_only
from tests_cbmjev.test_crossfit_training import CONFIG, schema_fixture, rows_fixture, oof_sources


class CrossfitArtifactTests(unittest.TestCase):
    def setUp(self):
        self.schema = schema_fixture()
        rows = rows_fixture("head", 6, 5)
        sources, assignments = oof_sources(rows, "responder")
        self.head, self.report = fit_head_only(
            rows, self.schema, CONFIG, response_artifact_by_group=assignments,
            provenance_records=sources)

    def test_roundtrip_preserves_predictions_receipt_and_frozen_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_crossfit_head(tmp, self.head, self.schema, self.report)
            head, report = load_crossfit_head(tmp, self.schema, device="cpu:0")
            self.assertEqual(head.config["device"], "cpu")
            self.assertEqual(str(head.device), "cpu:0")
            self.assertEqual(stable_hash(report), stable_hash(self.report))
            self.assertEqual(head.probabilities((-1, -1, -1)),
                             self.head.probabilities((-1, -1, -1)))
            self.assertFalse(any(p.requires_grad for p in head.network.parameters()))

    def test_corrupt_artifact_bytes_rejected(self):
        for filename in ("head.pt", "head_report.json", "receipt.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                save_crossfit_head(tmp, self.head, self.schema, self.report)
                path = Path(tmp) / filename
                if filename == "receipt.json":
                    path.write_text(path.read_text().replace("checkpoint_sha256", "changed_sha256"))
                else:
                    with path.open("ab") as stream:
                        stream.write(b" ")
                with self.assertRaises(ValueError):
                    load_crossfit_head(tmp, self.schema)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_crossfit_head(tmp, self.head, self.schema, self.report)
            before = (Path(tmp) / "head.pt").read_bytes()
            with self.assertRaises(ValueError):
                save_crossfit_head(tmp, self.head, self.schema, self.report)
            self.assertEqual(before, (Path(tmp) / "head.pt").read_bytes())

    def test_stale_report_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = copy.deepcopy(self.report)
            report["fit_rows_sha256"] = "invalid"
            out = Path(tmp) / "new"
            with self.assertRaises(ValueError):
                save_crossfit_head(out, self.head, self.schema, report)
            self.assertFalse(out.exists())

    def test_modified_model_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with torch.no_grad():
                next(self.head.network.parameters()).add_(1)
            with self.assertRaises(ValueError):
                save_crossfit_head(tmp, self.head, self.schema, self.report)


if __name__ == "__main__":
    unittest.main()

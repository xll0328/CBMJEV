"""Tiny real nested training chains; these are engineering tests, not results."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cbmjev.contracts import stable_hash
from cbmjev.crossfit import plan_crossfit_prepared
from cbmjev.crossfit_artifacts import load_crossfit_head
from cbmjev.crossfit_executor import execute_outer_fold
from cbmjev.crossfit_merge import load_outer_folds, merge_outer_folds, load_merged_controller
from cbmjev.io import read_json
from cbmjev.pipeline import load_prepared
from tests_cbmjev.test_crossfit import fixture, write_json, write_jsonl, load_rows


class CrossfitMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.prepared = fixture(cls.root)
        rows = load_rows(cls.prepared / "samples.jsonl")
        rows[0]["target"]["value"] = 1
        rows[1]["target"]["value"] = 0
        write_jsonl(cls.prepared / "samples.jsonl", rows)
        audit = read_json(cls.prepared / "audit.json")
        audit["source_revision"] = "SYNTHETIC_MERGE_TEST"
        write_json(cls.prepared / "audit.json", audit)
        cls.plan = cls.root / "plan"
        plan_crossfit_prepared(cls.prepared, cls.plan, inner_folds=2)
        cls.config = {"seed": 29, "learning": {"objective": "value", "hidden": 8,
            "head_epochs": 1, "policy_epochs": 1, "batch_size": 4, "masks_per_sample": 2}}
        cls.dirs = []
        for fold in range(3):
            directory = cls.root / ("outer%d" % fold)
            execute_outer_fold(cls.prepared, cls.plan, directory, outer_fold=fold,
                config=cls.config, responder_options={"epochs": 1, "batch_size": 4})
            cls.dirs.append(directory)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_roundtrip_and_final_head(self):
        out = self.root / "merged"
        receipt = merge_outer_folds(self.prepared, self.plan, reversed(self.dirs), out)
        self.assertEqual(receipt["status"], "COMPLETE")
        schema, data = load_outer_folds(self.prepared, self.plan, self.dirs)
        controller, report = load_merged_controller(out, schema)
        _, head_report = load_crossfit_head(out / "head", schema)
        samples = sorted(r["sample_id"] for r in data["rows"])
        self.assertEqual(report["target_sample_ids"], samples)
        self.assertEqual(head_report["fit_sample_ids"], samples)
        self.assertEqual(report["policy_examples_per_epoch"],
                         [sum(p["record_count"] for p in data["packages"])])
        self.assertEqual(controller.predict((-1,) * schema.num_atoms, [()]), (0.0,))
        with self.assertRaisesRegex(ValueError, "new"):
            merge_outer_folds(self.prepared, self.plan, self.dirs, out)

    def test_missing_duplicate_folds_rejected_before_output(self):
        for dirs in (self.dirs[:2], self.dirs[:2] + self.dirs[:1]):
            out = self.root / "invalid"
            with self.assertRaisesRegex(ValueError, "distinct"):
                merge_outer_folds(self.prepared, self.plan, dirs, out)
            self.assertFalse(out.exists())

    def test_rejects_different_training_source(self):
        path = self.dirs[0] / "receipt.json"
        original_bytes = path.read_bytes()
        original = read_json(path)
        changed = {**original, "source_code_hash": "OTHER_SOURCE"}
        changed.pop("receipt_hash")
        changed["receipt_hash"] = stable_hash(changed)
        try:
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, "different source"):
                load_outer_folds(self.prepared, self.plan, self.dirs)
        finally:
            path.write_bytes(original_bytes)

    def test_rejects_changed_target_bytes(self):
        path = self.dirs[0] / "action_targets.json"
        original_bytes = path.read_bytes()
        original = read_json(path)
        changed = dict(original)
        changed["metadata"]["record_count"] += 1
        try:
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, "files changed"):
                load_outer_folds(self.prepared, self.plan, self.dirs)
        finally:
            path.write_bytes(original_bytes)

    def test_no_completion_on_training_failure(self):
        out = self.root / "failed"
        with patch("cbmjev.crossfit_merge.fit_controller_from_targets",
                   side_effect=RuntimeError("failed fit")):
            with self.assertRaisesRegex(RuntimeError, "failed fit"):
                merge_outer_folds(self.prepared, self.plan, self.dirs, out)
        self.assertFalse((out / "receipt.json").exists())

    def test_source_change_prevents_completion(self):
        out = self.root / "source_changed"
        with patch("cbmjev.crossfit_merge.code_fingerprint", side_effect=["before", "after"]):
            with self.assertRaisesRegex(ValueError, "sources changed"):
                merge_outer_folds(self.prepared, self.plan, self.dirs, out, fit_final_head=False)
        self.assertFalse((out / "receipt.json").exists())

    def test_tensor_identical_heads_with_distinct_ancestry_fail_closed(self):
        root = self.root / "symmetric"
        root.mkdir()
        prepared = fixture(root)
        audit = read_json(prepared / "audit.json")
        audit["source_revision"] = "SYNTHETIC_SYMMETRIC_MERGE_TEST"
        write_json(prepared / "audit.json", audit)
        plan = root / "plan"
        plan_crossfit_prepared(prepared, plan, inner_folds=2)
        dirs = []
        for fold in range(3):
            directory = root / ("outer%d" % fold)
            execute_outer_fold(prepared, plan, directory, outer_fold=fold,
                config=self.config, responder_options={"epochs": 1, "batch_size": 4})
            dirs.append(directory)
        out = root / "merged"
        with self.assertRaisesRegex(ValueError, "conflicting provenance"):
            merge_outer_folds(prepared, plan, dirs, out)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()

"""Synthetic text-backend execution, not empirical research evidence."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cbmjev.config import learning_config, resolve_config
from cbmjev.contracts import stable_hash
from cbmjev.crossfit import plan_crossfit_prepared
from cbmjev.crossfit_artifacts import load_crossfit_head
from cbmjev.crossfit_cache import load_crossfit_cache
from cbmjev.crossfit_executor import execute_outer_fold
from cbmjev.crossfit_training import construct_action_targets, _validate_target_package
from cbmjev.io import file_hash, read_json
from cbmjev.learning import normalize_config
from cbmjev.pipeline import load_prepared
from cbmjev.provenance import validate_target_exclusion
from tests_cbmjev.test_crossfit import fixture, write_json


class OuterFoldExecutorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = fixture(self.root)
        audit = read_json(self.prepared / "audit.json")
        audit["source_revision"] = "SYNTHETIC_OUTER_EXECUTOR_FIXTURE"
        write_json(self.prepared / "audit.json", audit)
        self.plan, self.out = self.root / "plan", self.root / "execution"
        plan_crossfit_prepared(self.prepared, self.plan, inner_folds=2)
        self.config = {"seed": 29, "learning": {"objective": "value", "hidden": 8,
                       "head_epochs": 1, "policy_epochs": 1, "batch_size": 4,
                       "masks_per_sample": 2}}

    def execute(self, **kwargs):
        return execute_outer_fold(self.prepared, self.plan, self.out,
                                  config=self.config, responder_options={"epochs": 1,
                                  "batch_size": 4}, **kwargs)

    def test_real_nested_outer_fold_roundtrip(self):
        receipt = self.execute(outer_fold=0)
        self.assertEqual(receipt, read_json(self.out / "receipt.json"))
        unsigned = dict(receipt)
        self.assertEqual(unsigned.pop("receipt_hash"), stable_hash(unsigned))
        self.assertEqual(receipt["scope"], "SINGLE_OUTER_FOLD_NOT_FINAL_DEPLOYMENT")
        for name, digest in receipt["files_sha256"].items():
            self.assertEqual(file_hash(self.out / name), digest)
        for name in ("outer", "inner_000", "inner_001"):
            responder_receipt = read_json(self.out / name / "responder" / "receipt.json")
            self.assertEqual(responder_receipt["seed"], 29)
        schema, _, _ = load_prepared(self.prepared)
        head, report = load_crossfit_head(self.out / "head", schema)
        _, rows, manifest = load_crossfit_cache(self.out / "outer" / "cache",
            self.prepared, self.plan, self.out / "outer" / "responder")
        target_groups = sorted({row["group_id"] for row in rows})
        validate_target_exclusion(report["provenance"],
            artifact_ids=[report["head_artifact_id"]], target_group_ids=target_groups)
        for ancestor in report["provenance"]:
            self.assertFalse(set(ancestor["supervised_group_ids"]) & set(target_groups))
        cfg = normalize_config(learning_config(resolve_config(self.config)), schema)
        targets = construct_action_targets(rows, head, schema, cfg, head_report=report,
            response_artifact_by_group=manifest["response_artifact_by_group"],
            provenance_records=manifest["provenance"])
        from cbmjev.crossfit_targets import open_target_package
        stored = open_target_package(self.out / "action_targets.json")
        stored["records"] = list(stored["records"])
        self.assertEqual(targets, stored)
        _validate_target_package(targets, schema, cfg)
        self.assertTrue(targets["records"])
        self.assertEqual(target_groups, read_json(self.plan / "plan.json")["folds"][0]["target_group_ids"])
        with self.assertRaisesRegex(ValueError, "new"):
            self.execute(outer_fold=0)
        self.assertEqual(file_hash(self.out / "action_targets.json"),
                         receipt["files_sha256"]["action_targets.json"])

    def test_invalid_fold_has_no_side_effects(self):
        for fold in (-1, 3, True, None, "0"):
            with self.subTest(fold=fold), self.assertRaises(ValueError):
                self.execute(outer_fold=fold)
            self.assertFalse(self.out.exists())

    def test_invalid_options_have_no_side_effects(self):
        with self.assertRaises(ValueError):
            execute_outer_fold(self.prepared, self.plan, self.out, outer_fold=0,
                               config=self.config, responder_options={"final": True})
        self.assertFalse(self.out.exists())

    def test_failed_training_has_no_completion_receipt(self):
        with patch("cbmjev.crossfit_executor.train_crossfit_responder",
                   side_effect=RuntimeError("training failure")):
            with self.assertRaisesRegex(RuntimeError, "training failure"):
                self.execute(outer_fold=0)
        self.assertTrue(self.out.exists())
        self.assertFalse((self.out / "receipt.json").exists())

    def test_source_change_prevents_completion(self):
        with patch("cbmjev.crossfit_executor.code_fingerprint",
                   side_effect=["source-before", "source-after"]):
            with self.assertRaisesRegex(ValueError, "sources changed"):
                self.execute(outer_fold=0)
        self.assertTrue((self.out / "action_targets.json").exists())
        self.assertFalse((self.out / "receipt.json").exists())


if __name__ == "__main__":
    unittest.main()

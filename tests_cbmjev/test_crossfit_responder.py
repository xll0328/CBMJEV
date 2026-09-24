"""Real tiny hashing training exercises the verified group-only fit boundary."""
import tempfile
import unittest
from pathlib import Path

from cbmjev.contracts import stable_hash
from cbmjev.crossfit import plan_crossfit_prepared
from cbmjev.io import file_hash, read_json
from cbmjev.pipeline import train_crossfit_responder, train_responder
from tests_cbmjev.test_crossfit import fixture, write_json


class CrossfitResponderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = fixture(self.root)
        audit = read_json(self.prepared / "audit.json")
        audit["source_revision"] = "SYNTHETIC_CROSSFIT_TRAINING_FIXTURE"
        write_json(self.prepared / "audit.json", audit)
        self.planned = self.root / "plan"
        plan_crossfit_prepared(self.prepared, self.planned, inner_folds=2)
        self.plan = read_json(self.planned / "plan.json")

    def test_outer_inner_final_train_exact_groups_and_preserve_prepared(self):
        original = {p.name: file_hash(p) for p in self.prepared.iterdir()}
        stages = [("outer", {"outer_fold": 0}, self.plan["folds"][0]["fit_group_ids"]),
                  ("inner", {"outer_fold": 0, "inner_fold": 1},
                   self.plan["folds"][0]["inner_folds"][1]["fit_group_ids"]),
                  ("final", {"final": True}, self.plan["eligible_group_ids"])]
        for stage, selectors, groups in stages:
            with self.subTest(stage=stage):
                out = self.root / stage
                result = train_crossfit_responder(self.prepared, self.planned, out,
                                                  epochs=1, batch_size=4, **selectors)
                receipt = read_json(out / "receipt.json")
                self.assertEqual(receipt["supervised_group_ids"], groups)
                self.assertEqual(receipt["crossfit"]["stage"], stage)
                self.assertEqual(receipt["crossfit"]["plan_hash"], self.plan["plan_hash"])
                self.assertEqual(receipt["crossfit"]["prepared_files_sha256"], original)
                self.assertEqual(len(receipt["supervised_sample_ids"]), result["fitted_rows"])
                self.assertEqual(receipt["initialization"]["kind"], "random")
                self.assertEqual(file_hash(out / "responder.pt"), receipt["checkpoint_sha256"])
                self.assertTrue(set(groups).isdisjoint({"g12", "g13", "g14"}))
        self.assertEqual(original, {p.name: file_hash(p) for p in self.prepared.iterdir()})

    def test_invalid_selectors_fail_before_output(self):
        for selector in ({}, {"outer_fold": -1}, {"outer_fold": True},
                         {"outer_fold": 3}, {"inner_fold": 0},
                         {"outer_fold": 0, "inner_fold": 2},
                         {"final": True, "outer_fold": 0}, {"final": 1}):
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                train_crossfit_responder(self.prepared, self.planned,
                                         self.root / "invalid", **selector)
        self.assertFalse((self.root / "invalid").exists())

    def test_rehashed_validation_injection_rejected(self):
        self.plan["folds"][0]["fit_group_ids"].append("g12")
        self.plan.pop("plan_hash")
        self.plan["plan_hash"] = stable_hash(self.plan)
        write_json(self.planned / "plan.json", self.plan)
        with self.assertRaisesRegex(ValueError, "deterministic"):
            train_crossfit_responder(self.prepared, self.planned,
                                     self.root / "invalid", outer_fold=0)
        self.assertFalse((self.root / "invalid").exists())

    def test_arbitrary_groups_cannot_override_selection(self):
        with self.assertRaises(TypeError):
            train_crossfit_responder(self.prepared, self.planned, self.root / "invalid",
                                     outer_fold=0, fit_group_ids=["g12"])
        self.assertFalse((self.root / "invalid").exists())

    def test_legacy_entry_keeps_disjoint_role_training(self):
        out = self.root / "legacy"
        train_responder(self.prepared, out, epochs=1, batch_size=4)
        receipt = read_json(out / "receipt.json")
        self.assertEqual(receipt["supervised_group_ids"], ["g00", "g03", "g06", "g09"])
        self.assertNotIn("crossfit", receipt)


if __name__ == "__main__":
    unittest.main()

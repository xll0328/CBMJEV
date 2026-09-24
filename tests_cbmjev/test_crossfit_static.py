"""OOF global-prefix ranking: real persistence plus explicit pooled-gain tests."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cbmjev.contracts import Concept, QueryGroup, Schema, stable_hash
from cbmjev.baselines import fit_static_order
from cbmjev.crossfit import plan_crossfit_prepared
from cbmjev.crossfit_executor import execute_outer_fold
from cbmjev.crossfit_static import _greedy_order, fit_crossfit_static_order, load_crossfit_static_order
from cbmjev.io import file_hash, read_json
from cbmjev.provenance import validate_target_exclusion
from tests_cbmjev.test_crossfit import fixture, load_rows, write_json, write_jsonl


class CrossfitStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.prepared = fixture(cls.root)
        rows = load_rows(cls.prepared / "samples.jsonl")
        # Break the tensor-identical-head/unequal-ancestry symmetry deliberately.
        rows[0]["target"]["value"] = 1
        rows[1]["target"]["value"] = 0
        write_jsonl(cls.prepared / "samples.jsonl", rows)
        audit = read_json(cls.prepared / "audit.json")
        audit["source_revision"] = "SYNTHETIC_STATIC_OOF_TEST"
        write_json(cls.prepared / "audit.json", audit)
        cls.plan = cls.root / "plan"
        plan_crossfit_prepared(cls.prepared, cls.plan, inner_folds=2)
        cls.dirs = []
        for fold in range(3):
            out = cls.root / ("outer%d" % fold)
            execute_outer_fold(cls.prepared, cls.plan, out, outer_fold=fold,
                config={"seed": 29, "learning": {"objective": "value", "hidden": 8,
                    "head_epochs": 1, "policy_epochs": 1, "batch_size": 4,
                    "masks_per_sample": 2}},
                responder_options={"epochs": 1, "batch_size": 4})
            cls.dirs.append(out)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_persistence_ancestry_and_input_order(self):
        out = self.root / "static"
        order, report = fit_crossfit_static_order(self.prepared, self.plan, self.dirs, out)
        reverse_order, reverse_report = fit_crossfit_static_order(
            self.prepared, self.plan, reversed(self.dirs), self.root / "reversed")
        self.assertEqual(order, (0,))
        self.assertEqual(order, reverse_order)
        self.assertEqual(report, reverse_report)
        self.assertEqual(report, read_json(out / "static_order.json"))
        self.assertEqual((order, report), load_crossfit_static_order(
            out, self.prepared, self.plan, reversed(self.dirs)))
        self.assertEqual(report["rows_used"], 12)
        self.assertEqual(len(report["sources"]), 3)
        self.assertFalse(report["evaluation_holdout_labels_used"])
        self.assertTrue(report["outer_heldout_training_labels_used"])
        self.assertFalse(report["final_head_used_for_ranking"])
        unsigned = {k: v for k, v in report.items() if k != "report_hash"}
        self.assertEqual(report["report_hash"], stable_hash(unsigned))
        receipt = read_json(out / "receipt.json")
        self.assertEqual(receipt["static_order_sha256"], file_hash(out / "static_order.json"))
        self.assertEqual(receipt["receipt_hash"], stable_hash(
            {k: v for k, v in receipt.items() if k != "receipt_hash"}))
        validate_target_exclusion(report["provenance"],
            artifact_ids=[report["static_order_artifact_id"]],
            target_group_ids=["g12", "g13", "g14"])
        with self.assertRaises(ValueError):
            validate_target_exclusion(report["provenance"],
                artifact_ids=[report["static_order_artifact_id"]], target_group_ids=["g00"])
        for source in report["sources"]:
            validate_target_exclusion(report["provenance"],
                artifact_ids=[source["head_artifact_id"], source["responder_artifact_id"]],
                target_group_ids=source["target_group_ids"])
        with self.assertRaisesRegex(ValueError, "new"):
            fit_crossfit_static_order(self.prepared, self.plan, self.dirs, out)

    def test_invalid_batch_and_missing_fold_before_output(self):
        for batch in (True, 0, -1, 1.5):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                fit_crossfit_static_order(self.prepared, self.plan, self.dirs,
                    self.root / "invalid_batch", batch_size=batch)
        self.assertFalse((self.root / "invalid_batch").exists())
        with self.assertRaisesRegex(ValueError, "distinct"):
            fit_crossfit_static_order(self.prepared, self.plan, self.dirs[:2], self.root / "missing")
        self.assertFalse((self.root / "missing").exists())

    def test_changed_bytes_and_source_change_prevent_completion(self):
        path = self.dirs[0] / "action_targets.json"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "files changed"):
                fit_crossfit_static_order(self.prepared, self.plan, self.dirs, self.root / "changed")
        finally:
            path.write_bytes(original)
        self.assertFalse((self.root / "changed/receipt.json").exists())
        with patch("cbmjev.crossfit_static.code_fingerprint", side_effect=["before", "after"]):
            with self.assertRaisesRegex(ValueError, "sources changed"):
                fit_crossfit_static_order(self.prepared, self.plan, self.dirs, self.root / "race")
        self.assertFalse((self.root / "race/receipt.json").exists())

    def test_loader_rejects_semantic_and_source_tampering_even_with_rehashed_files(self):
        out = self.root / "tamper"
        fit_crossfit_static_order(self.prepared, self.plan, self.dirs, out)
        original_report = (out / "static_order.json").read_bytes()
        original_receipt = (out / "receipt.json").read_bytes()
        for field, value, message in (("order", [True], "permutation"),
                ("sources", [], "source binding"),
                ("order_group_ids", ["wrong"], "permutation"),
                ("stages", [], "stages"),
                ("source_code_hash", "unknown", "fingerprint")):
            try:
                report = read_json(out / "static_order.json")
                report[field] = value
                core = {k: v for k, v in report.items()
                        if k not in ("report_hash", "static_order_artifact_id", "provenance")}
                report["static_order_artifact_id"] = "static_order:" + stable_hash(core)
                report.pop("report_hash")
                report["report_hash"] = stable_hash(report)
                write_json(out / "static_order.json", report)
                receipt = read_json(out / "receipt.json")
                receipt.update(static_order_artifact_id=report["static_order_artifact_id"],
                    report_hash=report["report_hash"], static_order_sha256=file_hash(out / "static_order.json"))
                receipt.pop("receipt_hash")
                receipt["receipt_hash"] = stable_hash(receipt)
                write_json(out / "receipt.json", receipt)
                with self.assertRaisesRegex(ValueError, message):
                    load_crossfit_static_order(out, self.prepared, self.plan, self.dirs)
            finally:
                (out / "static_order.json").write_bytes(original_report)
                (out / "receipt.json").write_bytes(original_receipt)

    def test_loader_rechecks_original_outer_files(self):
        out = self.root / "reload_source"
        fit_crossfit_static_order(self.prepared, self.plan, self.dirs, out)
        path = self.dirs[0] / "action_targets.json"
        before = path.read_bytes()
        try:
            path.write_bytes(before + b" ")
            with self.assertRaisesRegex(ValueError, "files changed"):
                load_crossfit_static_order(out, self.prepared, self.plan, self.dirs)
        finally:
            path.write_bytes(before)


class PooledGreedyTests(unittest.TestCase):
    def setUp(self):
        self.schema = Schema("synthetic", 2,
            (Concept("a", "A", ("n", "y")), Concept("b", "B", ("n", "y"))),
            (QueryGroup("a", (0,)), QueryGroup("b", (1,))))

    def test_row_weighting_retains_each_excluding_head(self):
        class Head:
            def __init__(self, response, gains):
                self.response, self.gains = response, gains
                self.seen = []

            def probabilities_many(self, states):
                for state in states:
                    # A head cannot accidentally receive the other fold's rows.
                    assert all(value in (-1, self.response) for value in state)
                    self.seen.append(state)
                    loss = 3 - sum(gain for gain, value in zip(self.gains, state) if value != -1)
                    p = math.exp(-loss)
                    yield (p, 1 - p)

        first, second = Head(0, (0.9, 0)), Head(1, (0, 0.5))
        folds = [(first, [{"z": [0, 0], "y": 0}]),
                 (second, [{"z": [1, 1], "y": 0} for _ in range(3)])]
        order, stages = _greedy_order(folds, self.schema, 2)
        # Equal fold weights incorrectly select a; row weights select b.
        self.assertEqual(order, (1, 0))
        self.assertAlmostEqual(stages[0]["candidate_gains"]["0"], 0.225)
        self.assertAlmostEqual(stages[0]["candidate_gains"]["1"], 0.375)
        self.assertEqual(stages[0]["candidate_group_ids"], {"0": "a", "1": "b"})
        self.assertTrue(first.seen and second.seen)

    def test_exact_ties_and_probability_floor(self):
        class ConstantHead:
            def probabilities_many(self, states):
                return [(0.0, 1.0) for _ in states]

        order, stages = _greedy_order(
            [(ConstantHead(), [{"z": [0, 1], "y": 0}])], self.schema, 64)
        self.assertEqual(order, (0, 1))
        self.assertEqual(stages[0]["candidate_gains"], {"0": 0.0, "1": 0.0})

    def test_same_head_matches_existing_static_scoring(self):
        class Head:
            def probabilities_many(self, states):
                return [(0.5 + 0.1 * (s[0] != -1) + 0.2 * (s[1] != -1),
                         0.5 - 0.1 * (s[0] != -1) - 0.2 * (s[1] != -1)) for s in states]

        rows = [{"z": [0, 1], "y": 0, "split": "policy_fit", "sample_id": "s0", "group_id": "g0"},
                {"z": [1, 0], "y": 1, "split": "policy_fit", "sample_id": "s1", "group_id": "g1"}]
        head = Head()
        order, stages = _greedy_order([(head, rows)], self.schema, 2)
        legacy_order, legacy = fit_static_order(rows, head, self.schema, {"batch_size": 2})
        self.assertEqual(order, legacy_order)
        self.assertEqual([s["candidate_gains"] for s in stages],
                         [s["candidate_gains"] for s in legacy["stages"]])


if __name__ == "__main__":
    unittest.main()

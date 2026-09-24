"""End-to-end nested validation replay; synthetic engineering evidence only."""
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from cbmjev.crossfit import plan_crossfit_prepared
from cbmjev.crossfit_cache import cache_crossfit_responses
from cbmjev.crossfit_evaluation import evaluate_crossfit_validation
from cbmjev.crossfit_executor import execute_outer_fold
from cbmjev.crossfit_merge import merge_outer_folds
from cbmjev.crossfit_static import fit_crossfit_static_order
from cbmjev.io import read_json, read_jsonl
from cbmjev.pipeline import train_crossfit_responder
from cbmjev.provenance import make_fit_record
from cbmjev.runtime import run_episode
from scripts.evaluate_crossfit_minstop_grid import main as minstop_grid_main
from tests_cbmjev.test_crossfit import fixture, write_json, write_jsonl, load_rows


class CrossfitEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.prepared = fixture(cls.root)
        rows = load_rows(cls.prepared / "samples.jsonl")
        rows[0]["target"]["value"], rows[1]["target"]["value"] = 1, 0
        write_jsonl(cls.prepared / "samples.jsonl", rows)
        audit = read_json(cls.prepared / "audit.json")
        audit["source_revision"] = "SYNTHETIC_EVALUATOR_TEST"
        write_json(cls.prepared / "audit.json", audit)
        cls.plan = cls.root / "plan"
        plan_crossfit_prepared(cls.prepared, cls.plan, inner_folds=2)
        config = {"seed": 29, "learning": {"objective": "value", "hidden": 8,
            "head_epochs": 1, "policy_epochs": 1, "batch_size": 4, "masks_per_sample": 2}}
        cls.outers = []
        for fold in range(3):
            directory = cls.root / ("outer%d" % fold)
            execute_outer_fold(cls.prepared, cls.plan, directory, outer_fold=fold,
                config=config, responder_options={"epochs": 1, "batch_size": 4})
            cls.outers.append(directory)
        cls.merged = cls.root / "merged"
        merge_outer_folds(cls.prepared, cls.plan, cls.outers, cls.merged)
        cls.responder = cls.root / "final"
        train_crossfit_responder(cls.prepared, cls.plan, cls.responder, final=True,
                                 seed=29, epochs=1, batch_size=4)
        cls.cache = cls.root / "cache"
        cache_crossfit_responses(cls.prepared, cls.plan, cls.responder, cls.cache, split="validation")
        cls.calibration = cls.root / "calibration"
        cache_crossfit_responses(cls.prepared, cls.plan, cls.responder, cls.calibration, split="calibration")
        cls.static = cls.root / "static"
        fit_crossfit_static_order(cls.prepared, cls.plan, cls.outers, cls.static)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def evaluate(self, name, **kwargs):
        return evaluate_crossfit_validation(self.prepared, self.plan, self.merged,
            self.responder, self.cache, self.root / name, **kwargs)

    def test_complete_replay_and_no_labels_in_inference(self):
        def checked(env, schema, head, **kwargs):
            self.assertNotIn("y", kwargs)
            self.assertNotIn("sample_id", kwargs)
            trace = run_episode(env, schema, head, **kwargs)
            self.assertNotIn("y", trace)
            return trace
        with patch("cbmjev.crossfit_evaluation.run_episode", side_effect=checked):
            receipt = self.evaluate("eval")
        self.assertEqual(receipt["status"], "COMPLETE")
        self.assertFalse(receipt["paper_evidence"])
        self.assertEqual(receipt["samples_per_policy"], 1)
        self.assertEqual(receipt["num_policies"], 6)
        rows = read_jsonl(self.root / "eval/traces.jsonl")
        self.assertEqual({r["split"] for r in rows}, {"validation"})
        self.assertEqual({r["mode"] for r in rows}, {"offline_replay"})
        self.assertEqual(len(receipt["source_binding"]["ancestry_exclusion"]["artifact_ids"]), 3)
        with self.assertRaisesRegex(ValueError, "new"):
            self.evaluate("eval")

    def test_rejects_calibration_and_outer_cache(self):
        for label, responder, cache in (("cal", self.responder, self.calibration),
                ("outer", self.outers[0] / "outer/responder", self.outers[0] / "outer/cache")):
            out = self.root / ("bad_" + label)
            with self.assertRaisesRegex(ValueError, "final validation"):
                evaluate_crossfit_validation(self.prepared, self.plan, self.merged,
                                            responder, cache, out)
            self.assertFalse(out.exists())

    def test_static_order_used_and_bound_with_ancestry_and_cost_semantics(self):
        calls = []
        def checked(*args, **kwargs):
            calls.append((kwargs["method"], kwargs["order"]))
            return run_episode(*args, **kwargs)
        with patch("cbmjev.crossfit_evaluation.run_episode", side_effect=checked):
            receipt = self.evaluate("static_eval", methods=["static", "static_value", "fixed"],
                                    static_order_dir=self.static)
        self.assertEqual(calls, [("static", [0]), ("static_value", [0]), ("fixed", None)])
        binding = receipt["source_binding"]
        self.assertEqual(len(binding["ancestry_exclusion"]["artifact_ids"]), 4)
        self.assertIn(binding["static_order"]["artifact_id"], binding["ancestry_exclusion"]["artifact_ids"])
        self.assertIn("cost_assumptions", binding["static_order"])
        self.assertEqual(binding["static_order"]["loss_definition"],
                         "unweighted_CE_independent_of_head_training_weighting")
        metrics = read_json(self.root / "static_eval/metrics.json")
        self.assertEqual(metrics["policies"]["static"]["mean_queried_groups"], 1)

    def test_static_report_change_during_inference_prevents_completion(self):
        path = self.static / "static_order.json"
        before = path.read_bytes()
        def changed(*args, **kwargs):
            result = run_episode(*args, **kwargs)
            write_json(path, {"changed": True})
            return result
        try:
            with patch("cbmjev.crossfit_evaluation.run_episode", side_effect=changed):
                with self.assertRaisesRegex(ValueError, "report hash"):
                    self.evaluate("changed_static", methods=["static"], static_order_dir=self.static)
            self.assertFalse((self.root / "changed_static/receipt.json").exists())
        finally:
            path.write_bytes(before)

    def test_changed_outer_source_rejected(self):
        path = self.outers[0] / "action_targets.json"
        before = path.read_bytes()
        try:
            write_json(path, {"changed": True})
            with self.assertRaisesRegex(ValueError, "files changed"):
                self.evaluate("bad_outer")
            self.assertFalse((self.root / "bad_outer").exists())
        finally:
            path.write_bytes(before)

    def test_outer_receipt_or_plan_drift_prevents_completion(self):
        for index, path in enumerate((self.outers[0] / "receipt.json",
                                      self.plan / "fold_manifest.jsonl")):
            with self.subTest(path=path):
                before = path.read_bytes()
                out_name = "changed_outer_plan_during_eval_%d" % index

                def changed(*args, **kwargs):
                    trace = run_episode(*args, **kwargs)
                    with path.open("ab") as handle:
                        handle.write(b"tamper")
                    return trace

                try:
                    with patch("cbmjev.crossfit_evaluation.run_episode", side_effect=changed):
                        with self.assertRaisesRegex(ValueError, "outer (receipt|plan) changed"):
                            self.evaluate(out_name, methods=["stop"])
                    self.assertFalse((self.root / out_name / "receipt.json").exists())
                finally:
                    path.write_bytes(before)

    def test_changed_component_rejected(self):
        path = self.merged / "controller_report.json"
        before = path.read_bytes()
        try:
            write_json(path, {"changed": True})
            with self.assertRaisesRegex(ValueError, "files mismatch"):
                self.evaluate("bad_component")
        finally:
            path.write_bytes(before)

    def test_no_receipt_if_code_changes(self):
        with patch("cbmjev.crossfit_evaluation.code_fingerprint", side_effect=["before", "after"]):
            with self.assertRaisesRegex(ValueError, "sources changed"):
                self.evaluate("changed_code", methods=["stop"])
        self.assertFalse((self.root / "changed_code/receipt.json").exists())

    def test_transitive_final_responder_leakage_rejected(self):
        path = self.responder / "receipt.json"
        before = path.read_bytes()
        receipt = read_json(path)
        producer = next(r for r in receipt["provenance"] if r["artifact_id"] == receipt["artifact_id"])
        leaked = make_fit_record("leaked-parent", supervised_group_ids=["g12"])
        changed = make_fit_record(producer["artifact_id"],
            supervised_group_ids=producer["supervised_group_ids"],
            parent_ids=producer["parent_ids"] + [leaked["artifact_id"]],
            fit_kind=producer["fit_kind"], metadata=producer["metadata"])
        receipt["provenance"] = [changed if r == producer else r for r in receipt["provenance"]] + [leaked]
        try:
            write_json(path, receipt)
            with self.assertRaisesRegex(ValueError, "leakage"):
                self.evaluate("leaked_responder")
            self.assertFalse((self.root / "leaked_responder").exists())
        finally:
            path.write_bytes(before)

    def test_cache_change_during_inference_prevents_receipt(self):
        path = self.cache / "responses.jsonl"
        before = path.read_bytes()
        def changed(*args, **kwargs):
            trace = run_episode(*args, **kwargs)
            write_jsonl(path, [])
            return trace
        try:
            with patch("cbmjev.crossfit_evaluation.run_episode", side_effect=changed):
                with self.assertRaisesRegex(ValueError, "cache file hash"):
                    self.evaluate("changed_cache", methods=["stop"])
            self.assertFalse((self.root / "changed_cache/receipt.json").exists())
        finally:
            path.write_bytes(before)

    def test_invalid_methods_and_all_budget_rejected(self):
        for name, options in (("static", {"methods": ["static"]}),
                              ("duplicates", {"methods": ["stop", "stop"]}),
                              ("budget", {"max_groups": 0})):
            with self.assertRaises(ValueError):
                self.evaluate("bad_" + name, **options)
            self.assertFalse((self.root / ("bad_" + name)).exists())
        receipt = self.evaluate("zero_budget", methods=["value_singleton"], max_groups=0)
        self.assertEqual(receipt["status"], "COMPLETE")

    def test_configured_minstop_cli_matches_original_at_zero_floor(self):
        out = self.root / "configured_minstop"
        argv = ["evaluate_crossfit_minstop_grid.py", "--prepared", str(self.prepared),
                "--planned", str(self.plan), "--merged", str(self.merged),
                "--responder", str(self.responder), "--cache", str(self.cache),
                "--out", str(out), "--max-budgets", "1", "--min-groups", "0", "1",
                "--action-family", "configured", "--matched-fixed-count"]
        with patch.object(sys, "argv", argv):
            minstop_grid_main()
        receipt = read_json(out / "receipt.json")
        self.assertEqual(receipt["status"], "COMPLETE")
        self.assertEqual(receipt["action_family"], "configured")
        self.assertEqual(receipt["num_policies"], 4)
        self.assertTrue(receipt["matched_fixed_count"])
        self.assertIn("traces.jsonl", receipt["files_sha256"])
        self.evaluate("minstop_original", methods=["value"], max_groups=1)
        original = read_jsonl(self.root / "minstop_original/traces.jsonl")[0]
        diagnostic = next(row for row in read_jsonl(out / "traces.jsonl")
                          if row["min_groups"] == 0)
        matched = next(row for row in read_jsonl(out / "traces.jsonl")
                       if row["policy_id"] == "fixed_at_value_minstop_min0_K1")
        self.assertEqual(len(matched["queried_groups"]), len(diagnostic["queried_groups"]))
        self.assertEqual(matched["adaptive_realized_groups"],
                         len(diagnostic["queried_groups"]))
        comparison = read_json(out / "metrics.json")["matched_selection_comparisons"][
            "value_minstop_min0_K1"]
        self.assertEqual(comparison["num_samples"], 1)
        self.assertEqual(sum(comparison["paired_correctness"].values()), 1)
        for key in ("prediction", "probabilities", "final_state", "queried_groups",
                    "queried_atoms", "calls", "declared_cost"):
            self.assertEqual(diagnostic[key], original[key])
        self.assertEqual([step["action"] for step in diagnostic["steps"]],
                         [step["action"] for step in original["steps"]])


if __name__ == "__main__":
    unittest.main()

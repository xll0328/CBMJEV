"""Tiny synthetic M1 execution; protected rows and raw texts stay out of outputs."""
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cbmjev.contracts import Concept, QueryGroup, Schema
from cbmjev.io import file_hash, read_json, read_jsonl, write_json, write_jsonl
from cbmjev.verification import expected_signed_gain
from scripts import evaluate_concept_verification as evaluator


class ConceptVerificationEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.schema = Schema("synthetic", 3,
            (Concept("binary", "A synthetic binary aspect", ("no", "yes")),
             Concept("ternary", "A synthetic ternary aspect", ("low", "middle", "high"))),
            (QueryGroup("binary", (0,)), QueryGroup("ternary", (1,))))
        self.members, self.samples, self.a, self.b = [], [], [], []
        roles = ["responder_fit"] * 3 + ["head_fit"] * 12 + ["policy_fit"] * 8 + ["policy_tune"] * 4 + ["confirmation"]
        for i, role in enumerate(roles):
            sid, gid = f"sample{i}", f"group{i}"
            concepts = [i % 2, i % 3]
            self.members.append({"sample_id": sid, "group_id": gid, "split": role})
            self.samples.append({"sample_id": sid, "group_id": gid, "dataset": "synthetic", "split": role,
                "input": {"text": "SECRET_RAW_TEXT_DO_NOT_PUBLISH"},
                "target": {"status": "OBSERVED", "value": "PROTECTED_POISON" if role == "confirmation" else i % 3},
                "concepts": [{"concept_id": concept.id, "annotation_status": "OBSERVED", "value": value}
                             for concept, value in zip(self.schema.concepts, concepts)]})
            if role not in ("responder_fit", "confirmation"):
                self.a.append({"sample_id": sid, "group_id": gid, "split": role, "A": [i % 2, (i + 1) % 3]})
                self.b.append({"sample_id": sid, "group_id": gid, "split": role,
                               "typed_values": [(i + 1) % 2, i % 3], "shared_values": concepts})

    def save(self, *, bad_source=False, legacy=False):
        write_json(self.root / "schema.json", self.schema.to_dict())
        write_jsonl(self.root / "membership.jsonl", self.members)
        write_jsonl(self.root / "samples.jsonl", self.samples)
        if legacy:
            for row in self.a:
                row["z"] = row.pop("A")
        write_jsonl(self.root / "a.jsonl", self.a)
        write_jsonl(self.root / "b.jsonl", self.b)
        metadata = {"binding": {"schema_hash": self.schema.hash},
                    "provenance": {"membership_sha256": file_hash(self.root / "membership.jsonl")},
                    "fit_sample_ids": [r["sample_id"] for r in self.members if r["split"] == "responder_fit"],
                    "fit_group_ids": [r["group_id"] for r in self.members if r["split"] == "responder_fit"]}
        a_meta = dict(metadata, predictions_sha256=file_hash(self.root / "a.jsonl"))
        if legacy:
            a_meta = {"format": "cbmjev-cache-v1", "schema_hash": self.schema.hash,
                      "membership_sha256": file_hash(self.root / "membership.jsonl"),
                      "responses_sha256": file_hash(self.root / "a.jsonl"),
                      "provenance": [{"supervised_group_ids": metadata["fit_group_ids"]}]}
        write_json(self.root / "a.metadata.json", a_meta)
        b_meta = dict(metadata, predictions_sha256=file_hash(self.root / "b.jsonl"))
        if bad_source:
            b_meta["fit_sample_ids"] = ["sample3"]
        write_json(self.root / "b.metadata.json", b_meta)

    def run_evaluator(self, **kwargs):
        return evaluator.run_development(prepared=self.root, a_predictions=self.root / "a.jsonl",
            b_predictions=self.root / "b.jsonl", a_metadata=self.root / "a.metadata.json",
            b_metadata=self.root / "b.metadata.json", output=self.root / "output", seed=40,
            synthetic=True, **kwargs)

    def test_end_to_end_zero_cost_diagnostic_includes_strong_reference(self):
        self.save()
        actual_evaluate = evaluator.evaluate_actions

        def require_freeze(*args, **kwargs):
            self.assertTrue((self.root / "output" / "freeze.json").exists())
            return actual_evaluate(*args, **kwargs)

        with patch.object(evaluator, "evaluate_actions", side_effect=require_freeze):
            report = self.run_evaluator()
        self.assertEqual(report["cost_status"], "COST_NOT_MEASURED")
        self.assertEqual(report["evidence_status"], "development_not_primary_noJclaims")
        self.assertFalse(report["confirmation_evaluated"])
        self.assertIsNone(report["p_values"])
        results = report["results"]["policy_tune_development"]
        self.assertEqual(set(results), {"0.0"})
        methods = results["0.0"]
        for required in ("stop", "fixed_0", "fixed_1", "best_fixed", "fixed_stop", "direct_gain", "joint",
                         "same_q_product", "independent_factorized", "all_fused", "raw_b", "shared_strong", "hindsight"):
            self.assertIn(required, methods)
            self.assertIsNone(methods[required]["J"])
        self.assertTrue(methods["shared_strong"]["independent_task_head"])
        self.assertIsNone(methods["shared_strong"]["events"])
        self.assertEqual(methods["stop"]["events"]["selected_concept_events"]["samples"], 0)
        self.assertTrue(methods["hindsight"]["diagnostic"])
        output = self.root / "output"
        events = read_jsonl(output / "row_events.jsonl")
        self.assertTrue(events)
        self.assertTrue(all(row["role"] == "policy_tune_development" for row in events))
        for row in events:
            if row["method"] == "shared_strong":
                self.assertIsNone(row["task_event"])
                self.assertIsNone(row["before_prediction"])
        for path in output.glob("*.json*"):
            self.assertNotIn("SECRET_RAW_TEXT", path.read_text())
            self.assertNotIn("PROTECTED_POISON", path.read_text())
        training = read_json(output / "training.json")
        self.assertEqual(training["models"]["shared_strong_task_head"]["n_train"], 12)
        self.assertEqual(training["models"]["task_head"]["n_train"], 48)
        self.assertEqual(len(list(output.glob("*.pt"))), 5)
        self.assertTrue((output / "frozen_models.json").exists())
        with self.assertRaisesRegex(ValueError, "new or empty"):
            self.run_evaluator()

    def test_legacy_seen_validation_keeps_policy_tune_group_disjoint(self):
        for collection in (self.members, self.samples, self.a, self.b):
            for row in collection:
                if row["split"] == "policy_tune":
                    row["split"] = "validation"
        self.save(legacy=True)
        report = self.run_evaluator(legacy_a=True)
        self.assertEqual(set(report["results"]), {"policy_tune_development", "validation_seen_development"})
        self.assertEqual(report["source_provenance"]["A"], "SOURCE_GROUPS_AUDITED")
        freeze = read_json(self.root / "output" / "freeze.json")
        self.assertFalse(set(freeze["policy_train_ids"]) & set(freeze["policy_tune_ids"]))
        validation_ids = {row["sample_id"] for row in self.members if row["split"] == "validation"}
        self.assertFalse(validation_ids & (set(freeze["policy_train_ids"]) | set(freeze["policy_tune_ids"])))

    def test_source_fit_membership_guard_runs_before_training(self):
        self.save(bad_source=True)
        with self.assertRaisesRegex(ValueError, "non-source role"):
            self.run_evaluator()
        self.assertFalse((self.root / "output").exists())

    def test_protected_roles_and_hyperparameter_overrides_have_no_cli_route(self):
        parser = evaluator.build_parser()
        flags = {option for action in parser._actions for option in action.option_strings}
        self.assertFalse(flags & {"--roles", "--test", "--confirmation", "--allow-protected-roles", "--epochs", "--synthetic"})
        self.assertTrue(next(action for action in parser._actions if action.dest == "a_metadata").required)
        # A prepared corpus with only policy-fit and confirmation must not use
        # confirmation as a missing tune/validation fallback.
        for collection in (self.members, self.samples, self.a, self.b):
            for row in collection:
                if row["split"] == "policy_tune":
                    row["split"] = "confirmation"
        self.save()
        with self.assertRaisesRegex(ValueError, "role absent"):
            self.run_evaluator()
        self.assertFalse((self.root / "output").exists())

    def test_runtime_interface_and_joint_integration_never_take_realized_b(self):
        self.assertEqual(tuple(inspect.signature(evaluator.runtime_gains).parameters),
                         ("initial_a", "lookup", "task_head", "direct", "joint", "factorized"))
        initial_a = np.array([[0, 1], [1, 2]])

        class Head:
            def predict_proba(self, concepts):
                return np.tile([.7, .2, .1], (len(concepts), 1))

        class Lookup:
            def initial(self, concepts):
                np.testing.assert_array_equal(concepts, initial_a)
                return concepts

        class Selector:
            def predict_gains(self, concepts):
                np.testing.assert_array_equal(concepts, initial_a)
                return np.tile([-.2, .3], (len(concepts), 1))

            def predict_joint(self, concepts):
                np.testing.assert_array_equal(concepts, initial_a)
                return tuple(np.tile(np.array([.4, .35, .25])[None, :, None] / c, (2, 1, c)) for c in (2, 3))

        after = (np.array([[1, 2], [0, 1]]), np.array([[2, 1, 0], [0, 2, 1]]))
        selector = Selector()
        with patch.object(evaluator, "hypothetical_predictions", return_value=after):
            values = evaluator.runtime_gains(initial_a, Lookup(), Head(), selector, selector, selector)
        for i in range(2):
            for j, joint in enumerate(selector.predict_joint(initial_a)):
                self.assertAlmostEqual(values["joint"][i, j], expected_signed_gain(joint[i], 0, after[j][i]))
        self.assertLess(values["direct_gain"][0, 0], 0)
        rounded = (np.asarray(selector.predict_joint(initial_a)[0], dtype=np.float32),)
        unchanged = evaluator._joint_gains(rounded, np.zeros(2, dtype=int),
                                            (np.zeros((2, 2), dtype=int),), product=True)
        np.testing.assert_array_equal(unchanged, np.zeros((2, 1)))
        with self.assertRaises(TypeError):
            evaluator.runtime_gains(initial_a, Lookup(), Head(), selector, selector, selector, source_b=initial_a)
        np.testing.assert_array_equal(evaluator.choose_columns([[0, -1], [-1, -.2]], [0, 0], 0), [0, 0])

    def test_measured_cost_profile_uses_totals_and_charges_policy_after_stop(self):
        profile = {"format": "verification-cost-v1", "units": "seconds_per_sample",
            "validation_status": "VALIDATED",
            "normalization": {"reference": "shared_strong_isolated", "seconds": 2.0},
            "context": {"platform": "SYNTHETIC_TEST_ONLY", "batch_size": 1, "n": 100,
                        "isolation_scope": "synthetic fixture; not a real measurement"},
            "stop_seconds": .2, "singleton_seconds": [1., 1.4], "all_fused_seconds": 1.7,
            "raw_b_seconds": 1.6, "shared_strong_seconds": 2.,
            "policy_seconds": {name: .1 for name in evaluator.POLICIES}}
        cost_path = self.root / "cost.json"
        write_json(cost_path, profile)
        checked = evaluator._cost_profile(cost_path, 2)
        np.testing.assert_allclose(evaluator._incremental_costs(checked, 2), [.4, .6])
        np.testing.assert_allclose(evaluator._row_costs(checked, "joint", np.array([0, 1, 2])), [.15, .55, .75])
        np.testing.assert_allclose(evaluator._row_costs(checked, "all_fused", np.array([3])), [.85])
        self.save()
        report = self.run_evaluator(cost_profile=cost_path)
        results = report["results"]["policy_tune_development"]
        self.assertEqual(set(results), {str(value) for value in evaluator.LAMBDAS})
        self.assertEqual(report["selection"]["selection_lambda"], .02)
        for lam, methods in results.items():
            stop = methods["stop"]
            self.assertAlmostEqual(stop["J"], stop["error"] + float(lam) * .15)
            self.assertIsNone(methods["hindsight"]["J"])

    def test_cost_profile_rejects_missing_or_mismatched_cache_live_validation(self):
        for index, status in enumerate((None, "CACHE_RUNTIME_MISMATCH")):
            profile = {"format": "verification-cost-v1", "units": "seconds_per_sample"}
            if status is not None:
                profile["validation_status"] = status
            path = self.root / f"unvalidated_cost_{index}.json"
            write_json(path, profile)
            with self.assertRaisesRegex(ValueError, "VALIDATED cache/live agreement"):
                evaluator._cost_profile(path, 2)


if __name__ == "__main__":
    unittest.main()

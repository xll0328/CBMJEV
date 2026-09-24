from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from cbmjev.choice_artifacts import train_choice_pair
from cbmjev.choice_evaluation import evaluate_choice_crossfit_validation
from cbmjev.cli import dispatch, parser
from cbmjev.crossfit import plan_crossfit_prepared
from cbmjev.crossfit_cache import cache_crossfit_responses
from cbmjev.crossfit_executor import execute_outer_fold
from cbmjev.crossfit_merge import merge_outer_folds
from cbmjev.io import read_json, read_jsonl
from cbmjev.pipeline import train_crossfit_responder
from cbmjev.runtime import run_episode
from tests_cbmjev.test_crossfit import fixture, write_json, write_jsonl, load_rows


class ChoiceCrossfitEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.prepared = fixture(cls.root)
        rows = load_rows(cls.prepared / "samples.jsonl")
        rows[0]["target"]["value"], rows[1]["target"]["value"] = 1, 0
        write_jsonl(cls.prepared / "samples.jsonl", rows)
        audit = read_json(cls.prepared / "audit.json")
        audit["source_revision"] = "SYNTHETIC_CHOICE_NESTED_TEST"
        write_json(cls.prepared / "audit.json", audit)
        cls.plan = cls.root / "plan"
        plan_crossfit_prepared(cls.prepared, cls.plan, inner_folds=2)
        config = {"seed": 29, "learning": {"objective": "risk", "hidden": 8,
            "head_epochs": 1, "policy_epochs": 1, "batch_size": 4,
            "masks_per_sample": 2, "actions_per_state": 64}}
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
        cls.targets = [directory / "action_targets.json" for directory in cls.outers]
        cls.config = cls.outers[0] / "config.json"
        cls.choice = cls.root / "choice"
        train_choice_pair(cls.prepared, cls.targets, cls.config, cls.choice, epochs=1,
                          max_questions=128, batch_size=4)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def evaluate(self, name, **kwargs):
        return evaluate_choice_crossfit_validation(self.prepared, self.plan, self.merged,
            self.responder, self.cache, self.choice, self.targets, self.config, self.root / name, **kwargs)

    def test_actual_nested_pair_same_head_cache_budget_no_label_inference(self):
        heads = []
        def checked(*args, **kwargs):
            self.assertNotIn("y", kwargs)
            self.assertNotIn("sample_id", kwargs)
            heads.append(id(args[2]))
            self.assertEqual(kwargs["max_groups"], 1)
            self.assertEqual(kwargs["cost_weight"], .03)
            trace = run_episode(*args, **kwargs)
            self.assertNotIn("y", trace)
            for step in trace["steps"]:
                chosen = "STOP" if not step["action"] else str(step["action"][0])
                self.assertEqual(step["scores"][chosen], max(step["scores"].values()))
            return trace
        with patch("cbmjev.choice_artifacts.iter_decision_sets",
                   side_effect=AssertionError("sealed training stream reparsed at evaluation")):
            with patch("cbmjev.choice_evaluation.run_episode", side_effect=checked):
                result = self.evaluate("evaluation")
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(len(set(heads)), 1)
        self.assertEqual(result["num_policies"], 2)
        self.assertTrue(result["source_binding"]["same_complete_outer_training_packages_verified"])
        traces = read_jsonl(self.root / "evaluation/traces.jsonl")
        self.assertEqual({row["split"] for row in traces}, {"validation"})
        self.assertEqual(len(traces), 2)
        self.assertEqual(len({row["sample_id"] for row in traces}), 1)
        self.assertFalse(result["coverage"]["cost_penalty_applied_again_at_inference"])
        self.assertFalse(result["paper_evidence"])

    def test_sealed_reload_matches_strict_training_stream_revalidation(self):
        from cbmjev.choice_artifacts import load_choice_pair

        def strict_reload(*args, **kwargs):
            kwargs["reuse_sealed_training_validation"] = False
            return load_choice_pair(*args, **kwargs)

        fast = self.root / "sealed_reload_eval"
        strict = self.root / "strict_reload_eval"
        self.evaluate(fast.name)
        with patch("cbmjev.choice_evaluation.load_choice_pair", side_effect=strict_reload):
            self.evaluate(strict.name)
        for name in ("settings.json", "metrics.json"):
            self.assertEqual((fast / name).read_bytes(), (strict / name).read_bytes())
        fast_traces = read_jsonl(fast / "traces.jsonl")
        strict_traces = read_jsonl(strict / "traces.jsonl")
        for trace in fast_traces + strict_traces:
            trace.pop("replay_wall_ms")
        self.assertEqual(fast_traces, strict_traces)

    def test_uniform_budget_cli_and_coverage(self):
        uniform = self.root / "uniform"
        train_choice_pair(self.prepared, self.targets, self.config, uniform, epochs=1,
                          max_questions=128, batch_size=4, budget_mode="uniform_remaining")
        args = parser().parse_args(["evaluate-choice-crossfit", "--prepared", str(self.prepared),
            "--planned", str(self.plan), "--merged", str(self.merged), "--responder", str(self.responder),
            "--cache", str(self.cache), "--choice", str(uniform), "--targets", *map(str, self.targets),
            "--target-config", str(self.config), "--out", str(self.root / "uniform_eval")])
        receipt = dispatch(args)
        self.assertEqual(receipt["coverage"]["training_budget_mode"], "uniform_remaining")
        self.assertEqual(set(receipt["coverage"]["training_counts"]), {"0", "1"})

    def test_capacity_control_triple_artifact_and_nested_replay(self):
        triple = self.root / "capacity_triple"
        receipt = train_choice_pair(self.prepared, self.targets, self.config, triple,
            epochs=1, max_questions=128, batch_size=4, capacity_control=True)
        self.assertEqual(receipt["format"], "cbmjev-structured-choice-capacity-artifact-v1")
        self.assertIn("independent_mlp.pt", receipt["files"])
        pair_fit = read_json(self.choice / "report.json")["fit"]
        triple_fit = read_json(triple / "report.json")["fit"]
        for key in ("source_content_sha256", "effective_features_sha256", "order_sha256",
                    "initial_active_logits_sha256"):
            self.assertEqual(pair_fit[key], triple_fit[key])
        for name in ("scalar", "attention"):
            self.assertEqual(pair_fit["heads"][name], triple_fit["heads"][name])
        result = evaluate_choice_crossfit_validation(self.prepared, self.plan, self.merged,
            self.responder, self.cache, triple, self.targets, self.config,
            self.root / "capacity_triple_eval")
        self.assertEqual(result["num_policies"], 3)
        metrics = read_json(self.root / "capacity_triple_eval/metrics.json")
        self.assertEqual(set(metrics["policies"]), {"structured_choice_scalar",
            "structured_choice_attention", "structured_choice_independent_mlp"})
        self.assertEqual({trace["split"] for trace in
                          read_jsonl(self.root / "capacity_triple_eval/traces.jsonl")}, {"validation"})

    def test_forbidden_split_lambda_and_initial_budget_fail_without_output(self):
        for index, options in enumerate(({"split": "test"}, {"split": "calibration"},
                                        {"cost_weight": .04}, {"max_groups": 0})):
            name = "forbidden%d" % index
            with self.assertRaises(ValueError):
                self.evaluate(name, **options)
            self.assertFalse((self.root / name).exists())

    def test_subset_targets_not_accepted_as_complete_nested_training(self):
        subset = self.root / "subset"
        train_choice_pair(self.prepared, self.targets[:1], self.config, subset, epochs=1, max_questions=8)
        out = self.root / "subset_eval"
        with self.assertRaisesRegex(ValueError, "complete merged outer"):
            evaluate_choice_crossfit_validation(self.prepared, self.plan, self.merged, self.responder,
                self.cache, subset, self.targets[:1], self.config, out)
        self.assertFalse(out.exists())

    def test_mutated_choice_blocks_completion(self):
        path = self.choice / "scalar.pt"
        before = path.read_bytes()
        def changed(*args, **kwargs):
            trace = run_episode(*args, **kwargs)
            with path.open("ab") as handle:
                handle.write(b"tamper")
            return trace
        try:
            with patch("cbmjev.choice_evaluation.run_episode", side_effect=changed):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    self.evaluate("mutation")
            self.assertFalse((self.root / "mutation/receipt.json").exists())
        finally:
            path.write_bytes(before)

    def test_mutated_outer_or_plan_blocks_completion(self):
        paths = (self.outers[0] / "head/head.pt",
                 self.plan / "fold_manifest.jsonl")
        for index, path in enumerate(paths):
            with self.subTest(path=path):
                before = path.read_bytes()
                out = self.root / ("outer_plan_mutation_%d" % index)

                def changed(*args, **kwargs):
                    trace = run_episode(*args, **kwargs)
                    with path.open("ab") as handle:
                        handle.write(b"tamper")
                    return trace

                try:
                    with patch("cbmjev.choice_evaluation.run_episode", side_effect=changed):
                        with self.assertRaisesRegex(ValueError, "outer.*changed"):
                            self.evaluate(out.name)
                    self.assertFalse((out / "receipt.json").exists())
                finally:
                    path.write_bytes(before)

    def test_three_group_nested_fixed_budget_extrapolation_vs_forced_stop(self):
        # Real three-group nested artifacts, fitting, export, reload and source
        # checks. Only preference logits are overridden to force a known path;
        # this is a coverage-bookkeeping regression, not learned-policy evidence.
        with tempfile.TemporaryDirectory() as root_string:
            root = Path(root_string)
            prepared = fixture(root)
            schema = read_json(prepared / "schema.json")
            schema["concepts"] = [{"id": "c%d" % i, "description": "concept %d" % i,
                                   "values": ["no", "yes"]} for i in range(3)]
            schema["groups"] = [{"id": "c%d" % i, "atoms": [i]} for i in range(3)]
            write_json(prepared / "schema.json", schema)
            rows = load_rows(prepared / "samples.jsonl")
            for row in rows:
                original = row["concepts"][0]
                row["concepts"] = [{**original, "concept_id": "c%d" % i} for i in range(3)]
            rows[0]["target"]["value"], rows[1]["target"]["value"] = 1, 0
            write_jsonl(prepared / "samples.jsonl", rows)
            from cbmjev.contracts import stable_hash
            audit = read_json(prepared / "audit.json")
            audit.update(source_revision="SYNTHETIC_THREE_GROUP_CHOICE_COVERAGE",
                         schema_hash=stable_hash(schema))
            write_json(prepared / "audit.json", audit)
            plan = root / "plan"
            plan_crossfit_prepared(prepared, plan, inner_folds=2)
            config = {"seed": 29, "learning": {"objective": "risk", "hidden": 8,
                "head_epochs": 1, "policy_epochs": 1, "batch_size": 4,
                "masks_per_sample": 2, "actions_per_state": 64}}
            outers = []
            for fold in range(3):
                directory = root / ("outer%d" % fold)
                execute_outer_fold(prepared, plan, directory, outer_fold=fold,
                    config=config, responder_options={"epochs": 1, "batch_size": 4})
                outers.append(directory)
            merged, responder, cache, choice = (root / name for name in ("merged", "final", "cache", "choice"))
            merge_outer_folds(prepared, plan, outers, merged)
            train_crossfit_responder(prepared, plan, responder, final=True, seed=29, epochs=1, batch_size=4)
            cache_crossfit_responses(prepared, plan, responder, cache, split="validation")
            targets = [directory / "action_targets.json" for directory in outers]
            target_config = outers[0] / "config.json"
            train_choice_pair(prepared, targets, target_config, choice, epochs=1,
                              max_questions=128, batch_size=4, remaining_groups=2)

            def forced_acquisition(observed, actions, **kwargs):
                return tuple(1. if action else 0. for action in actions)

            with patch("cbmjev.choice_runtime.StructuredChoiceController.predict_logits",
                       side_effect=forced_acquisition):
                receipt = evaluate_choice_crossfit_validation(prepared, plan, merged, responder,
                    cache, choice, targets, target_config, root / "evaluation", max_groups=2)
            self.assertEqual(receipt["coverage"]["training_counts"], {"2": 24})
            for policy in receipt["coverage"]["policies"].values():
                self.assertEqual(policy["covered_nontrivial_decisions_by_budget"], {"2": 1})
                self.assertEqual(policy["uncovered_nontrivial_decisions_by_budget"], {"1": 1})
                self.assertEqual(policy["forced_single_candidate_decisions_by_budget"], {"0": 1})
            for trace in read_jsonl(root / "evaluation/traces.jsonl"):
                self.assertEqual(trace["queried_groups"], [0, 1])
                self.assertEqual([step["legal_candidate_count"] for step in trace["steps"]], [4, 3, 1])
                self.assertEqual([step["decision"] for step in trace["steps"]], ["ACQUIRE", "ACQUIRE", "STOP"])


if __name__ == "__main__":
    unittest.main()

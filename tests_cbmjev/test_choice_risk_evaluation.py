from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from cbmjev.choice_artifacts import train_choice_pair
from cbmjev.choice_evaluation import _bound_sources
from cbmjev.crossfit import plan_crossfit_prepared
from cbmjev.crossfit_cache import cache_crossfit_responses
from cbmjev.crossfit_executor import execute_outer_fold
from cbmjev.crossfit_merge import merge_outer_folds
from cbmjev.io import read_json, read_jsonl
from cbmjev.pipeline import train_crossfit_responder
from cbmjev.runtime import run_episode
from scripts.evaluate_choice_risk_pair import evaluate_choice_risk_pair
from scripts.train_choice_risk_pair import train_choice_risk_pair
from tests_cbmjev.test_crossfit import fixture, write_json, write_jsonl, load_rows


class ChoiceRiskEvaluationTests(unittest.TestCase):
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
        audit["source_revision"] = "SYNTHETIC_CHOICE_RISK_NESTED_TEST"
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
        cache_crossfit_responses(cls.prepared, cls.plan, cls.responder, cls.cache,
                                 split="validation")
        cls.targets = [directory / "action_targets.json" for directory in cls.outers]
        cls.config = cls.outers[0] / "config.json"
        cls.soft, cls.risk = cls.root / "soft", cls.root / "risk"
        train_choice_pair(cls.prepared, cls.targets, cls.config, cls.soft,
                          epochs=1, max_questions=128, batch_size=4)
        train_choice_risk_pair(cls.prepared, cls.targets, cls.config,
                               cls.soft, cls.risk)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_nested_replay_is_validation_only_and_label_free_inference(self):
        seen = []

        def checked(*args, **kwargs):
            self.assertNotIn("y", kwargs)
            self.assertNotIn("sample_id", kwargs)
            seen.append(kwargs["method"])
            trace = run_episode(*args, **kwargs)
            self.assertNotIn("y", trace)
            return trace

        out = self.root / "risk_eval"
        with patch("scripts.evaluate_choice_risk_pair.run_episode", side_effect=checked):
            receipt = evaluate_choice_risk_pair(self.prepared, self.plan, self.merged,
                self.responder, self.cache, self.soft, self.risk, self.targets,
                self.config, out, max_groups=1)
        self.assertEqual(receipt["status"], "COMPLETE")
        self.assertEqual(receipt["num_policies"], 2)
        self.assertEqual(seen, ["structured_choice", "structured_choice"])
        self.assertFalse(receipt["paper_evidence"])
        traces = read_jsonl(out / "traces.jsonl")
        self.assertEqual(len(traces), 2)
        self.assertEqual({row["split"] for row in traces}, {"validation"})
        self.assertEqual({row["policy_id"] for row in traces},
            {"structured_choice_risk_scalar", "structured_choice_risk_attention"})
        self.assertFalse(receipt["coverage"]["cost_penalty_applied_again_at_inference"])
        with self.assertRaisesRegex(ValueError, "new"):
            evaluate_choice_risk_pair(self.prepared, self.plan, self.merged,
                self.responder, self.cache, self.soft, self.risk, self.targets,
                self.config, out, max_groups=1)

    def test_drifted_nested_source_cannot_publish_complete_receipt(self):
        checks = 0

        def drifted(*args, **kwargs):
            nonlocal checks
            checks += 1
            result = _bound_sources(*args, **kwargs)
            if checks == 2:
                return (*result[:-1], {**result[-1], "nested_system": "changed"})
            return result

        out = self.root / "drifted_eval"
        with patch("scripts.evaluate_choice_risk_pair._bound_sources", side_effect=drifted):
            with self.assertRaisesRegex(ValueError, "source changed"):
                evaluate_choice_risk_pair(self.prepared, self.plan, self.merged,
                    self.responder, self.cache, self.soft, self.risk, self.targets,
                    self.config, out, max_groups=1)
        self.assertEqual(checks, 2)
        self.assertFalse((out / "receipt.json").exists())


if __name__ == "__main__":
    unittest.main()

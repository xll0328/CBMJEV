"""Fixed-order Nano stopping control; mock scoring is not empirical evidence."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cbmjev import pipeline
from cbmjev.config import resolve_config
from cbmjev.io import read_json, write_json
from cbmjev.runtime import ReplayEnvironment, run_episode
from tests_cbmjev.test_crossfit_training import schema_fixture
from tests_cbmjev.test_stopping_controls import Head
from tests_cbmjev import test_matched_pair_pipeline


class RiskController:
    objective = "risk"

    def __init__(self):
        self.seen = []

    def predict(self, observed, actions):
        self.seen.append((observed, actions))
        return [0.5 if not a else (0.8 if observed[0] == 0 else 0.2) for a in actions]


class NanoStaticRiskTests(unittest.TestCase):
    def episode(self, answers=(0, 1, 1), **kwargs):
        schema = schema_fixture()
        controller = RiskController()
        trace = run_episode(ReplayEnvironment(answers, schema), schema, Head(),
                            method="nano_static_risk", controller=controller, **kwargs)
        return trace, controller

    def test_prefix_uses_training_order_and_evidence_changes_stop(self):
        a, c = self.episode(order=(2, 0, 1))
        b, _ = self.episode(answers=(1, 1, 1), order=(2, 0, 1))
        self.assertEqual([x["action"] for x in a["steps"]], [[2], [0], []])
        self.assertEqual([x["action"] for x in b["steps"]], [[2], [0], [1], []])
        self.assertEqual(c.seen[0], ((-1, -1, -1), ((), (2,))))
        self.assertTrue(all(x["legal_candidate_count"] == 2 for x in a["steps"]))
        self.assertTrue(all(x["scored_candidate_count"] == 2 for x in a["steps"]))
        self.assertEqual(b["steps"][-1]["legal_candidate_count"], 1)
        self.assertEqual(b["steps"][-1]["scored_candidate_count"], 0)

    def test_stop_tie_and_zero_budget(self):
        t, _ = self.episode(cost_weight=0.3)
        self.assertEqual(t["steps"][0]["action"], [])
        t, c = self.episode(max_groups=0)
        self.assertEqual(t["queried_groups"], [])
        self.assertEqual(c.seen, [])

    def test_infeasible_next_prefix_does_not_skip_to_cheaper_group(self):
        class Cost:
            units = "test"
            def __call__(self, observed, action):
                return 0 if not action else (5 if 2 in action else 1)
        t, c = self.episode(order=(2, 0, 1), max_cost=1, cost=Cost())
        self.assertEqual(t["queried_groups"], [])
        self.assertEqual(c.seen, [])

    def test_risk_adapter_allowed_independent_of_parent_objective(self):
        for objective in ("risk", "value"):
            resolve_config({"learning": {"objective": objective},
                            "evaluation": {"methods": ["nano_static_risk"]}})
        schema = schema_fixture()
        controller = RiskController()
        controller.objective = "value"
        with self.assertRaisesRegex(ValueError, "objective"):
            run_episode(ReplayEnvironment((0, 1, 1), schema), schema, Head(),
                        method="nano_static_risk", controller=controller)

    def test_pipeline_shares_scorer_binds_identity_and_rejects_formal_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, original = test_matched_pair_pipeline.PairedPipelineTests().run_pair(root)
            schema, _, manifest, config, parent, head, _, _ = original
            parent = {**parent, "schema_hash": schema.hash, "cache_sha256": "cache",
                      "static_order_sha256": "static", "responder_checkpoint_sha256": "responder",
                      "head_component_sha256": "head", "controller_component_sha256": "old-policy",
                      "source_code_hash": "source", "controller_artifact_id": "old-policy"}
            write_json(root / "models/receipt.json", parent)
            rows = [{"sample_id": "v", "group_id": "v", "split": "validation", "y": 0, "z": [0, 1]}]
            learned = type("Learned", (), {"pairs": None})()
            bundle = (schema, rows, manifest, config, parent, head, learned, (1, 0))
            with patch.object(pipeline, "load_model_bundle", return_value=bundle), \
                 patch("cbmjev.provenance.validate_target_exclusion"), \
                 patch("cbmjev.nanojev.load_nano_head") as load, \
                 patch("cbmjev.nanojev.NanoRiskController", return_value=RiskController()), \
                 patch.object(pipeline, "run_episode", return_value={}) as episode, \
                 patch("cbmjev.evaluation.summarize_traces", return_value={}):
                pipeline.evaluate_models(root / "models", root / "cache", root / "eval",
                    methods=["nano_risk", "nano_static_risk"], nano_dir=root / "out")
                load.assert_called_once()
                first, second = episode.call_args_list
                self.assertIs(first.kwargs["controller"], second.kwargs["controller"])
                self.assertEqual(second.kwargs["order"], (1, 0))
                metrics = read_json(root / "eval/metrics.json")
                self.assertEqual(set(metrics["nano_identity"]), {"nano_head.pt", "training.json"})
                for options in ({"split": "test", "evaluate_test": True}, {"split": "calibration"}, {"certification_run": True}):
                    with self.assertRaises(ValueError):
                        pipeline.evaluate_models(root / "models", root / "cache", root / "bad",
                            methods=["nano_static_risk"], nano_dir=root / "out", **options)
                with self.assertRaisesRegex(ValueError, "validation-only"):
                    pipeline.freeze_family(root / "models", root / "cache", root / "freeze",
                                           methods=["nano_static_risk"])
            args = dict(method="nano_static_risk", nano_hash="head", nano_identity=metrics["nano_identity"])
            digest = pipeline.system_hash(parent, schema, config, **args)
            args["nano_identity"] = {**args["nano_identity"], "training.json": "changed"}
            self.assertNotEqual(digest, pipeline.system_hash(parent, schema, config, **args))
            with self.assertRaises(ValueError):
                pipeline.system_hash(parent, schema, config, method="nano_static_risk")


if __name__ == "__main__":
    unittest.main()

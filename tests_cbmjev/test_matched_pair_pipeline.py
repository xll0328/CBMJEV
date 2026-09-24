"""Paired orchestration checks; fake backbone is not empirical Nano evidence."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from cbmjev import pipeline
from cbmjev.cli import parser
from cbmjev.config import resolve_config
from cbmjev.io import read_json
from cbmjev.matched_risk import fit_matched_mlp_risk
from tests_cbmjev.test_matched_risk import fixture, ReferenceScorer


class PairedPipelineTests(unittest.TestCase):
    def run_pair(self, root, *, mutate_source=False):
        schema, events = fixture()
        config = resolve_config({"seed": 21, "learning": {"hidden": 8}})
        cache, models = root / "cache", root / "models"
        cache.mkdir()
        models.mkdir()
        bundle = (schema, [{"split": "policy_fit", "group_id": "p"}],
                  {"responses_sha256": "cache"}, config,
                  {"models_sha256": "parent", "provenance": [], "head_artifact_id": "head"},
                  object(), None, None)
        items = [{"observed": e.observed, "action": e.action, "error": e.error} for e in events]
        shared = []
        from cbmjev.nanojev import fit_nano_risk
        def nano_fit(scorer, examples, schema, **kwargs):
            shared.append(examples)
            return fit_nano_risk(scorer, examples, schema, **kwargs)
        def mlp_fit(examples, schema, **kwargs):
            self.assertIs(examples, shared[0])
            return fit_matched_mlp_risk(examples, schema, **kwargs)
        def save(scorer, path, schema, task):
            torch.save(scorer.state_dict(), path)
        with patch.object(pipeline, "load_model_bundle", return_value=bundle), \
             patch.object(pipeline, "code_fingerprint", side_effect=["source", "changed" if mutate_source else "source"]), \
             patch("cbmjev.provenance.validate_target_exclusion") as exclusion, \
             patch("cbmjev.learning.iter_risk_training_examples", return_value=iter(items)), \
             patch("cbmjev.nanojev.load_local_nano", return_value=ReferenceScorer(schema, events, 21, 8)), \
             patch("cbmjev.nanojev.fit_nano_risk", side_effect=nano_fit), \
             patch("cbmjev.nanojev.save_nano_head", side_effect=save), \
             patch("cbmjev.matched_risk.fit_matched_mlp_risk", side_effect=mlp_fit):
            result = pipeline.train_nano_controller(cache, models, "fake-backbone", root / "out",
                         epochs=2, batch_size=3, matched_mlp=True)
            exclusion.assert_called_once()
        return result, bundle

    def test_shared_events_receipts_tensor_loader_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, bundle = self.run_pair(root)
            receipt = read_json(root / "out/training.json")
            report = read_json(root / "out/matched_mlp_training.json")
            self.assertEqual(receipt["event_sha256"], report["event_sha256"])
            self.assertEqual(receipt["training_loss"], report["training_loss_minibatch_mean"])
            self.assertEqual(receipt["event_exposures"], 10)
            self.assertTrue(result["matched_mlp"])
            self.assertEqual(receipt["feature_normalization"], "none")
            payload = torch.load(root / "out/matched_mlp.pt", weights_only=True)
            self.assertTrue(all(torch.is_tensor(v) for v in payload.values()))
            with patch.object(pipeline, "load_model_bundle", return_value=bundle):
                controller, _ = pipeline.load_matched_mlp_controller(root / "out", root / "models", root / "cache")
                self.assertEqual(controller.objective, "risk")
                self.assertEqual(len(controller.predict((-1, -1), [(), (0,)])), 2)
                torch.save({}, root / "out/matched_mlp.pt")
                with self.assertRaisesRegex(ValueError, "completion receipt"):
                    pipeline.load_matched_mlp_controller(root / "out", root / "models", root / "cache")

    def test_source_mutation_never_publishes_completion_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "changed during"):
                self.run_pair(root, mutate_source=True)
            self.assertFalse((root / "out/training.json").exists())

    def test_cli_opt_in(self):
        args = parser().parse_args(["train-nano-controller", "--cache", "c", "--models", "m",
                                   "--backbone", "b", "--out", "o", "--matched-mlp"])
        self.assertTrue(args.matched_mlp)
        self.assertEqual(args.feature_normalization, "none")
        args = parser().parse_args(["train-nano-controller", "--cache", "c", "--models", "m",
                                   "--backbone", "b", "--out", "o", "--feature-normalization", "layernorm"])
        self.assertEqual(args.feature_normalization, "layernorm")

    def test_invalid_feature_normalization_fails_before_io(self):
        with self.assertRaisesRegex(ValueError, "feature_normalization"):
            pipeline.train_nano_controller("missing", "missing", "missing", "missing",
                                           feature_normalization="batchnorm")

    def test_runtime_matched_alias_is_identical_risk_selection(self):
        from cbmjev.runtime import choose_action
        from cbmjev.contracts import DeclaredCost
        schema, events = fixture()
        controller, _ = fit_matched_mlp_risk(events, schema, epochs=1, hidden=8)
        args = dict(controller=controller, cost=DeclaredCost(), cost_weight=.03)
        expected = choose_action((-1, -1), [(), (0,), (1,), (0, 1)], method="risk", **args)
        self.assertEqual(expected, choose_action((-1, -1), [(), (0,), (1,), (0, 1)],
                                                method="matched_mlp_risk", **args))

    def test_validation_dispatch_identity_and_no_backbone_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, original = self.run_pair(root)
            schema, _, manifest, config, parent, head, _, _ = original
            parent = {**parent, "schema_hash": schema.hash, "cache_sha256": "cache",
                      "static_order_sha256": "static", "responder_checkpoint_sha256": "responder",
                      "head_component_sha256": "head", "controller_component_sha256": "old-policy",
                      "source_code_hash": "source", "controller_artifact_id": "old-policy"}
            from cbmjev.io import write_json
            write_json(root / "models/receipt.json", parent)
            controller, _ = fit_matched_mlp_risk(fixture()[1], schema, epochs=1, hidden=8)
            rows = [{"sample_id": "v", "group_id": "v", "split": "validation", "y": 0, "z": [0, 1]}]
            bundle = (schema, rows, manifest, config, parent, head, controller, (0, 1))
            with patch.object(pipeline, "load_model_bundle", return_value=bundle), \
                 patch("cbmjev.provenance.validate_target_exclusion"), \
                 patch("cbmjev.nanojev.load_nano_head") as backbone, \
                 patch.object(pipeline, "run_episode", return_value={}) as episode, \
                 patch("cbmjev.evaluation.summarize_traces", return_value={}):
                pipeline.evaluate_models(root / "models", root / "cache", root / "eval",
                                         methods=["matched_mlp_risk"], nano_dir=root / "out")
                backbone.assert_not_called()
                self.assertEqual(episode.call_args.kwargs["controller"].objective, "risk")
                self.assertEqual(episode.call_args.kwargs["method"], "matched_mlp_risk")
                metrics = read_json(root / "eval/metrics.json")
                self.assertIn("matched_mlp.pt", metrics["matched_mlp_identity"])
                with self.assertRaisesRegex(ValueError, "validation-only"):
                    pipeline.evaluate_models(root / "models", root / "cache", root / "test",
                          split="test", evaluate_test=True, methods=["matched_mlp_risk"], nano_dir=root / "out")
                with self.assertRaisesRegex(ValueError, "validation-only"):
                    pipeline.freeze_family(root / "models", root / "cache", root / "freeze",
                                           methods=["matched_mlp_risk"])
            identity = metrics["matched_mlp_identity"]
            digest = pipeline.system_hash(parent, schema, config, method="matched_mlp_risk", matched_hash=identity)
            self.assertNotEqual(digest, pipeline.system_hash(parent, schema, config, method="matched_mlp_risk",
                                 matched_hash={**identity, "matched_mlp.pt": "changed"}))


if __name__ == "__main__":
    unittest.main()

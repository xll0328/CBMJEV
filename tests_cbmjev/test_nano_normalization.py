"""Normalization isolation and raw frozen-feature cache gradient checks."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from cbmjev.nanojev import (NanoCandidateScorer, RiskTrainingExample, fit_nano_risk,
                           save_nano_head, load_nano_head)
from tests_cbmjev.test_responders import FakeBackbone, FakeTokenizer, schema


class NanoNormalizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(15)
        self.schema = schema()
        self.events = [RiskTrainingExample((-1, -1), (), 0.),
                       RiskTrainingExample((-1, -1), (0,), 1.),
                       RiskTrainingExample((1, -1), (), 0.),
                       RiskTrainingExample((-1, -1), (), 1.),
                       RiskTrainingExample((-1, -1), (0,), 0.)]

    def scorer(self, mode="none"):
        return NanoCandidateScorer(FakeBackbone(), FakeTokenizer(), max_length=2048,
                                   feature_normalization=mode)

    def test_raw_cache_preserves_trainable_normalization_updates(self):
        direct = self.scorer("layernorm")
        cached = copy.deepcopy(direct)
        initial = copy.deepcopy(direct.state_dict())
        _, r1 = fit_nano_risk(direct, self.events, self.schema, epochs=4, batch_size=2, seed=9)
        _, r2 = fit_nano_risk(cached, self.events, self.schema, epochs=4, batch_size=2,
                             seed=9, cache_features=True, feature_batch_size=2)
        for key in direct.state_dict():
            torch.testing.assert_close(direct.state_dict()[key], cached.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(r1["training_loss"], r2["training_loss"])
        for name, parameter in cached.norm.named_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
            self.assertFalse(torch.equal(parameter, initial["norm." + name]))
        self.assertIsNone(cached.backbone.embedding.weight.grad)
        self.assertEqual(r2["feature_normalization"], "layernorm")

    def test_default_exact_linear_legacy(self):
        scorer = self.scorer()
        features = scorer.features(["hello", "x"])
        self.assertTrue(torch.equal(scorer.score_features(features), scorer.head(features).squeeze(-1).float()))
        self.assertEqual(list(scorer.norm.parameters()), [])
        with self.assertRaises(ValueError):
            self.scorer("batchnorm")

    def test_new_save_reload_and_injected_mismatch(self):
        for mode in ("none", "layernorm"):
            original = self.scorer(mode)
            fit_nano_risk(original, self.events, self.schema, epochs=2, batch_size=2, cache_features=True)
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "head.pt"
                save_nano_head(original, path, self.schema)
                restored = copy.deepcopy(original)
                with torch.no_grad():
                    restored.head.weight.zero_()
                    for p in restored.norm.parameters():
                        p.zero_()
                load_nano_head(path, self.schema, scorer=restored, expected_task="risk")
                self.assertTrue(torch.equal(original.score_prompts(["hello"]), restored.score_prompts(["hello"])))
                other = self.scorer("layernorm" if mode == "none" else "none")
                with self.assertRaisesRegex(ValueError, "normalization mismatch"):
                    load_nano_head(path, self.schema, scorer=other)

    def test_old_v1_absent_normalization_loads_none(self):
        original = self.scorer()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "old.pt"
            save_nano_head(original, path, self.schema)
            data = torch.load(path, weights_only=True)
            del data["feature_normalization"]
            del data["norm"]
            torch.save(data, path)
            restored = copy.deepcopy(original)
            load_nano_head(path, self.schema, scorer=restored)
            self.assertEqual(restored.feature_normalization, "none")
            self.assertTrue(torch.equal(original.score_prompts(["hello"]), restored.score_prompts(["hello"])))

    def test_layernorm_missing_state_rejected(self):
        original = self.scorer("layernorm")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.pt"
            save_nano_head(original, path, self.schema)
            data = torch.load(path, weights_only=True)
            del data["norm"]
            torch.save(data, path)
            with self.assertRaisesRegex(ValueError, "missing normalization"):
                load_nano_head(path, self.schema, scorer=original)


if __name__ == "__main__":
    unittest.main()

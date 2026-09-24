"""Engineering equivalence checks; no pretrained quality or wall-clock speed claim."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from cbmjev.nanojev import NanoCandidateScorer, RiskTrainingExample, fit_nano_risk
from tests_cbmjev.test_responders import FakeBackbone, FakeTokenizer, schema


class NanoFeatureCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(18)
        self.schema = schema()
        self.scorer = NanoCandidateScorer(FakeBackbone(), FakeTokenizer(), max_length=2048)
        self.examples = [RiskTrainingExample((-1, -1), (), 0.),
                         RiskTrainingExample((-1, -1), (0,), 1.),
                         RiskTrainingExample((1, -1), (), 0.),
                         RiskTrainingExample((-1, -1), (), 1.),
                         RiskTrainingExample((-1, -1), (0,), 0.)]

    def test_cached_training_same_updates_tail_weighting_and_unique_encodes(self):
        direct, cached = copy.deepcopy(self.scorer), copy.deepcopy(self.scorer)
        with patch.object(direct.backbone, "forward", wraps=direct.backbone.forward) as direct_calls:
            _, direct_report = fit_nano_risk(direct, self.examples, self.schema,
                                             epochs=3, batch_size=2, seed=9)
        with patch.object(cached.backbone, "forward", wraps=cached.backbone.forward) as cache_calls:
            _, cached_report = fit_nano_risk(cached, self.examples, self.schema,
                epochs=3, batch_size=2, seed=9, cache_features=True, feature_batch_size=2)
        self.assertEqual(direct_calls.call_count, 9)
        self.assertEqual(cache_calls.call_count, 2)
        for key in direct.state_dict():
            torch.testing.assert_close(direct.state_dict()[key], cached.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(direct_report["training_loss"], cached_report["training_loss"])
        self.assertEqual(direct_report["sample_weighted_training_loss"],
                         cached_report["sample_weighted_training_loss"])
        self.assertNotEqual(cached_report["training_loss"], cached_report["sample_weighted_training_loss"])
        self.assertEqual(cached_report["optimizer_steps"], 9)
        self.assertEqual(cached_report["total_event_exposures"], 15)
        self.assertEqual(cached_report["feature_cache"]["unique_prompts"], 3)
        self.assertEqual(cached_report["feature_cache"]["storage_bytes"], 3 * 8 * 4)

    def test_unfrozen_or_tampered_backbone_rejected(self):
        for declared_frozen in (False, True):
            self.scorer.freeze_backbone = declared_frozen
            self.scorer.backbone.requires_grad_(True)
            with self.assertRaisesRegex(ValueError, "frozen"):
                fit_nano_risk(self.scorer, self.examples, self.schema, cache_features=True)

    def test_bound_rejected_before_any_encoding_or_updates(self):
        before = copy.deepcopy(self.scorer.state_dict())
        with patch.object(self.scorer.backbone, "forward", wraps=self.scorer.backbone.forward) as calls:
            with self.assertRaisesRegex(ValueError, "max_cache_bytes"):
                fit_nano_risk(self.scorer, self.examples, self.schema,
                              cache_features=True, max_cache_bytes=95)
            self.assertEqual(calls.call_count, 0)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, self.scorer.state_dict()[key]))

    def test_dropout_backbone_stays_eval_and_frozen(self):
        class DropoutBackbone(FakeBackbone):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout(.8)

            def forward(self, *args, **kwargs):
                result = super().forward(*args, **kwargs)
                result.last_hidden_state = self.dropout(result.last_hidden_state)
                return result
        scorer = NanoCandidateScorer(DropoutBackbone(), FakeTokenizer(), max_length=2048)
        baseline = copy.deepcopy(scorer)
        fit_nano_risk(scorer, self.examples, self.schema, epochs=2, cache_features=True)
        fit_nano_risk(baseline, self.examples, self.schema, epochs=2)
        self.assertTrue(all(not module.training for module in scorer.backbone.modules()))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(scorer.state_dict()[name], value, atol=1e-7, rtol=1e-6)
        self.assertIsNone(scorer.backbone.embedding.weight.grad)

    def test_cache_parameters_and_nontraining_events_fail_closed(self):
        for options in ({"feature_batch_size": 0}, {"max_cache_bytes": 0}):
            with self.assertRaises(ValueError):
                fit_nano_risk(self.scorer, self.examples, self.schema, cache_features=True, **options)
        with self.assertRaisesRegex(ValueError, "policy_fit"):
            fit_nano_risk(self.scorer, [RiskTrainingExample((-1, -1), (), 0., "validation")],
                          self.schema, cache_features=True)


if __name__ == "__main__":
    unittest.main()

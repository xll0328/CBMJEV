from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import torch

from cbmjev.choice_language import fit_frozen_language_choice_pair, FrozenLanguageChoiceController
from cbmjev.choice_encoding import encode_frozen_choice_features
from cbmjev.choice_targets import ChoiceExample, ChoiceInputs, ChoiceSupervision
from cbmjev.contracts import DeclaredCost
from cbmjev.nano_choice import NanoChoiceHead
from cbmjev.nanojev import NanoCandidateScorer
from cbmjev.runtime import ReplayEnvironment, run_episode
from tests_cbmjev.test_responders import FakeBackbone, FakeTokenizer, schema


class LanguageChoiceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.schema = schema()
        self.scorer = NanoCandidateScorer(FakeBackbone(), FakeTokenizer(),
            max_length=4096, feature_normalization="layernorm")
        self.inputs = ChoiceInputs((-1, -1), ((), (0,), (1,)), 2, (0., 1., 1.), .1)
        self.examples = (ChoiceExample(self.inputs,
            ChoiceSupervision((1., 0., 1.), (.1, .8, .1), .5), "fixture", None),)
        self.limits = dict(max_padded_tokens=4096, max_candidates_per_batch=2)

    def fit(self, examples=None):
        return fit_frozen_language_choice_pair(self.scorer,
            self.examples if examples is None else examples, self.schema,
            max_questions=2, **self.limits, epochs=2, batch_size=1, lr=.01, seed=5)

    def controller(self, head=None):
        head = head or NanoChoiceHead(8).eval().requires_grad_(False)
        return FrozenLanguageChoiceController(self.scorer, head, self.schema, **self.limits)

    def test_fit_to_episode_and_visible_only_encoding(self):
        with patch("cbmjev.choice_language.encode_frozen_choice_features",
                   wraps=encode_frozen_choice_features) as encoder:
            scalar, attention, report = self.fit()
        self.assertEqual(encoder.call_args.args[1], (self.inputs,))
        self.assertFalse(report["upstream_provenance_verified"])
        self.assertFalse(report["full_body_training"])
        self.assertEqual(report["feature_width"], 8)
        for head in (scalar, attention):
            controller = self.controller(head)
            task_head = Mock()
            task_head.probabilities.return_value = (.3, .7)
            for budget in (0, 1, 2):
                trace = run_episode(ReplayEnvironment((1, 0), self.schema), self.schema,
                    task_head, method="structured_choice", controller=controller, max_groups=budget)
                self.assertLessEqual(len(trace["queried_groups"]), budget)
                self.assertEqual(trace["steps"][-1]["decision"], "STOP")
                self.assertNotIn("y", trace)
            self.assertEqual(controller.last_stats["backend"], "frozen-language-choice")
            self.assertTrue(all(p.grad is None for p in self.scorer.backbone.parameters()))

    def test_supervision_only_changes_training_not_encoding(self):
        _, _, original = self.fit()
        changed = (replace(self.examples[0], derived_id="other",
            supervision=ChoiceSupervision((0., 1., 1.), (.8, .1, .1), .5),
            source={"hidden_label": 900, "sample_id": "private"}),)
        _, _, altered = self.fit(changed)
        self.assertEqual(original["encoder"], altered["encoder"])
        self.assertEqual(original["trainer"]["effective_features_sha256"],
                         altered["trainer"]["effective_features_sha256"])
        self.assertNotEqual(original["binding_sha256"], altered["binding_sha256"])

    def test_no_state_cache_and_no_scorer_normalization(self):
        controller = self.controller()
        args = dict(remaining_groups=2, cost=DeclaredCost(), cost_weight=.1)
        with patch.object(self.scorer, "features", wraps=self.scorer.features) as encode:
            first = controller.predict_logits(self.inputs.observed, self.inputs.actions, **args)
            calls = encode.call_count
            with torch.no_grad():
                self.scorer.norm.weight.fill_(19)
                self.scorer.head.weight.fill_(22)
            second = controller.predict_logits(self.inputs.observed, self.inputs.actions, **args)
            self.assertEqual(encode.call_count, 2 * calls)
        self.assertEqual(first, second)
        self.assertFalse(controller.last_stats["normalization_cached"])
        self.assertEqual(controller.last_stats["encoding"]["questions"], 1)

    def test_invalid_width_modes_limits_and_inputs(self):
        for head in (NanoChoiceHead(9).eval().requires_grad_(False), NanoChoiceHead(8),
                     NanoChoiceHead(8).double().eval().requires_grad_(False)):
            with self.assertRaises(ValueError):
                self.controller(head)
        with self.assertRaises(ValueError):
            FrozenLanguageChoiceController(self.scorer,
                NanoChoiceHead(8).eval().requires_grad_(False), self.schema,
                max_padded_tokens=0, max_candidates_per_batch=1)
        controller = self.controller()
        bad_training = (replace(self.examples[0], model_inputs=replace(self.inputs,
            actions=((), (0, 1)), incremental_costs=(0., 2.))),)
        with self.assertRaisesRegex(ValueError, "singleton"):
            self.fit(bad_training)
        with patch.object(self.scorer, "features", wraps=self.scorer.features) as encode:
            original_limit = self.scorer.max_length
            self.scorer.max_length = 1
            with self.assertRaisesRegex(ValueError, "no truncation"):
                self.fit()
            self.assertEqual(encode.call_count, 0)
            self.scorer.max_length = original_limit
        for actions, budget, weight in ((((), (0,), (0,)), 2, .1),
                (((), (0, 1)), 2, .1), (((), (0,)), 0, .1), (((),), 0, float("nan"))):
            with self.assertRaises(ValueError):
                controller.predict_logits((-1, -1), actions, remaining_groups=budget,
                    cost=DeclaredCost(), cost_weight=weight)
        self.scorer.backbone.train()
        with self.assertRaises(ValueError):
            controller.predict_logits((-1, -1), ((),), remaining_groups=0,
                cost=DeclaredCost(), cost_weight=0.)
        with self.assertRaises(ValueError):
            self.fit(list(self.examples))

    def test_device_and_head_mutation_rejected(self):
        controller = self.controller()
        controller.head.train()
        with self.assertRaisesRegex(ValueError, "frozen"):
            controller.predict_logits((-1, -1), ((),), remaining_groups=0,
                cost=DeclaredCost(), cost_weight=0.)
        self.scorer.head.to("meta")
        with self.assertRaisesRegex(ValueError, "one device"):
            self.controller()


if __name__ == "__main__":
    unittest.main()

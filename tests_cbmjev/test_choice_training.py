from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from cbmjev.choice_targets import ChoiceExample, ChoiceInputs, ChoiceSupervision
from cbmjev.choice_training import fit_choice_head_pair
from cbmjev.nano_choice import choice_soft_target_loss


class ChoicePairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.examples = tuple(ChoiceExample(
            ChoiceInputs((-1, -1), ((),) + tuple((i,) for i in range(k - 1)),
                         2, (0.0,) + (1.0,) * (k - 1), 0.1),
            ChoiceSupervision((0.0,) * k, tuple(1 / k for _ in range(k)), 0.5),
            "fixture-%d" % k, None) for k in (1, 3, 2))
        generator = torch.Generator().manual_seed(7)
        self.features = tuple(torch.randn(k, 4, generator=generator) for k in (1, 3, 2))

    def fit(self, **options):
        args = dict(epochs=3, batch_size=2, lr=0.01, seed=11, device="cpu")
        args.update(options)
        return fit_choice_head_pair(self.examples, self.features, **args)

    def test_tail_exposure_repeatability_and_no_target_regeneration(self):
        for seed in (11, 19):
            a, b, first = self.fit(seed=seed)
            c, d, second = self.fit(seed=seed)
            self.assertEqual(first, second)
            for left, right in ((a, c), (b, d)):
                for name, tensor in left.state_dict().items():
                    self.assertTrue(torch.equal(tensor, right.state_dict()[name]))
            self.assertEqual(first["optimizer_steps_per_head"], 6)
            self.assertEqual(first["questions_exposed_per_head"], 9)
            self.assertEqual(first["candidates_exposed_per_head"], 18)
            self.assertTrue(first["initial_active_logits_equal"])
            self.assertFalse(first["teachers_regenerated"])
            self.assertFalse(first["upstream_provenance_verified"])

    def test_capacity_control_has_identical_questions_order_and_initial_logits(self):
        scalar, attention, control, report = self.fit(capacity_control=True)
        _, _, paired = self.fit()
        self.assertEqual(report["format"], "cbmjev-capacity-controlled-choice-head-fit-v1")
        self.assertEqual(report["order_sha256"], paired["order_sha256"])
        self.assertEqual(report["initial_active_logits_sha256"],
                         paired["initial_active_logits_sha256"])
        self.assertEqual(report["source_content_sha256"], paired["source_content_sha256"])
        self.assertEqual(report["heads"]["scalar"], paired["heads"]["scalar"])
        self.assertEqual(report["heads"]["attention"], paired["heads"]["attention"])
        self.assertTrue(report["initial_active_logits_equal"])
        self.assertEqual(report["optimizer_steps_per_head"], 6)
        self.assertFalse(report["capacity_control"]["candidate_interaction"])
        counts = report["capacity_control"]["parameter_count"]
        self.assertLess(abs(counts["attention"] - counts["independent_mlp"]), 400)
        self.assertEqual(control.set_head, "independent_mlp")
        self.assertFalse(any(p.requires_grad for head in (scalar, attention, control)
                             for p in head.parameters()))

    def test_teachers_do_not_affect_initialization_or_orders(self):
        _, _, original = self.fit()
        self.examples = tuple(replace(e, supervision=replace(e.supervision,
            teacher_probabilities=(1.0,) + (0.0,) * (len(e.model_inputs.actions) - 1)))
            for e in self.examples)
        _, _, altered = self.fit()
        for key in ("initial_active_logits_sha256", "initial_shared_tensors_sha256", "order_sha256"):
            self.assertEqual(original[key], altered[key])
        self.assertNotEqual(original["source_content_sha256"], altered["source_content_sha256"])
        for name in ("scalar", "attention"):
            self.assertEqual(original["heads"][name]["initial_weights_sha256"],
                             altered["heads"][name]["initial_weights_sha256"])

    def test_source_mutation_guard(self):
        called = False
        def mutate(logits, teacher, valid):
            nonlocal called
            if not called:
                self.features[0][0, 0] += 1
                called = True
            return choice_soft_target_loss(logits, teacher, valid)
        with patch("cbmjev.choice_training.choice_soft_target_loss", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "source mutated"):
                self.fit()

    def test_cpu_initialization_preserves_rng_without_cuda_calls(self):
        before = torch.random.get_rng_state().clone()
        with patch("torch.cuda.manual_seed_all", side_effect=AssertionError("CUDA seed touched")), \
             patch("torch.cuda.get_rng_state", side_effect=AssertionError("CUDA RNG touched")):
            self.fit()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_invalid_features_and_alignment(self):
        original = self.features
        for first in (original[0].clone().requires_grad_(), torch.zeros(2, 4),
                      torch.full((1, 4), float("nan")), torch.zeros(1, 5)):
            self.features = (first,) + original[1:]
            with self.assertRaises(ValueError):
                self.fit()
        self.features = original[:-1]
        with self.assertRaises(ValueError):
            self.fit()

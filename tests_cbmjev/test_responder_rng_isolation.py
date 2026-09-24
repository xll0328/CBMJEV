import unittest
from contextlib import ExitStack, nullcontext
from unittest.mock import patch

import torch

from cbmjev.contracts import Concept, ModelInput, QueryGroup, Schema
from cbmjev.responders import (ConceptTrainingExample, HashingTextResponder,
                              _seed_responder_rng, fit_responder, fit_text_responder)


class ResponderRNGIsolationTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(torch.use_deterministic_algorithms,
                        torch.are_deterministic_algorithms_enabled(),
                        warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.schema = Schema("synthetic_engineering", 2,
                             (Concept("a", "a", ("no", "yes")),),
                             (QueryGroup("a", (0,)),))
        self.examples = [ConceptTrainingExample(ModelInput(text="good"), (1,)),
                         ConceptTrainingExample(ModelInput(text="bad"), (0,))]

    def test_cpu_never_touches_cuda_and_restores_cpu_rng(self):
        before = torch.get_rng_state().clone()
        with ExitStack() as stack:
            for name in ("get_rng_state", "set_rng_state", "manual_seed", "manual_seed_all",
                         "current_device", "device_count", "_lazy_init"):
                stack.enter_context(patch("torch.cuda." + name,
                                          side_effect=AssertionError("CPU touched CUDA " + name)))
            first, report = fit_text_responder(self.examples, self.schema, seed=19,
                                               epochs=2, vocab_size=32, embedding_dim=8)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            second, second_report = fit_text_responder(self.examples, self.schema, seed=19,
                                                       epochs=2, vocab_size=32, embedding_dim=8)
        self.assertEqual(report["training_loss"], second_report["training_loss"])
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]), key)

    def test_exact_legacy_cpu_seed_initialization_and_fit(self):
        # CPU generator part of legacy torch.manual_seed, without its unrelated
        # device side effects. The initial parameters and training sequence match.
        torch.random.default_generator.manual_seed(23)
        legacy_model = HashingTextResponder(self.schema, 32, 8)
        legacy_model, legacy_report = fit_responder(legacy_model, self.examples, seed=23, epochs=2)
        actual, report = fit_text_responder(self.examples, self.schema, seed=23, epochs=2,
                                            vocab_size=32, embedding_dim=8)
        self.assertEqual(report["training_loss"], legacy_report["training_loss"])
        for key, value in actual.state_dict().items():
            self.assertTrue(torch.equal(value, legacy_model.state_dict()[key]), key)

    def test_cuda_fork_saves_only_selected_logical_device_on_exception(self):
        before = torch.get_rng_state().clone()
        fake_state = torch.tensor([1], dtype=torch.uint8)
        with patch("torch.cuda.get_rng_state", return_value=fake_state) as get_state, \
             patch("torch.cuda.set_rng_state") as set_state, \
             patch("torch.cuda.device_count", side_effect=AssertionError("enumerated GPUs")), \
             patch("cbmjev.responders.fit_responder", side_effect=RuntimeError("training failed")):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                fit_text_responder(self.examples, self.schema, device="cuda:2", seed=13)
        get_state.assert_called_once_with(2)
        set_state.assert_called_once_with(fake_state, 2)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_cuda_seed_only_selected_device(self):
        with patch("torch.cuda.device", return_value=nullcontext()) as device, \
             patch("torch.cuda.manual_seed") as seed, \
             patch("torch.cuda.manual_seed_all", side_effect=AssertionError("seeded every GPU")):
            _seed_responder_rng(17, torch.device("cuda:2"))
        device.assert_called_once_with(torch.device("cuda:2"))
        seed.assert_called_once_with(17)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_actual_restore_and_repeatability(self):
        before_cpu = torch.get_rng_state().clone()
        before_cuda = torch.cuda.get_rng_state(0).clone()
        states, reports = [], []
        for _ in range(2):
            model, report = fit_text_responder(self.examples, self.schema, device="cuda:0", seed=31,
                                               epochs=2, vocab_size=32, embedding_dim=8)
            states.append({key: value.cpu().clone() for key, value in model.state_dict().items()})
            reports.append(report)
            self.assertTrue(torch.equal(before_cpu, torch.get_rng_state()))
            self.assertTrue(torch.equal(before_cuda, torch.cuda.get_rng_state(0)))
        self.assertEqual(reports[0]["training_loss"], reports[1]["training_loss"])
        for key in states[0]:
            self.assertTrue(torch.equal(states[0][key], states[1][key]), key)


if __name__ == "__main__":
    unittest.main()

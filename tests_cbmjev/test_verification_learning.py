"""Synthetic CPU checks of selector information boundaries and proper losses."""
import inspect
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cbmjev.verification_learning import (
    SELECTOR_CONFIGS, TASK_CONFIGS, FitConfig, GainSelector, JointResponseMLP,
    JointSelector, SignedGainMLP, TaskMLP, _train_epoch, fit_gain_selector,
    fit_joint_selector, fit_task_head, load_model, refit_task_head,
)


class VerificationLearningTests(unittest.TestCase):
    def setUp(self):
        self.counts = (2, 3)
        self.a = np.array([[0, 0], [1, 1], [0, 2], [1, 0], [0, 1], [1, 2]], dtype=np.int64)
        self.y = np.array([0, 1, 2, 1, 0, 2], dtype=np.int64)
        self.b = self.a[:, ::-1].copy()
        self.b[:, 0] %= 2
        self.fast = dict(epochs=3, configs=[FitConfig(1e-2)], hidden=8, synthetic=True, batch_size=4)

    def joint(self, **kwargs):
        return fit_joint_selector(self.a, self.y, self.b, self.a, self.y, self.b,
                                  category_counts=self.counts, num_classes=3, seed=40,
                                  **self.fast, **kwargs)

    def task(self, **kwargs):
        return fit_task_head(self.a, self.y, self.a, self.y, category_counts=self.counts,
                             num_classes=3, seed=40, **self.fast, **kwargs)

    def test_joint_normalized_shared_y_marginal_and_proper_tune_loss(self):
        predictor = self.joint()
        joints = predictor.predict_joint(self.a, batch_size=2)
        self.assertEqual([p.shape for p in joints], [(6, 3, 2), (6, 3, 3)])
        for p in joints:
            np.testing.assert_allclose(p.sum(axis=(1, 2)), 1, atol=2e-7)
            self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(joints[0].sum(axis=2), joints[1].sum(axis=2), atol=1e-7)
        with torch.no_grad():
            y_logits, response_logits = predictor.model(torch.tensor(self.a))
            y = torch.tensor(self.y)
            response_nll = sum(float(F.cross_entropy(r[torch.arange(len(y)), y],
                                                      torch.tensor(self.b[:, j])))
                               for j, r in enumerate(response_logits)) / len(self.counts)
            expected = float(F.cross_entropy(y_logits, y)) + response_nll
        self.assertAlmostEqual(predictor.report["best_tune_loss"], expected, places=6)
        self.assertEqual(predictor.report["selection_metric"],
                         "task_nll_plus_concept_mean_conditional_response_nll")

    def test_factorized_model_independence_is_fitted_separately(self):
        conditional, factorized = self.joint(), self.joint(factorized=True)
        self.assertIsNot(conditional.model, factorized.model)
        self.assertLess(factorized.report["parameter_count"], conditional.report["parameter_count"])
        for p in factorized.predict_joint(self.a):
            np.testing.assert_allclose(p, p.sum(axis=2)[:, :, None] * p.sum(axis=1)[:, None, :], atol=1e-7)
        self.assertEqual(factorized.report["selection_metric"],
                         "task_nll_plus_concept_mean_factorized_response_nll")

    def test_conditional_response_can_interact_with_initial_a_and_y(self):
        model = JointResponseMLP((2,), 2, hidden=1)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.representation[0].weight.copy_(torch.tensor([[0.0, 1.0]]))
            model.response_heads[0].weight.copy_(torch.tensor([[-2.0], [2.0], [2.0], [-2.0]]))
        joint = JointSelector(model, {}).predict_joint(np.array([[0], [1]]))[0]
        response = joint / joint.sum(axis=2)[:, :, None]
        self.assertGreater(response[1, 0, 1], response[0, 0, 1])
        self.assertLess(response[1, 1, 1], response[0, 1, 1])

    def test_negative_gains_are_learned_without_clipping(self):
        targets = np.tile([-1, 1], (len(self.a), 1))
        predictor = fit_gain_selector(self.a, targets, self.a, targets, category_counts=self.counts,
                                      seed=40, epochs=30, configs=[(0.03, 0)], hidden=8, synthetic=True)
        gains = predictor.predict_gains(self.a)
        self.assertTrue((gains[:, 0] < -0.5).all())
        self.assertTrue((gains[:, 1] > 0.5).all())
        self.assertAlmostEqual(predictor.report["best_tune_loss"], float(((gains - targets) ** 2).mean()), places=6)
        with torch.no_grad():
            for p in predictor.model.parameters():
                p.zero_()
            predictor.model.network[-1].bias.copy_(torch.tensor([-2.0, 2.0]))
        np.testing.assert_array_equal(predictor.predict_gains(self.a)[0], [-2, 2])

    def test_weighted_task_tune_nll_and_checkpoint_selection(self):
        train_weights = np.array([1, 1, 0.25, 0.25, 0.25, 0.25])
        tune_weights = np.array([0, 3, 1, 2, 0, 4])
        predictor = self.task(train_weights=train_weights, tune_weights=tune_weights)
        p = predictor.predict_proba(self.a, batch_size=2)
        expected = (-np.log(p[np.arange(len(self.y)), self.y]) * tune_weights).sum() / tune_weights.sum()
        self.assertAlmostEqual(predictor.report["best_tune_loss"], expected, places=6)
        history = predictor.report["trials"][0]["history"]
        self.assertEqual(len(history), self.fast["epochs"])
        self.assertEqual(predictor.report["selected_epoch"], 1 + np.argmin([h["tune_loss"] for h in history]))
        self.assertFalse(predictor.report["early_stopping"])

    def test_global_weight_normalization_and_partial_batch_gradient(self):
        # Zero learning rate keeps logits fixed: accumulated minibatch gradients
        # must equal the full weighted objective scaled by N/effective batch.
        model = TaskMLP(self.counts, 3, hidden=3, dropout=0)
        x, y = torch.tensor(self.a[:5]), torch.tensor(self.y[:5])
        weights = torch.tensor([1.0, 2.0, 0.1, 0.1, 0.1])
        loss = (F.cross_entropy(model(x), y, reduction="none") * weights).sum() / weights.sum()
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))
        recorded = []

        class RecordingSGD(torch.optim.SGD):
            def step(self, closure=None):
                recorded.append([p.grad.clone() for p in model.parameters()])
                return super().step(closure)

        optimizer = RecordingSGD(model.parameters(), lr=0)
        _train_epoch(model, optimizer, x, (y,), weights, "task", torch.Generator().manual_seed(10), 3)
        self.assertEqual(len(recorded), 2)
        for index, expected in enumerate(gradients):
            actual = sum(batch[index] for batch in recorded)
            torch.testing.assert_close(actual, expected * 5 / 3, atol=1e-6, rtol=1e-5)

    def test_refit_uses_frozen_epoch_config_and_all_supplied_states(self):
        selected = self.task()
        refitted = refit_task_head(np.concatenate([self.a, self.a]), np.tile(self.y, 2),
                                   selected=selected, seed=41, weights=np.ones(12))
        self.assertEqual(refitted.report["n_train"], 12)
        self.assertEqual(refitted.report["n_tune"], 0)
        self.assertEqual(refitted.report["selected_config"], selected.report["selected_config"])
        self.assertEqual(len(refitted.report["history"]), selected.report["selected_epoch"])
        self.assertEqual(refitted.report["checkpoint_rule"], "final frozen epoch")
        self.assertEqual(refitted.report["selection_report"], selected.report)
        np.testing.assert_allclose(refitted.predict_proba(self.a).sum(axis=1), 1, atol=1e-7)

    def test_determinism_and_rng_thread_restoration(self):
        rng = torch.random.get_rng_state().clone()
        numpy_rng = np.random.get_state()
        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        first, second = self.joint(), self.joint()
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(threads, torch.get_num_threads())
        self.assertEqual(deterministic, torch.are_deterministic_algorithms_enabled())
        self.assertEqual(numpy_rng[0], np.random.get_state()[0])
        np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
        self.assertEqual(first.report["trials"][0]["history"], second.report["trials"][0]["history"])
        for key, tensor in first.model.state_dict().items():
            torch.testing.assert_close(tensor, second.model.state_dict()[key], atol=0, rtol=0)

    def test_serialization_parity_and_aggregate_only_report(self):
        models = [(self.task(), "predict_proba"), (self.joint(), "predict_joint"),
                  (self.joint(factorized=True), "predict_joint"),
                  (GainSelector(SignedGainMLP(self.counts), {}), "predict_gains")]
        with tempfile.TemporaryDirectory() as directory:
            for index, (predictor, method) in enumerate(models):
                path = Path(directory) / f"model{index}.pt"
                predictor.save(path)
                before_rng = torch.random.get_rng_state().clone()
                restored = load_model(path)
                self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))
                pickled = pickle.loads(pickle.dumps(predictor))
                expected = getattr(predictor, method)(self.a)
                for actual in (getattr(restored, method)(self.a), getattr(pickled, method)(self.a)):
                    if isinstance(expected, tuple):
                        for left, right in zip(expected, actual):
                            np.testing.assert_array_equal(left, right)
                    else:
                        np.testing.assert_array_equal(expected, actual)
                report = json.dumps(restored.report)
                self.assertNotIn("train_a", report)
                self.assertNotIn("train_b", report)
                self.assertNotIn("train_y", report)
                self.assertEqual(set(vars(restored)), {"model", "report"})

    def test_no_hidden_b_y_confidence_or_ids_at_inference(self):
        for predictor, method in ((self.joint(), "predict_joint"),
                                  (GainSelector(SignedGainMLP(self.counts), {}), "predict_gains")):
            function = getattr(predictor, method)
            self.assertEqual(tuple(inspect.signature(function).parameters), ("initial_a", "batch_size"))
            for forbidden in ("source_b", "y", "confidences", "source_ids", "action_ids"):
                with self.assertRaises(TypeError):
                    function(self.a, **{forbidden: self.b})
            with self.assertRaises(ValueError):
                function(np.concatenate([self.a, self.b], axis=1))
            with self.assertRaises(ValueError):
                function(np.eye(sum(self.counts)))

    def test_default_search_budget_and_explicit_synthetic_override(self):
        self.assertEqual([(c.learning_rate, c.weight_decay) for c in TASK_CONFIGS], [(3e-4, 0), (1e-3, 0)])
        self.assertEqual(set((c.learning_rate, c.weight_decay) for c in SELECTOR_CONFIGS),
                         {(3e-4, 0), (3e-4, 1e-3), (1e-3, 0), (1e-3, 1e-3)})
        with self.assertRaisesRegex(ValueError, "synthetic=True"):
            fit_gain_selector(self.a, self.a, self.a, self.a,
                              category_counts=self.counts, seed=40, epochs=1)
        targets = np.zeros_like(self.a)
        predictor = fit_gain_selector(self.a, targets, self.a, targets, category_counts=self.counts, seed=40)
        self.assertEqual(len(predictor.report["trials"]), 4)
        self.assertTrue(all(len(t["history"]) == 100 for t in predictor.report["trials"]))

    def test_validation_of_shapes_ranges_nonfinite_and_width(self):
        predictor = self.task()
        for invalid in (self.a.astype(float), np.ones((6, 3), dtype=int), self.a * 9,
                        np.full(self.a.shape, np.nan), np.zeros(self.a.shape, dtype=bool)):
            with self.assertRaises(ValueError):
                predictor.predict_proba(invalid)
        for weights in ([0] * 6, [1, -1, 1, 1, 1, 1], [1, np.inf, 1, 1, 1, 1], [1, 1]):
            with self.assertRaises(ValueError):
                self.task(train_weights=weights)
        with self.assertRaises(ValueError):
            fit_joint_selector(self.a, self.y, self.b[:-1], self.a, self.y, self.b,
                               category_counts=self.counts, num_classes=3, seed=40, **self.fast)
        with self.assertRaises(ValueError):
            fit_gain_selector(self.a, np.full(self.a.shape, np.nan), self.a, self.a,
                              category_counts=self.counts, seed=40, **self.fast)
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                TaskMLP(self.counts, 3, hidden=bad)
        with self.assertRaises(ValueError):
            TaskMLP(self.counts, 3, dropout=1)
        self.assertEqual(predictor.predict_proba(np.empty((0, 2), dtype=int)).shape, (0, 3))


if __name__ == "__main__":
    unittest.main()

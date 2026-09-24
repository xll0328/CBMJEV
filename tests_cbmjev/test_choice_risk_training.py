import unittest

import torch

from cbmjev.choice_risk_training import fit_choice_utility_pair
from cbmjev.choice_training import fit_choice_head_pair
from tests_cbmjev.test_choice_training import ChoicePairTests


class ChoiceRiskTrainingTests(unittest.TestCase):
    def setUp(self):
        ChoicePairTests.setUp(self)

    def test_matched_initialization_order_and_exposure(self):
        args = dict(epochs=2, batch_size=2, lr=.01, seed=11, device="cpu")
        scalar_r, attention_r, risk = fit_choice_utility_pair(
            self.examples, self.features, **args)
        _, _, soft = fit_choice_head_pair(self.examples, self.features, **args)
        for key in ("source_content_sha256", "effective_features_sha256",
                    "derived_ids_in_order", "initial_shared_tensors_sha256",
                    "initial_active_logits_sha256", "epoch_order_sha256",
                    "questions_exposed_per_head", "candidates_exposed_per_head",
                    "optimizer_steps_per_head"):
            self.assertEqual(risk[key], soft[key])
        for name in ("scalar", "attention"):
            self.assertEqual(risk["heads"][name]["initial_weights_sha256"],
                             soft["heads"][name]["initial_weights_sha256"])
        self.assertEqual(risk["config"]["loss"], "question-mean-active-utility-MSE")
        self.assertNotEqual(risk["heads"]["scalar"]["final_weights_sha256"],
                            soft["heads"]["scalar"]["final_weights_sha256"])
        self.assertTrue(all(not p.requires_grad for p in scalar_r.parameters()))
        self.assertTrue(all(not p.requires_grad for p in attention_r.parameters()))

    def test_repeated_run_and_rng_preservation(self):
        before = torch.random.get_rng_state().clone()
        left, _, first = fit_choice_utility_pair(self.examples, self.features,
            epochs=2, batch_size=2, lr=.01, seed=19)
        right, _, second = fit_choice_utility_pair(self.examples, self.features,
            epochs=2, batch_size=2, lr=.01, seed=19)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for name, tensor in left.state_dict().items():
            self.assertTrue(torch.equal(tensor, right.state_dict()[name]))

    def test_cost_and_error_guards(self):
        from dataclasses import replace
        original = self.examples
        changed = replace(original[1], model_inputs=replace(original[1].model_inputs,
            cost_weight=.2))
        with self.assertRaisesRegex(ValueError, "common cost weight"):
            fit_choice_utility_pair((original[0], changed), self.features[:2],
                epochs=1, batch_size=2, lr=.01, seed=11)
        bad = replace(original[0], supervision=replace(original[0].supervision,
            realized_errors=(.5,)))
        with self.assertRaisesRegex(ValueError, "Boolean errors"):
            fit_choice_utility_pair((bad,), self.features[:1],
                epochs=1, batch_size=1, lr=.01, seed=11)


if __name__ == "__main__":
    unittest.main()

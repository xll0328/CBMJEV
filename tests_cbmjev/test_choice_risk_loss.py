import unittest

import torch

from cbmjev.choice_risk_loss import choice_utility_regression_loss


class ChoiceRiskLossTests(unittest.TestCase):
    def test_question_mean_padding_and_gradient(self):
        logits = torch.tensor([[0., -1., -torch.inf], [1., -1., 0.]],
                              requires_grad=True)
        errors = torch.tensor([[1., 0., 0.], [0., 1., 1.]])
        costs = torch.tensor([[0., 1., 0.], [0., 1., 1.]])
        valid = torch.tensor([[True, True, False], [True, True, True]])
        loss = choice_utility_regression_loss(logits, errors, costs, .2, valid)
        expected = ((0. - (-1.))**2 + (-1. - (-.2))**2) / 2
        expected += ((1. - 0.)**2 + (-1. - (-1.2))**2 + (0. - (-1.2))**2) / 3
        self.assertAlmostEqual(loss.item(), expected / 2)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad[valid]).all())
        self.assertEqual(logits.grad[~valid].abs().sum().item(), 0.)

    def test_t8_expected_utility_ranking_is_consistent(self):
        # At the indistinguishable state in T8, squared-loss population logits
        # are negative expected utility, unlike mean per-case softmax labels.
        p = .52
        errors_i = torch.tensor([[1.] + [0., 1., 0.] + [1.] * 25])
        errors_ii = torch.tensor([[1.] + [1., 0., 1.] + [1.] * 25])
        costs = torch.tensor([[0.] + [1.] * 28])
        valid = torch.ones((1, 29), dtype=torch.bool)
        optimal = -(p * errors_i + (1-p) * errors_ii + .03 * costs)
        self.assertEqual(optimal.argmax(-1).item(), 1)
        self.assertAlmostEqual(optimal[0, 1].item(), -.51)
        self.assertAlmostEqual(optimal[0, 2].item(), -.55)
        logits = optimal.clone().requires_grad_()
        risk = p * choice_utility_regression_loss(logits, errors_i, costs, .03, valid)
        risk += (1-p) * choice_utility_regression_loss(logits, errors_ii, costs, .03, valid)
        risk.backward()
        self.assertLess(logits.grad.abs().max().item(), 1e-6)

    def test_invalid_targets_and_masks(self):
        logits = torch.tensor([[0., -torch.inf]])
        errors = torch.tensor([[1., 0.]])
        costs = torch.tensor([[0., 0.]])
        valid = torch.tensor([[True, False]])
        for changed_errors, changed_costs, weight in (
                (torch.tensor([[.5, 0.]]), costs, .1),
                (errors, torch.tensor([[-1., 0.]]), .1),
                (errors, costs, float("nan")),
                (errors, costs, True),
                (torch.tensor([[1., 1.]]), costs, .1)):
            with self.assertRaises(ValueError):
                choice_utility_regression_loss(logits, changed_errors,
                                               changed_costs, weight, valid)


if __name__ == "__main__":
    unittest.main()

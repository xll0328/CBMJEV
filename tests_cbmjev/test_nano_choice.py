import copy
import io
import unittest

import torch

from cbmjev.nano_choice import NanoChoiceHead, choice_soft_target_loss


class NanoChoiceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.x = torch.randn(2, 3, 8)
        self.mask = torch.tensor([[True, True, True], [True, False, True]])
        self.target = torch.tensor([[0.2, 0.3, 0.5], [0.5, 0., 0.5]])

    def active_head(self):
        h = NanoChoiceHead(8)
        # Nonzero residual tests actual mixing rather than the initialized bypass.
        torch.nn.init.normal_(h.set_output.weight, std=.1)
        return h

    def test_zero_residual_equals_scalar_exactly(self):
        h = NanoChoiceHead(8)
        expected = h.scalar(h.norm(self.x)).squeeze(-1).float()
        self.assertTrue(torch.equal(h(self.x, self.mask)[self.mask], expected[self.mask]))

    def test_matched_capacity_independent_mlp_control(self):
        scalar = NanoChoiceHead(8, "none")
        attention = NanoChoiceHead(8, "attention")
        independent = NanoChoiceHead(8, "independent_mlp")
        for head in (attention, independent):
            head.norm.load_state_dict(scalar.norm.state_dict())
            head.scalar.load_state_dict(scalar.scalar.state_dict())
            self.assertTrue(torch.equal(head(self.x, self.mask)[self.mask],
                                        scalar(self.x, self.mask)[self.mask]))
        size = lambda head: sum(p.numel() for p in head.parameters())
        self.assertLess(abs(size(attention) - size(independent)), 400)
        self.assertGreater(size(independent), size(scalar))

        torch.nn.init.normal_(independent.independent_output.weight, std=.1)
        independent.eval()
        original = independent(self.x, self.mask)
        changed = self.x.clone()
        changed[0, 1:] += 50
        self.assertEqual(original[0, 0], independent(changed, self.mask)[0, 0])
        permutation = [2, 0, 1]
        torch.testing.assert_close(independent(self.x[:, permutation], self.mask[:, permutation]),
                                   original[:, permutation])
        padded = torch.cat((self.x, torch.full((2, 2, 8), float("nan"))), dim=1)
        padded_mask = torch.cat((self.mask, torch.zeros(2, 2, dtype=torch.bool)), dim=1)
        torch.testing.assert_close(independent(padded, padded_mask)[:, :3], original)

    def test_independent_mlp_gradient_reaches_hidden_layer_after_first_step(self):
        head = NanoChoiceHead(8, "independent_mlp")
        optimizer = torch.optim.SGD(head.parameters(), lr=.1)
        choice_soft_target_loss(head(self.x, self.mask), self.target, self.mask).backward()
        self.assertGreater(head.independent_output.weight.grad.abs().sum().item(), 0.)
        self.assertEqual(head.independent_hidden.weight.grad.abs().sum().item(), 0.)
        optimizer.step(); optimizer.zero_grad()
        choice_soft_target_loss(head(self.x, self.mask), self.target, self.mask).backward()
        self.assertGreater(head.independent_hidden.weight.grad.abs().sum().item(), 0.)

    def test_permutation_equivariance_and_padding_nan_invariance(self):
        h = self.active_head().eval()
        z = h(self.x, self.mask)
        permutation = [2, 0, 1]
        torch.testing.assert_close(h(self.x[:, permutation], self.mask[:, permutation]), z[:, permutation])
        padded = torch.cat((self.x, torch.full((2, 2, 8), float("nan"))), dim=1)
        mask = torch.cat((self.mask, torch.zeros(2, 2, dtype=torch.bool)), dim=1)
        padded[1, 1] = float("inf")
        torch.testing.assert_close(h(padded, mask)[:, :3], z)
        self.assertTrue(torch.isneginf(h(padded, mask)[~mask]).all())

    def test_singleton_and_uniform_tie_targets(self):
        h = self.active_head()
        x = self.x[:, :1]
        m = torch.ones(2, 1, dtype=torch.bool)
        loss = choice_soft_target_loss(h(x, m), torch.ones(2, 1), m)
        self.assertEqual(loss.item(), 0.)
        tied = torch.zeros(2, 3, requires_grad=True)
        loss = choice_soft_target_loss(tied, self.target, self.mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tied.grad[1, 1], 0.)

    def test_first_and_second_step_gradient_flow(self):
        h = NanoChoiceHead(8)
        opt = torch.optim.SGD(h.parameters(), lr=.1)
        choice_soft_target_loss(h(self.x, self.mask), self.target, self.mask).backward()
        self.assertGreater(h.set_output.weight.grad.abs().sum().item(), 0.)
        self.assertEqual(h.set_project.weight.grad.abs().sum().item(), 0.)
        self.assertEqual(h.set_attention.in_proj_weight.grad.abs().sum().item(), 0.)
        opt.step(); opt.zero_grad()
        choice_soft_target_loss(h(self.x, self.mask), self.target, self.mask).backward()
        self.assertGreater(h.set_project.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(h.set_attention.in_proj_weight.grad.abs().sum().item(), 0.)

    def test_checkpoint_roundtrip_and_float32_output(self):
        for mode in ("none", "attention", "independent_mlp"):
            h = NanoChoiceHead(8, mode).double()
            z = h(self.x.double(), self.mask)
            self.assertEqual(z.dtype, torch.float32)
            b = io.BytesIO(); torch.save(h.state_dict(), b); b.seek(0)
            other = NanoChoiceHead(8, mode).double()
            other.load_state_dict(torch.load(b, weights_only=True))
            self.assertTrue(torch.equal(z, other(self.x.double(), self.mask)))

    def test_complete_question_gradient_accumulation_parity(self):
        h = self.active_head()
        other = copy.deepcopy(h)
        choice_soft_target_loss(h(self.x, self.mask), self.target, self.mask).backward()
        for i in range(2):
            (choice_soft_target_loss(other(self.x[i:i+1], self.mask[i:i+1]),
                                     self.target[i:i+1], self.mask[i:i+1]) / 2).backward()
        for a, b in zip(h.parameters(), other.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=2e-5)

    def test_invalid_inputs(self):
        h = NanoChoiceHead(8)
        for x, m in [(self.x, self.mask.float()), (self.x, torch.zeros_like(self.mask)),
                     (self.x[:, :, :7], self.mask), (self.x[0], self.mask),
                     (self.x[:0], self.mask[:0]), (self.x.long(), self.mask)]:
            with self.assertRaises(ValueError): h(x, m)
        bad = self.x.clone(); bad[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError): h(bad, self.mask)
        for width in (True, 0, -1, 2.5):
            with self.assertRaises(ValueError): NanoChoiceHead(width)
        with self.assertRaises(ValueError): NanoChoiceHead(8, "unknown")

    def test_loss_padding_gradients_and_invalid_targets(self):
        z = self.active_head()(self.x, self.mask).detach().requires_grad_()
        loss = choice_soft_target_loss(z, self.target, self.mask)
        reference = sum(-(self.target[i, self.mask[i]] * z[i, self.mask[i]].log_softmax(-1)).sum()
                        for i in range(2)) / 2
        torch.testing.assert_close(loss, reference)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(z.grad[1, 1], 0.)
        for target in [self.target * 2, torch.full_like(self.target, float("nan")),
                       -self.target, torch.ones_like(self.target), self.target[:, :2]]:
            with self.assertRaises(ValueError): choice_soft_target_loss(z, target, self.mask)
        invalid = z.detach().clone(); invalid[0, 0] = float("inf")
        with self.assertRaises(ValueError): choice_soft_target_loss(invalid, self.target, self.mask)


if __name__ == "__main__":
    unittest.main()

"""Standalone cardinality surrogate checks, not evidence of better trained masks."""
import unittest
import torch

from cbmjev.static_mask import cardinality_soft_gate, cardinality_st_topk, StaticGroupMask
from tests_cbmjev import test_static_mask as fixtures


class CardinalityGateTests(unittest.TestCase):
    def test_gradcheck_implicit_first_derivative(self):
        x = torch.tensor([-.8, .1, .4, 1.5, 2.], dtype=torch.float64, requires_grad=True)
        for k in (1, 2, 4):
            self.assertTrue(torch.autograd.gradcheck(
                lambda z: cardinality_soft_gate(z, k, temperature=.7), (x,), eps=1e-6, atol=1e-6, rtol=1e-4))

    def test_sum_shift_and_zero_common_gradient(self):
        x = torch.tensor([-5., -1., .4, 3., 8.], dtype=torch.float64, requires_grad=True)
        soft = cardinality_soft_gate(x, 2, temperature=1.3)
        self.assertAlmostEqual(float(soft.detach().sum()), 2., places=13)
        shifted = cardinality_soft_gate(x + 1234., 2, temperature=1.3)
        torch.testing.assert_close(soft, shifted, rtol=0, atol=1e-13)
        weights = torch.tensor([1., -2., 3., -4., 5.], dtype=torch.float64)
        gradient, = torch.autograd.grad((soft * weights).sum(), x, retain_graph=True)
        self.assertAlmostEqual(float(gradient.sum()), 0., places=13)
        null, = torch.autograd.grad(soft.sum(), x)
        torch.testing.assert_close(null, torch.zeros_like(null), rtol=0, atol=1e-14)

    def test_analytic_jacobian_and_symmetry(self):
        x = torch.tensor([-.2, .7, 1.3], dtype=torch.float64, requires_grad=True)
        temp = .9
        p = cardinality_soft_gate(x, 1, temperature=temp)
        s = p.detach() * (1 - p.detach())
        expected = (torch.diag(s) - s[:, None] * s[None, :] / s.sum()) / temp
        actual = torch.autograd.functional.jacobian(lambda z: cardinality_soft_gate(z, 1, temperature=temp), x)
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-14)

    def test_hard_forward_exact_and_legacy_unchanged(self):
        fixture = fixtures.StaticMaskTests()
        fixture.setUp()
        mask = StaticGroupMask(fixture.schema, 1, seed=7)
        for values in ([0., 0., 0.], [-.1, .7, .2]):
            with torch.no_grad():
                mask.logits.copy_(torch.tensor(values))
            hard = cardinality_st_topk(mask.logits, 1, seed=7)
            self.assertTrue(torch.equal(hard, mask()))
            self.assertEqual(float(hard.detach().sum()), 1.)
            self.assertTrue(all(v in (0., 1.) for v in hard.detach().tolist()))
            weights = torch.tensor([1., 3., 2.])
            gradient, = torch.autograd.grad((hard * weights).sum(), mask.logits)
            self.assertAlmostEqual(float(gradient.sum()), 0., places=6)
        # Existing module still uses independent sigmoid backward, not new projection.
        legacy, = torch.autograd.grad(mask().sum(), mask.logits)
        self.assertGreater(float(legacy.sum()), 0.)

    def test_extreme_finite_logits_and_endpoints(self):
        for values in ([-1e308, 0., 1e308], [1e308, 1e308, 1e308], [-1e6, -1e6, 1e6]):
            for k in range(4):
                x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
                p = cardinality_soft_gate(x, k, temperature=.01)
                self.assertTrue(torch.isfinite(p).all())
                self.assertAlmostEqual(float(p.detach().sum()), k, places=12)
                p.sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                if k in (0, 3):
                    self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))
        huge = cardinality_soft_gate(torch.tensor([-1e308, 1e308, 1e308], dtype=torch.float64), 1,
                                     temperature=1e308)
        ordinary = cardinality_soft_gate(torch.tensor([-1., 1., 1.], dtype=torch.float64), 1)
        torch.testing.assert_close(huge, ordinary, rtol=1e-13, atol=1e-14)

    def test_invalid_inputs_fail_closed(self):
        cases = [(torch.tensor([float('nan')]), 0, 1.), (torch.tensor([float('inf')]), 1, 1.),
                 (torch.tensor([1]), 1, 1.), (torch.tensor([]), 0, 1.),
                 (torch.tensor([1.]), True, 1.), (torch.tensor([1.]), 2, 1.),
                 (torch.tensor([1.]), 1, 0.), (torch.tensor([1.]), 1, float('nan'))]
        for x, k, t in cases:
            with self.assertRaises(ValueError):
                cardinality_soft_gate(x, k, temperature=t)

    def test_cli_and_module_variant_validation(self):
        from cbmjev.cli import parser
        p = parser()
        args = ["train-static-mask", "--cache", "c", "--models", "m", "--out", "o", "--k", "1"]
        self.assertEqual(p.parse_args(args).relaxation, "sigmoid")
        self.assertEqual(p.parse_args(args + ["--relaxation", "cardinality"]).relaxation, "cardinality")
        fixture = fixtures.StaticMaskTests()
        fixture.setUp()
        with self.assertRaises(ValueError):
            StaticGroupMask(fixture.schema, 1, relaxation="unknown")
        default = StaticGroupMask(fixture.schema, 1)
        explicit = StaticGroupMask(fixture.schema, 1, relaxation="sigmoid")
        self.assertTrue(torch.equal(default(), explicit()))


if __name__ == "__main__":
    unittest.main()

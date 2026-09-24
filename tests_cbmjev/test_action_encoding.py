import itertools
import unittest

import torch

from cbmjev.learning import encode_actions
from tests_cbmjev.test_crossfit_training import schema_fixture


def legacy(actions, schema, device):
    result = torch.zeros((len(actions), schema.num_groups + 1), device=device)
    for index, action in enumerate(actions):
        if not action:
            result[index, -1] = 1
        else:
            result[index, list(action)] = 1
    return result


class ActionEncodingTests(unittest.TestCase):
    def check_device(self, device):
        schema = schema_fixture()
        actions = [tuple(i for i, present in enumerate(mask) if present)
                   for mask in itertools.product((False, True), repeat=schema.num_groups)]
        for rows in ([], actions, actions * 8):
            actual = encode_actions(rows, schema, device)
            expected = legacy(rows, schema, device)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(actual.shape, (len(rows), schema.num_groups + 1))
        # Same downstream losses/gradients, not merely the same argmax.
        weight = torch.arange(schema.num_groups + 1, dtype=torch.float32,
                              device=device).requires_grad_()
        old = (legacy(actions, schema, device) @ weight).square().mean()
        new = (encode_actions(actions, schema, device) @ weight).square().mean()
        self.assertTrue(torch.equal(old, new))
        self.assertTrue(torch.equal(torch.autograd.grad(old, weight)[0],
                                    torch.autograd.grad(new, weight)[0]))

    def test_cpu_exact_outputs_losses_gradients(self):
        self.check_device("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_exact_outputs_losses_gradients(self):
        self.check_device("cuda:0")

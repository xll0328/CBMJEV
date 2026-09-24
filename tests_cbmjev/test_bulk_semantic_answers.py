import math
import unittest
from unittest.mock import Mock, patch

import torch

from cbmjev.contracts import ModelInput
from cbmjev.responders import CheapSession, _bulk_semantic_answers


def scalar_answers(distributions, counts, threshold):
    return tuple(count if threshold is not None and float(p.max()) < threshold
                 else int(p.argmax()) for p, count in zip(distributions, counts))


class BulkSemanticAnswerTests(unittest.TestCase):
    def check_device(self, device):
        for dtype in (torch.float16, torch.float32, torch.float64):
            logits = [torch.tensor(row, dtype=dtype, device=device) for row in
                      ([0., 0.], [-2., 4., 1.], [1.], [3., 3., 0., -9.], [float('nan'), 0.])]
            distributions = [p.softmax(-1) for p in logits]
            counts = [len(p) for p in distributions]
            boundary = float(distributions[0].max())
            for threshold in (None, 0., 1., .7, boundary,
                              math.nextafter(boundary, 1.), math.nextafter(boundary, 0.)):
                with self.subTest(device=device, dtype=dtype, threshold=threshold):
                    self.assertEqual(_bulk_semantic_answers(distributions, counts, threshold),
                                     scalar_answers(distributions, counts, threshold))
            self.assertEqual(_bulk_semantic_answers(distributions[:1], counts[:1], boundary), (0,))
            self.assertEqual(_bulk_semantic_answers(distributions[:1], counts[:1],
                                                   math.nextafter(boundary, 1.)), (2,))

    def test_exact_cpu_threshold_ties_heterogeneous_categories_and_unknown(self):
        self.check_device("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_exact_cuda(self):
        self.check_device("cuda")

    def test_constant_bulk_transfers_no_scalar_extractions(self):
        distributions = [torch.tensor([.25, .75])] * 312
        original_cpu = torch.Tensor.cpu
        calls = []

        def cpu(tensor, *args, **kwargs):
            calls.append(tuple(tensor.shape))
            return original_cpu(tensor, *args, **kwargs)

        for threshold, expected_transfers in ((None, 1), (.8, 2)):
            calls.clear()
            with patch.object(torch.Tensor, "cpu", cpu), \
                 patch.object(torch.Tensor, "__int__", side_effect=AssertionError("scalar int")), \
                 patch.object(torch.Tensor, "__float__", side_effect=AssertionError("scalar float")), \
                 patch.object(torch.Tensor, "item", side_effect=AssertionError("scalar item")):
                result = _bulk_semantic_answers(distributions, [2] * 312, threshold)
            self.assertEqual(result, (1 if threshold is None else 2,) * 312)
            self.assertEqual(calls, [(312,)] * expected_transfers)

    def test_session_preserves_cache_costs_and_empty_query(self):
        logits = [torch.tensor([[0., 0.]]), torch.tensor([[0., 2., 0.]])]

        class Model:
            value_counts = (2, 3)
            threshold = .6
            shared_cost_mode = "test_shared"
            eval = Mock()
            forward = Mock(return_value=logits)

            def __call__(self, payloads):
                return self.forward(payloads)

        model = Model()
        session = CheapSession(model, ModelInput(text="example"))
        self.assertEqual(session.respond(()), ())
        model.forward.assert_not_called()
        self.assertEqual(session.respond((1, 0)), (1, 2))
        self.assertEqual(session.last_stats, dict(encoder_forwards=1, atoms_computed=2,
                         images_encoded=0, cache_hit=False, cost_mode="test_shared"))
        self.assertEqual(session.respond((0,)), (2,))
        self.assertEqual(session.last_stats, dict(encoder_forwards=0, atoms_computed=0,
                         images_encoded=0, cache_hit=True, cost_mode="test_shared"))
        model.forward.assert_called_once()
        self.assertEqual(session.respond(()), ())
        self.assertEqual(session.last_stats["encoder_forwards"], 0)

    def test_empty_heads(self):
        self.assertEqual(_bulk_semantic_answers([], [], .5), ())


if __name__ == "__main__":
    unittest.main()

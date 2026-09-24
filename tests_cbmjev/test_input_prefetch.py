import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import torch

from cbmjev.contracts import Concept, ModelInput, QueryGroup, Schema
from cbmjev.input_prefetch import ordered_input_batches, validate_input_prefetch
from cbmjev.responders import ConceptTrainingExample, fit_text_responder


class InputPrefetchTests(unittest.TestCase):
    def test_parameters_reject_bool_float_and_invalid_bounds(self):
        for workers in (-1, True, 1.0, "2", None):
            with self.assertRaises(ValueError):
                validate_input_prefetch(workers, 2)
        for window in (0, -1, True, 2.0, None):
            with self.assertRaises(ValueError):
                validate_input_prefetch(0, window)
        validate_input_prefetch(0, 1)

    def test_sync_is_lazy_without_executor(self):
        examples = Mock()
        examples.__getitem__ = Mock(side_effect=lambda i: i)
        with patch("cbmjev.input_prefetch.ThreadPoolExecutor") as executor:
            with ordered_input_batches(examples, [(3, 1), (2,)]) as batches:
                self.assertEqual(examples.__getitem__.call_count, 0)
                self.assertEqual(next(batches), [3, 1])
                self.assertEqual(examples.__getitem__.call_count, 2)
            executor.assert_not_called()

    def test_out_of_order_completion_consumed_in_original_order(self):
        second_done = threading.Event()
        reads = []

        class Examples:
            def __getitem__(self, i):
                if i == 5:
                    if not second_done.wait(5):
                        raise RuntimeError("second read never started")
                if i == 2:
                    second_done.set()
                reads.append(threading.current_thread().name)
                return i

        with ordered_input_batches(Examples(), [(5,), (2,), (4, 1)],
                                   input_workers=2, input_prefetch_batches=2) as batches:
            self.assertEqual(list(batches), [[5], [2], [4, 1]])
        self.assertTrue(all(name.startswith("cbmjev-input") for name in reads))

    def test_submitted_not_consumed_futures_bounded(self):
        class ImmediateExecutor:
            def __init__(self, **kwargs):
                self.outstanding = self.maximum = 0
                self.shutdown = Mock()

            def submit(self, function, *args):
                self.outstanding += 1
                self.maximum = max(self.maximum, self.outstanding)
                result = function(*args)

                def consume():
                    self.outstanding -= 1
                    return result

                return Mock(result=consume)

        executor = ImmediateExecutor()
        with patch("cbmjev.input_prefetch.ThreadPoolExecutor", return_value=executor):
            with ordered_input_batches(list(range(100)), [(i,) for i in range(100)],
                                       input_workers=4, input_prefetch_batches=3) as batches:
                self.assertEqual(list(batches), [[i] for i in range(100)])
        self.assertEqual(executor.maximum, 3)
        self.assertEqual(executor.outstanding, 0)
        executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)

    def test_read_and_consumer_errors_close_executor(self):
        for origin in ("read", "consumer"):
            with self.subTest(origin=origin):
                executor = ThreadPoolExecutor(max_workers=2)
                shutdown = Mock(wraps=executor.shutdown)
                executor.shutdown = shutdown

                class Examples:
                    def __getitem__(self, i):
                        if origin == "read" and i == 1:
                            raise OSError("broken input")
                        return i

                with patch("cbmjev.input_prefetch.ThreadPoolExecutor", return_value=executor):
                    with self.assertRaisesRegex(OSError, "broken"):
                        with ordered_input_batches(Examples(), [(0,), (1,), (2,)],
                                                   input_workers=2) as batches:
                            self.assertEqual(next(batches), [0])
                            if origin == "consumer":
                                raise OSError("broken model")
                            next(batches)
                shutdown.assert_called_once_with(wait=True, cancel_futures=True)
                self.assertTrue(all(not thread.is_alive() for thread in executor._threads))

    def test_hashing_training_exact_weights_losses_and_missing_labels(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.addCleanup(torch.use_deterministic_algorithms,
                        torch.are_deterministic_algorithms_enabled(),
                        warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
        torch.set_num_threads(1)
        schema = Schema("synthetic_engineering", 2,
                        (Concept("a", "a", ("no", "yes")), Concept("b", "b", ("no", "yes"))),
                        (QueryGroup("a", (0,)), QueryGroup("b", (1,))))
        examples = [ConceptTrainingExample(ModelInput(text=f"text {i}"), labels)
                    for i, labels in enumerate([(0, 1), (1, None), (None, None), (None, 0), (1, 1)])]
        results = [fit_text_responder(examples, schema, epochs=3, batch_size=2, seed=17,
                                      vocab_size=32, embedding_dim=8, input_workers=workers,
                                      input_prefetch_batches=2) for workers in (0, 2)]
        self.assertEqual(results[0][1]["training_loss"], results[1][1]["training_loss"])
        self.assertEqual(results[0][1]["observed_labels_per_atom"], [3, 3])
        self.assertEqual(results[1][1]["input_workers"], 2)
        self.assertEqual(results[1][1]["input_prefetch_batches"], 2)
        for key, value in results[0][0].state_dict().items():
            self.assertTrue(torch.equal(value, results[1][0].state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()

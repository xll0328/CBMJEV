import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from cbmjev.brig import GroupQ, _terminal
from cbmjev.contracts import Concept, QueryGroup, Schema
from cbmjev.io import write_json
from scripts.brig_batched_rollout import empty_rollout_targets
from scripts.brig_cached_rollout_targets import precompute_empty_rollout_risks
from scripts.brig_cached_fit import fit_brig_cached
from scripts.brig_fast_fit import fit_brig_fast
from scripts.train_evaluate_cub_brig_cached_crossfit import (
    _write_sidecar, source_hashes, verify_cached_fold)


class DeterministicHead:
    def probabilities_many(self, states):
        return [(.7, .3) if sum(value >= 0 for value in state) % 2 == 0 else (.2, .8)
                for state in states]


class BRiGCachedRolloutTargetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.schema = Schema("toy", 2,
            tuple(Concept(f"c{i}", f"c{i}", ("no", "yes")) for i in range(4)),
            tuple(QueryGroup(f"g{i}", (i,)) for i in range(4)))
        torch.manual_seed(19)
        self.models = {}
        for budget in range(1, 4):
            model = GroupQ(self.schema, hidden=8)
            self.models[budget] = model.eval().requires_grad_(False)
        self.head = DeterministicHead()
        self.examples = [((0, 1, 1, 0), 0), ((1, 0, 0, 1), 1),
                         ((1, 1, 0, 0), 1)]

    def test_cache_matches_uncached_targets_and_preserves_example_action_order(self):
        cached = precompute_empty_rollout_risks(
            self.schema, self.models, self.head, self.examples,
            budget=4, device="cpu", chunk_size=2)
        uncached_batches = []
        for example in self.examples:
            _, actions, states, labels = empty_rollout_targets(
                self.schema, self.models, [example], 4)
            self.assertEqual(actions, list(range(self.schema.num_groups)))
            uncached_batches.append(_terminal(self.head, states, labels, "cpu"))
        expected = torch.stack(uncached_batches)
        self.assertTrue(torch.equal(cached, expected))

    def test_invalid_budget_or_chunk_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            precompute_empty_rollout_risks(
                self.schema, self.models, self.head, self.examples,
                budget=1)
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            precompute_empty_rollout_risks(
                self.schema, self.models, self.head, self.examples,
                budget=4, chunk_size=0)

    def test_cached_and_uncached_auxiliary_updates_have_identical_gradients(self):
        cached = precompute_empty_rollout_risks(
            self.schema, self.models, self.head, self.examples,
            budget=4, device="cpu", chunk_size=2)
        selected_indices = torch.tensor([2, 0])
        batch = [self.examples[i] for i in selected_indices.tolist()]
        _, actions, states, labels = empty_rollout_targets(
            self.schema, self.models, batch, 4)
        uncached_targets = _terminal(self.head, states, labels, "cpu")
        cached_targets = cached.index_select(0, selected_indices).reshape(-1)
        self.assertTrue(torch.equal(uncached_targets, cached_targets))

        uncached_model = GroupQ(self.schema, hidden=8)
        cached_model = GroupQ(self.schema, hidden=8)
        initial = {key: value.detach().clone()
                   for key, value in uncached_model.state_dict().items()}
        cached_model.load_state_dict(initial)
        empty_states = [self.schema.empty_state()] * len(actions)
        pred_a = uncached_model(empty_states, actions, 4)
        pred_b = cached_model(empty_states, actions, 4)
        loss_a = torch.nn.functional.mse_loss(pred_a, uncached_targets)
        loss_b = torch.nn.functional.mse_loss(pred_b, cached_targets)
        loss_a.backward()
        loss_b.backward()
        self.assertTrue(torch.equal(loss_a, loss_b))
        for left, right in zip(uncached_model.parameters(), cached_model.parameters()):
            self.assertTrue(torch.equal(left.grad, right.grad))

    def test_complete_multibudget_fit_matches_uncached_history_and_weights(self):
        rows = [dict(sample_id=str(i), group_id=str(i), split="policy_fit",
                     z=[i % 2, (i // 2) % 2, (i // 3) % 2, (i // 4) % 2],
                     y=i % 2) for i in range(8)]
        config = dict(seed=23, device="cpu", hidden=8, max_budget=4,
                      epochs=8, batch_size=4, learning_rate=.001)
        kwargs = dict(excluded_head_group_ids=[str(i) for i in range(8)])
        reference, reference_report = fit_brig_fast(
            rows, self.head, self.schema, config, **kwargs)
        cached, cached_report = fit_brig_cached(
            rows, self.head, self.schema, config, **kwargs)

        self.assertEqual(reference_report["history"], cached_report["history"])
        self.assertEqual(cached_report["event_digest_protocol"],
                         "cache-indexed-objective-v1")
        self.assertEqual(cached_report["cached_rollout_examples"], len(rows))
        self.assertEqual(len(cached_report["cached_rollout_target_sha256"]), 64)
        for budget in reference.models:
            for name, value in reference.models[budget].state_dict().items():
                self.assertTrue(torch.equal(
                    value, cached.models[budget].state_dict()[name]))

    def test_receipt_binds_cached_sources_base_receipt_and_checkpoint(self):
        with TemporaryDirectory() as root:
            directory = Path(root)
            write_json(directory / "receipt.json", {"status": "COMPLETE"})
            torch.save({"weight": torch.ones(1)}, directory / "brig.pt")
            _write_sidecar(directory, command="train-fold",
                           source_binding=source_hashes())
            self.assertEqual(verify_cached_fold(directory)["status"], "COMPLETE")
            torch.save({"weight": torch.zeros(1)}, directory / "brig.pt")
            with self.assertRaisesRegex(ValueError, "receipt/source/checkpoint"):
                verify_cached_fold(directory)


if __name__ == "__main__":
    unittest.main()

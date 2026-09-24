import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from cbmjev.brig import GroupQ
from cbmjev.contracts import Concept, QueryGroup, Schema
from cbmjev.io import write_json
from scripts.brig_batched_rollout import empty_rollout_targets
from scripts.brig_cached_rollout_targets import precompute_empty_rollout_risks
from scripts.brig_fast_fit import fit_brig_fast
from scripts.brig_trusted_ops import (
    TrustedTrainingGroupQ, empty_rollout_targets_trusted,
    precompute_empty_rollout_risks_trusted,
    precompute_empty_rollout_terminal_states_trusted)
from scripts.brig_cached_fit import fit_brig_cached
from scripts.brig_trusted_cached_fit import fit_brig_trusted_cached
from scripts.train_evaluate_cub_brig_trusted_cached_crossfit import (
    source_hashes, verify_trusted_cached_fold, _write_sidecar)


class DeterministicHead:
    def probabilities_many(self, states):
        return [(.7, .3) if sum(value >= 0 for value in state) % 2 == 0 else (.2, .8)
                for state in states]


class BRiGTrustedOpsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.schema = Schema("trusted-toy", 2,
            tuple(Concept(f"c{i}", f"c{i}", ("no", "yes")) for i in range(5)),
            tuple(QueryGroup(f"g{i}", (i,)) for i in range(5)))
        self.head = DeterministicHead()
        self.examples = [((i % 2, (i // 2) % 2, (i // 3) % 2,
                           (i // 4) % 2, (i // 5) % 2), i % 2)
                         for i in range(8)]

    def _models(self, count):
        torch.manual_seed(43)
        reference = {i: GroupQ(self.schema, 12).eval().requires_grad_(False)
                     for i in range(1, count + 1)}
        trusted = {}
        for budget, model in reference.items():
            candidate = TrustedTrainingGroupQ(self.schema, 12)
            candidate.load_state_dict(model.state_dict())
            trusted[budget] = candidate.eval().requires_grad_(False)
        return reference, trusted

    def test_trusted_forward_matches_validating_forward_outputs_and_gradients(self):
        states = [(-1, -1, -1, -1, -1)] * 5
        states += [(0, 1, -1, -1, -1)] * 3
        actions = [0, 1, 2, 3, 4, 2, 3, 4]
        reference = GroupQ(self.schema, 12)
        trusted = TrustedTrainingGroupQ(self.schema, 12)
        trusted.load_state_dict(reference.state_dict())
        output_a = reference(states, actions, 1)
        output_b = trusted.forward_trusted(states, actions, 1)
        self.assertTrue(torch.equal(output_a, output_b))
        output_a.square().sum().backward()
        output_b.square().sum().backward()
        for left, right in zip(reference.parameters(), trusted.parameters()):
            self.assertTrue(torch.equal(left.grad, right.grad))
        # External invalid inputs still fail through the public model API.
        with self.assertRaises(ValueError):
            trusted([(0, -1, -1)], [0], 1)

    def test_trusted_trajectories_and_risk_cache_match_reference(self):
        reference, trusted = self._models(4)
        for budget in range(2, 6):
            for chunk in (5, 17, 8192):
                with self.subTest(budget=budget, chunk=chunk):
                    old = empty_rollout_targets(
                        self.schema, reference, self.examples[:2], budget,
                        max_candidate_evals=chunk)
                    new = empty_rollout_targets_trusted(
                        self.schema, trusted, self.examples[:2], budget,
                        max_candidate_evals=chunk)
                    self.assertEqual(old, new)
            old_risks = precompute_empty_rollout_risks(
                self.schema, reference, self.head, self.examples, budget,
                chunk_size=3)
            new_risks = precompute_empty_rollout_risks_trusted(
                self.schema, trusted, self.head, self.examples, budget,
                chunk_size=3)
            self.assertTrue(torch.equal(old_risks, new_risks))

    def test_cached_terminal_states_keep_example_and_action_order(self):
        reference, trusted = self._models(3)
        for budget in (2, 3):
            cached = precompute_empty_rollout_terminal_states_trusted(
                self.schema, trusted, self.examples, budget, chunk_size=3)
            self.assertEqual(len(cached), len(self.examples))
            for index, example in enumerate(self.examples):
                _, _, states, labels = empty_rollout_targets(
                    self.schema, reference, [example], budget)
                self.assertEqual(cached[index], tuple(states))
                self.assertEqual(labels, [example[1]] * self.schema.num_groups)

    def test_non_singleton_group_schema_matches_exactly(self):
        schema = Schema("trusted-groups", 2,
            tuple(Concept(f"c{i}", f"c{i}", ("no", "yes")) for i in range(6)),
            (QueryGroup("g0", (0, 1)), QueryGroup("g1", (2,)),
             QueryGroup("g2", (3, 4, 5))))
        reference = {budget: GroupQ(schema, 12).eval().requires_grad_(False)
                     for budget in (1, 2)}
        trusted = {}
        for budget, model in reference.items():
            candidate = TrustedTrainingGroupQ(schema, 12)
            candidate.load_state_dict(model.state_dict())
            trusted[budget] = candidate.eval().requires_grad_(False)
        examples = [(tuple((i >> atom) & 1 for atom in range(6)), i % 2)
                    for i in range(8)]
        for chunk in (3, 7):
            self.assertEqual(
                empty_rollout_targets(schema, reference, examples[:3], 3,
                                      max_candidate_evals=chunk),
                empty_rollout_targets_trusted(schema, trusted, examples[:3], 3,
                                              max_candidate_evals=chunk))
        self.assertTrue(torch.equal(
            precompute_empty_rollout_risks(schema, reference, self.head,
                                           examples, 3, chunk_size=2),
            precompute_empty_rollout_risks_trusted(schema, trusted, self.head,
                                                   examples, 3, chunk_size=2)))

        rows = [dict(sample_id=str(i), group_id=str(i), split="policy_fit",
                     z=list(answer), y=label) for i, (answer, label) in enumerate(examples)]
        config = dict(seed=31, device="cpu", hidden=12, max_budget=3,
                      epochs=8, batch_size=4, learning_rate=.001)
        kwargs = dict(excluded_head_group_ids=[str(i) for i in range(8)])
        vanilla, vanilla_report = fit_brig_fast(rows, self.head, schema, config, **kwargs)
        cached, cached_report = fit_brig_cached(rows, self.head, schema, config, **kwargs)
        fast, fast_report = fit_brig_trusted_cached(rows, self.head, schema, config, **kwargs)
        self.assertEqual(vanilla_report["history"], fast_report["history"])
        self.assertEqual(cached_report["history"], fast_report["history"])
        self.assertEqual(fast_report["event_digest_protocol"],
                         "terminal-state-cache-minibatch-risk-v1")
        self.assertEqual(len(fast_report["cached_rollout_terminal_state_sha256"]), 64)
        for budget in vanilla.models:
            for name, value in vanilla.models[budget].state_dict().items():
                self.assertTrue(torch.equal(value, fast.models[budget].state_dict()[name]))
                self.assertTrue(torch.equal(value, cached.models[budget].state_dict()[name]))

    def test_complete_fit_matches_uncached_history_digest_and_weights(self):
        rows = [dict(sample_id=str(i), group_id=str(i), split="policy_fit",
                     z=list(answer), y=label)
                for i, (answer, label) in enumerate(self.examples)]
        config = dict(seed=23, device="cpu", hidden=12, max_budget=4,
                      epochs=8, batch_size=4, learning_rate=.001)
        kwargs = dict(excluded_head_group_ids=[str(i) for i in range(8)])
        reference, reference_report = fit_brig_fast(
            rows, self.head, self.schema, config, **kwargs)
        cached, cached_report = fit_brig_cached(
            rows, self.head, self.schema, config, **kwargs)
        trusted, trusted_report = fit_brig_trusted_cached(
            rows, self.head, self.schema, config, **kwargs)
        self.assertEqual(reference_report["history"], cached_report["history"])
        self.assertEqual(reference_report["history"], trusted_report["history"])
        self.assertEqual(trusted_report["event_digest_protocol"],
                         "terminal-state-cache-minibatch-risk-v1")
        self.assertEqual(len(trusted_report["cached_rollout_terminal_state_sha256"]), 64)
        for budget in reference.models:
            for name, value in reference.models[budget].state_dict().items():
                self.assertTrue(torch.equal(value, cached.models[budget].state_dict()[name]))
                self.assertTrue(torch.equal(value, trusted.models[budget].state_dict()[name]))

    def test_variant_receipt_binds_sources_and_checkpoint(self):
        with TemporaryDirectory() as root:
            directory = Path(root)
            write_json(directory / "receipt.json", {"status": "COMPLETE"})
            torch.save({"weight": torch.ones(1)}, directory / "brig.pt")
            _write_sidecar(directory, command="train-fold",
                           source_binding=source_hashes())
            receipt = verify_trusted_cached_fold(directory)
            self.assertEqual(receipt["status"], "COMPLETE")
            torch.save({"weight": torch.zeros(1)}, directory / "brig.pt")
            with self.assertRaisesRegex(ValueError, "receipt/source/checkpoint"):
                verify_trusted_cached_fold(directory)


if __name__ == "__main__":
    unittest.main()

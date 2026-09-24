import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from cbmjev.brig import BRiGPolicy, GroupQ, _reveal, fit_brig
from cbmjev.contracts import Concept, QueryGroup, Schema, stable_hash
from cbmjev.io import file_hash, write_json
from scripts.brig_batched_rollout import empty_rollout_targets
from scripts.brig_fast_fit import fit_brig_fast
from scripts.train_evaluate_cub_brig_fast_crossfit import source_hashes, verify_fast_fold


class BatchedRolloutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.schema = Schema("rollout", 2,
            tuple(Concept(str(i), "feature", ("0", "1")) for i in range(5)),
            tuple(QueryGroup(str(i), (i,)) for i in range(5)))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(901)
            self.models = {i: GroupQ(self.schema, 12).eval() for i in range(1, 5)}
        self.batch = [((0, 1, 0, 1, 0), 0), ((1, 0, 1, 0, 1), 1)]

    def _serial(self, budget):
        empty_states, empty_actions, terminal_states, terminal_labels = [], [], [], []
        policy = BRiGPolicy(self.schema, {i: self.models[i] for i in range(1, budget)})
        for answers, label in self.batch:
            for action in range(self.schema.num_groups):
                state = _reveal(self.schema, self.schema.empty_state(), action, answers)
                for left in range(budget - 1, 0, -1):
                    state = _reveal(self.schema, state, policy.choose(state, left)[0], answers)
                empty_states.append(self.schema.empty_state())
                empty_actions.append(action)
                terminal_states.append(state)
                terminal_labels.append(label)
        return empty_states, empty_actions, terminal_states, terminal_labels

    def test_matches_serial_for_every_budget_and_chunk_size(self):
        for budget in range(2, 6):
            for chunk in (5, 17, 8192):
                with self.subTest(budget=budget, chunk=chunk):
                    self.assertEqual(empty_rollout_targets(self.schema, self.models,
                        self.batch, budget, max_candidate_evals=chunk), self._serial(budget))

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            empty_rollout_targets(self.schema, self.models, self.batch, 1)
        with self.assertRaises(ValueError):
            empty_rollout_targets(self.schema, self.models, [], 2)
        with self.assertRaises(ValueError):
            empty_rollout_targets(self.schema, {}, self.batch, 2)
        with self.assertRaises(ValueError):
            empty_rollout_targets(self.schema, self.models, self.batch, 2,
                                  max_candidate_evals=4)

    def test_fit_matches_serial_training_events(self):
        class Head:
            def probabilities_many(self, states):
                return [(.8, .2) if state[0] >= 0 else (.4, .6) for state in states]

        rows = [dict(sample_id=str(i), group_id=str(i), split="policy_fit",
                     z=[i % 2] * 5, y=i % 2) for i in range(8)]
        config = dict(seed=17, device="cpu", hidden=12, max_budget=3,
                      epochs=8, batch_size=4, learning_rate=.001)
        serial, reference = fit_brig(rows, Head(), self.schema, config,
                                     excluded_head_group_ids=[str(i) for i in range(8)])
        batched, candidate = fit_brig_fast(rows, Head(), self.schema, config,
                                           excluded_head_group_ids=[str(i) for i in range(8)])
        self.assertEqual(candidate["event_sha256"], reference["event_sha256"])
        self.assertEqual(candidate["history"], reference["history"])
        for budget in serial.models:
            for key, value in serial.models[budget].state_dict().items():
                self.assertTrue(torch.equal(value, batched.models[budget].state_dict()[key]))

    def test_fast_sidecar_binds_sources_and_checkpoint(self):
        with TemporaryDirectory() as root:
            directory = Path(root)
            write_json(directory / "receipt.json", {"status": "COMPLETE"})
            torch.save({"weight": torch.ones(1)}, directory / "brig.pt")
            receipt = {"format": "cbmjev-brig-batched-rollout-fold-sidecar-v1",
                "status": "COMPLETE", "outer_fold": 0, "sources_sha256": source_hashes(),
                "base_receipt_sha256": file_hash(directory / "receipt.json"),
                "checkpoint_sha256": file_hash(directory / "brig.pt")}
            receipt["receipt_hash"] = stable_hash(receipt)
            write_json(directory / "fast_variant_receipt.json", receipt)
            self.assertEqual(verify_fast_fold(directory), receipt)
            torch.save({"weight": torch.zeros(1)}, directory / "brig.pt")
            with self.assertRaises(ValueError):
                verify_fast_fold(directory)


if __name__ == "__main__":
    unittest.main()

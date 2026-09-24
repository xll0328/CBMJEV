import copy
import inspect
import unittest

import torch
from torch import nn

from cbmjev.contracts import Concept, QueryGroup, Schema
from cbmjev.learning import MaskedHead, encode_states, mask_answers, normalize_config
from cbmjev.static_mask import StaticGroupMask, fit_static_mask


class StaticMaskTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.schema = Schema("grouped", 2,
                             tuple(Concept(str(i), "atom", ("no", "yes")) for i in range(4)),
                             (QueryGroup("pair", (0, 1)), QueryGroup("a", (2,)), QueryGroup("b", (3,))))
        cfg = normalize_config(dict(hidden=8), self.schema)
        self.head = MaskedHead(self.schema, cfg)
        width = sum(self.schema.num_categories) + self.schema.num_atoms
        self.head.network = nn.Linear(width, 2)
        with torch.no_grad():
            self.head.network.weight[0].fill_(.1)
            self.head.network.weight[1].fill_(.4)
            self.head.network.bias.zero_()
        self.rows = [dict(sample_id=str(i), group_id=str(i), split="policy_fit",
                          z=[i % 4, (i + 1) % 4, 0, 1], y=i % 2) for i in range(6)]
        self.excluded = [str(i) for i in range(6)]

    def test_hard_forward_matches_original_encoded_semantics(self):
        mask = StaticGroupMask(self.schema, 1)
        with torch.no_grad():
            mask.logits.copy_(torch.tensor([3., 1., -2.]))
        responses = [r["z"] for r in self.rows]
        observed = [mask_answers(z, (True, False, False), self.schema) for z in responses]
        actual = mask.encode_training_responses(responses)
        expected = encode_states(observed, self.schema)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(self.head.network(actual), self.head.network(expected)))
        self.assertEqual(mask.selected_groups(), (0,))
        self.assertEqual(tuple(inspect.signature(mask.forward).parameters), ())
        self.assertEqual(tuple(inspect.signature(mask.selected_groups).parameters), ())

    def test_gate_gradient_and_frozen_head(self):
        before = copy.deepcopy(self.head.network.state_dict())
        mode = self.head.network.training
        flags = [p.requires_grad for p in self.head.network.parameters()]
        mask, report = fit_static_mask(self.rows, self.head, self.schema,
                                       dict(k=1, epochs=3, batch_size=4, learning_rate=.02),
                                       excluded_head_group_ids=self.excluded)
        self.assertGreater(float(mask.logits.grad.abs().sum()), 0.)
        self.assertTrue(any(float(v) != 0 for v in mask.logits.detach()))
        self.assertEqual(report["optimizer_steps"], 6)
        self.assertEqual(report["event_exposures"], 18)
        self.assertEqual(len(mask.selected_groups()), 1)
        self.assertEqual(self.head.network.training, mode)
        self.assertEqual([p.requires_grad for p in self.head.network.parameters()], flags)
        for k, v in before.items():
            self.assertTrue(torch.equal(self.head.network.state_dict()[k], v))
        self.assertTrue(all(p.grad is None for p in self.head.network.parameters()))
        selected = mask.selected_groups()
        mask.encode_training_responses([[3, 3, 3, 3]])
        self.assertEqual(mask.selected_groups(), selected)

    def test_endpoints_skip_optimization(self):
        for k in (0, 3):
            mask, report = fit_static_mask(self.rows, self.head, self.schema, dict(k=k),
                                           excluded_head_group_ids=self.excluded)
            self.assertEqual(len(mask.selected_groups()), k)
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertEqual(report["effective_epochs"], 0)
            self.assertFalse(mask.logits.requires_grad)
            expected = encode_states([[-1] * 4 if k == 0 else self.rows[0]["z"]], self.schema)
            self.assertTrue(torch.equal(mask.encode_training_responses([self.rows[0]["z"]]), expected))

    def test_split_and_exclusion_boundaries(self):
        with self.assertRaisesRegex(ValueError, "exclude"):
            fit_static_mask(self.rows, self.head, self.schema, excluded_head_group_ids=[])
        overlap = dict(self.rows[0], sample_id="head", split="head_fit")
        with self.assertRaisesRegex(ValueError, "overlaps"):
            fit_static_mask(self.rows + [overlap], self.head, self.schema,
                            excluded_head_group_ids=self.excluded)
        poisoned = dict(split="test", z=object(), y=object())
        a, ra = fit_static_mask(self.rows, self.head, self.schema, dict(epochs=1),
                                excluded_head_group_ids=self.excluded)
        b, rb = fit_static_mask(self.rows + [poisoned], self.head, self.schema, dict(epochs=1),
                                excluded_head_group_ids=self.excluded)
        self.assertTrue(torch.equal(a.logits, b.logits))
        self.assertEqual(ra, rb)

    def test_ties_seeded_and_invalid_configs(self):
        a, b = StaticGroupMask(self.schema, 2, seed=3), StaticGroupMask(self.schema, 2, seed=3)
        self.assertEqual(a.selected_groups(), b.selected_groups())
        for kw in (dict(k=4), dict(k=True), dict(k=1, temperature=0)):
            with self.assertRaises(ValueError):
                StaticGroupMask(self.schema, **kw)


if __name__ == "__main__":
    unittest.main()

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from cbmjev import asap_data
from scripts.evaluate_asap_gold_branch import (
    PartialHead, encode, family_bootstrap_delta, greedy_orders, internal_role,
    sampled_masks, trajectory_audit,
)


def write_split(root, name, records):
    with (Path(root) / (name + ".csv")).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=asap_data.HEADER)
        writer.writeheader()
        for sid, review, star, aspects in records:
            writer.writerow(dict(zip(asap_data.HEADER, (sid, review, star) + tuple(aspects))))


class ASAPGoldBranchTests(unittest.TestCase):
    def test_not_mentioned_is_observable_and_unqueried_values_hidden(self):
        z = np.asarray([[-2] * 18, [1] * 18], dtype=np.int8)
        mask = np.zeros((2, 18), dtype=bool)
        np.testing.assert_array_equal(encode(z, mask)[0], encode(z, mask)[1])
        mask[:, 3] = True
        self.assertNotEqual(encode(z, mask)[0, 3 * 5 + 1],
                            encode(z, mask)[1, 3 * 5 + 1])
        self.assertEqual(encode(z, mask)[0, 3 * 5 + 1], 1.)

    def test_group_role_deterministic(self):
        self.assertEqual(internal_role("same review"), internal_role("same review"))
        self.assertIn(internal_role("same review"), ("head_fit", "policy_fit", "tune"))

    def test_test_labels_never_loaded_and_cross_split_duplicate_purged(self):
        with TemporaryDirectory() as root:
            saved = asap_data.EXPECTED_COUNTS
            asap_data.EXPECTED_COUNTS = {"train": 2, "dev": 1, "test": 1}
            try:
                a = (-2,) * 18
                write_split(root, "train", [("a", "相同　评论", 3., a),
                                             ("b", "另一个评论", 4., a)])
                write_split(root, "dev", [("c", "相同 评论", 5., a)])
                # The test target and concepts are deliberately invalid. Only
                # identity/text enter development duplicate checks.
                write_split(root, "test", [("d", "另一个评论", "unread", ("unread",) * 18)])
                train, dev, report = asap_data.load_development(root)
                self.assertEqual(len(train), 0)
                self.assertEqual(len(dev), 1)
                self.assertFalse(report["test_labels_loaded"])
                self.assertEqual(report["cross_split_group_overlap"]["train_dev"], 1)
                self.assertEqual(report["cross_split_group_overlap"]["train_test"], 1)
            finally:
                asap_data.EXPECTED_COUNTS = saved

    def test_sampled_masks_have_requested_cardinalities(self):
        masks = sampled_masks(100, np.random.default_rng(40), (0, 2, 4, 8, 12, 18))
        self.assertTrue(set(masks.sum(axis=1)).issubset({0, 2, 4, 8, 12, 18}))

    def test_food_exclusion_refits_order_on_seventeen_legal_actions(self):
        z = np.asarray([[-2] * 18, [1] * 18], dtype=np.int8)
        data = (z, np.asarray([0, 4]), ["a", "b"])
        orders, meta = greedy_orders(PartialHead(), data, starts=2, excluded=17)
        self.assertEqual(meta["excluded_aspect"], 17)
        self.assertTrue(all(len(order) == 17 and 17 not in order for order in orders))

    def test_paired_family_delta_and_branching(self):
        y = np.asarray([0, 0, 1])
        fixed_pred = np.asarray([1, 1, 1])
        adaptive_pred = np.asarray([0, 0, 1])
        fm = np.zeros((3, 18), dtype=bool)
        am = fm.copy()
        am[:, 0] = True
        delta = family_bootstrap_delta(["family1", "family1", "family2"], y,
                                       fixed_pred, fm, adaptive_pred, am, .18,
                                       seed=17, replicates=100)
        self.assertAlmostEqual(delta["adaptive_minus_fixed_error"], -2 / 3)
        self.assertAlmostEqual(delta["adaptive_minus_fixed_J"], -2 / 3 + .01)
        self.assertLess(delta["J_ci95"][0], delta["J_ci95"][1])
        paths = np.full((3, 18), -1, dtype=np.int8)
        paths[:, 0] = 0
        paths[0, 1] = 1
        paths[1, 1] = 2
        z = np.zeros((3, 18), dtype=np.int8)
        z[1, 0] = 1
        audit = trajectory_audit(paths, z)
        self.assertGreaterEqual(audit["branching_prefix_count"], 1)


if __name__ == "__main__":
    unittest.main()

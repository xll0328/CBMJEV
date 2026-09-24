import unittest

from scripts.evaluate_cub_gold_order_controls import (
    order_controls, summarize_random_orders)


class GoldOrderControlTests(unittest.TestCase):
    def test_predeclared_random_orders_reproduce(self):
        first = order_controls(5, [4, 1, 3, 0, 2],
                               random_seed=123, num_random=4)
        second = order_controls(5, [4, 1, 3, 0, 2],
                                random_seed=123, num_random=4)
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], [0, 1, 2, 3, 4])
        self.assertEqual(first["reverse_schema"], [4, 3, 2, 1, 0])
        self.assertEqual(first["auto_oof"], [4, 1, 3, 0, 2])
        self.assertEqual(len(first), 7)
        for order in first.values():
            self.assertEqual(sorted(order), [0, 1, 2, 3, 4])

    def test_summary_correct_and_ce_have_opposite_better_direction(self):
        results = {
            "random_00": {"16": {"correct": {"gold_minus_automatic": .1},
                                  "ce": {"gold_minus_automatic": -.2}}},
            "random_01": {"16": {"correct": {"gold_minus_automatic": -.1},
                                  "ce": {"gold_minus_automatic": .2}}}}
        summary = summarize_random_orders(results)
        self.assertEqual(summary["num_predeclared_orders"], 2)
        self.assertEqual(summary["correct"]["gold_better_count"], 1)
        self.assertEqual(summary["ce"]["gold_better_count"], 1)
        self.assertEqual(summary["correct"]["mean"], 0)


if __name__ == "__main__":
    unittest.main()

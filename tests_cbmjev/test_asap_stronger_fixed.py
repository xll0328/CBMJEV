import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts.evaluate_asap_gold_branch import PartialHead
from scripts.evaluate_asap_stronger_fixed import (
    BUDGET, MAX_EVALUATIONS, search_sets, set_mask,
)


class ASAPStrongerFixedTests(unittest.TestCase):
    def test_bounded_static_set_search_keeps_eight_legal_aspects(self):
        z = np.asarray([[-2] * 18, [1] * 18], dtype=np.int8)
        data = (z, np.asarray([0, 4]), ["a", "b"])
        starting_orders = [tuple(range(18)), tuple(reversed(range(18)))]
        candidates, evaluated = search_sets(PartialHead(), data, starting_orders)
        self.assertEqual(len(candidates), 2)
        self.assertLessEqual(evaluated, MAX_EVALUATIONS)
        for candidate in candidates:
            selected = candidate["selected"]
            self.assertEqual(len(selected), BUDGET)
            self.assertEqual(len(set(selected)), BUDGET)
            self.assertTrue(np.all(set_mask(data, selected).sum(axis=1) == BUDGET))


if __name__ == "__main__":
    unittest.main()

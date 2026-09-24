import json
import unittest

from cbmjev.contracts import stable_hash
from scripts.diagnose_cub_k16_last_query_headroom import summarize_headroom


class HeadroomTests(unittest.TestCase):
    def test_hindsight_summary(self):
        rows = [
            {"ce": {0: 1.0, 1: .5}, "correct": {0: 0, 1: 1}},
            {"ce": {0: .2, 1: .8}, "correct": {0: 1, 1: 0}},
        ]
        got = summarize_headroom(rows, (0, 1), 0)
        self.assertEqual(got["samples"], 2)
        self.assertAlmostEqual(got["fixed_ce"], .6)
        self.assertAlmostEqual(got["label_clairvoyant_min_ce"], .35)
        self.assertAlmostEqual(got["fixed_accuracy"], .5)
        self.assertAlmostEqual(got["label_clairvoyant_any_correct_accuracy"], 1)
        self.assertEqual(got["ce_winner_counts"], {"0": 1, "1": 1})
        self.assertEqual(stable_hash(got),
                         stable_hash(json.loads(json.dumps(got, sort_keys=True))))

    def test_rejects_incomplete_scores(self):
        with self.assertRaises(ValueError):
            summarize_headroom([{"ce": {0: 1.}, "correct": {0: 1, 1: 0}}],
                               (0, 1), 0)

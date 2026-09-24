import unittest

from scripts.diagnose_cub_gold_nll_tails import quantiles, summarize


class GoldNllTailTests(unittest.TestCase):
    def test_quantiles_and_paired_error_accounting(self):
        self.assertEqual(quantiles([3, 1, 2])["max"], 3)
        entries = [
            {"automatic": {"nll": 2., "correct": 0, "confidence": .7,
                           "brier": .9},
             "gold": {"nll": 1., "correct": 1, "confidence": .6,
                      "brier": .5}},
            {"automatic": {"nll": 1., "correct": 1, "confidence": .6,
                           "brier": .5},
             "gold": {"nll": 3., "correct": 0, "confidence": .9,
                      "brier": 1.1}}]
        result = summarize(entries)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["paired"]["gold_correct_auto_wrong"], 1)
        self.assertEqual(result["paired"]["auto_correct_gold_wrong"], 1)
        self.assertEqual(result["paired"]["gold_higher_nll_cases"], 1)
        self.assertEqual(result["paired"]["gold_lower_nll_cases"], 1)
        self.assertEqual(result["paired"]["top5_positive_delta_fraction_of_positive_mass"], 1)
        self.assertEqual(result["paired"]["gold_minus_auto_mean_nll"], .5)


if __name__ == "__main__":
    unittest.main()

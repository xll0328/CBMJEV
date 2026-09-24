import unittest

from scripts.evaluate_oof_conditional_second_query import evaluate


class OOFConditionalSecondQueryTest(unittest.TestCase):
    def test_training_only_policy_and_validation_outcomes(self):
        train = [{"outcome": (i % 2,),
                  "ce": {0: .1 if i % 2 == 0 else 1.,
                         1: .1 if i % 2 == 1 else 1.}}
                 for i in range(10)]
        val = [{"outcome": (i % 2,),
                "ce": {0: .1 if i % 2 == 0 else 1.,
                       1: .1 if i % 2 == 1 else 1.},
                "correct": {0: int(i % 2 == 0), 1: int(i % 2 == 1)}}
               for i in range(10)]
        result = evaluate(train, val, (0, 1), smoothing=1, seed=9)
        self.assertLess(result["aggregate"]["adaptive_minus_fixed_ce"], 0)
        self.assertEqual(result["unseen_validation_outcome_count"], 0)


if __name__ == "__main__":
    unittest.main()

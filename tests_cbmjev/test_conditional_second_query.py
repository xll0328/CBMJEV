import unittest

from scripts.diagnose_conditional_second_query import analyze, choose_second


class ConditionalSecondQueryTest(unittest.TestCase):
    def test_outcome_crossing_improves_heldout(self):
        rows = []
        for index in range(100):
            outcome = (index % 2,)
            good = outcome[0]
            rows.append({"group_id": f"case:{index}", "outcome": outcome,
                         "ce": {0: 0.1 if good == 0 else 1.0,
                                1: 0.1 if good == 1 else 1.0},
                         "correct": {0: int(good == 0), 1: int(good == 1)}})
        result = analyze(rows, (0, 1), smoothing=2, seed=7)
        self.assertLess(result["aggregate"]["adaptive_minus_fixed_ce"], 0)
        self.assertGreater(result["aggregate"]["adaptive_minus_fixed_accuracy"], 0)
        self.assertLess(result["outcome_shuffle_control"]["observed_adaptive_minus_fixed_ce"],
                        result["outcome_shuffle_control"]["shuffled_mean"])

    def test_no_crossing_collapses_to_fixed(self):
        rows = [{"outcome": (i % 2,), "ce": {0: 0.2, 1: 0.9}}
                for i in range(10)]
        fixed, policy, _, _ = choose_second(rows, (0, 1), smoothing=1)
        self.assertEqual(fixed, 0)
        self.assertEqual(set(policy.values()), {0})


if __name__ == "__main__":
    unittest.main()

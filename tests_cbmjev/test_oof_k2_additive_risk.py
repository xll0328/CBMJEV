import unittest

import numpy as np

from scripts.evaluate_oof_k2_additive_risk import (
    choose_alpha, evaluate, feature_matrix, fit_ridge,
)


class OOFK2AdditiveRiskTest(unittest.TestCase):
    def test_training_only_additive_risk_learns_answer_dependent_action(self):
        outcomes = [(i % 2,) for i in range(40)]
        x = feature_matrix(outcomes, (2,))
        y = np.asarray([[0.1, 2.0] if state == (0,) else [2.0, 0.1]
                        for state in outcomes])
        # Keep each held-out fold representative of both answer categories.
        folds = [(i // 2) % 4 for i in range(40)]
        alpha, scores = choose_alpha(x, y, folds)
        self.assertIn(alpha, (1.0, 10.0, 100.0, 1000.0))
        self.assertEqual(len(scores), 4)
        predictions = x @ fit_ridge(x, y, alpha)
        rows = [{"outcome": state, "ce": {1: losses[0], 2: losses[1]},
                 "correct": {1: int(state == (0,)),
                             2: int(state == (1,))}}
                for state, losses in zip(outcomes, y)]
        result = evaluate(rows, (1, 2), predictions, fixed=1,
                          lookup={(0,): 1, (1,): 2})
        self.assertLess(result["additive_vs_fixed"]["adaptive_minus_fixed_ce"], -0.8)
        self.assertAlmostEqual(result["additive_vs_lookup"]["adaptive_minus_fixed_ce"], 0.0)

    def test_invalid_observation_and_alpha_fail_closed(self):
        with self.assertRaises(ValueError):
            feature_matrix([(2,)], (2,))
        with self.assertRaises(ValueError):
            fit_ridge(np.ones((2, 2)), np.ones((2, 1)), 0)


if __name__ == "__main__":
    unittest.main()

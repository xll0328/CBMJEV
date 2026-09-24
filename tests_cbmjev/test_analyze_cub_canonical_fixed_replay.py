import unittest

from scripts.analyze_cub_canonical_fixed_replay import (
    BUDGETS, POLICIES, SEEDS, aggregate,
)


class AnalyzeCanonicalFixedReplayTests(unittest.TestCase):
    def test_aggregate_keeps_seed_pairs_and_reports_descriptive_deltas(self):
        rows = []
        for index, seed in enumerate(SEEDS):
            for budget in BUDGETS:
                fixed = 0.5 + index * 0.01
                for policy in POLICIES:
                    acc = fixed + (0.02 if policy == "value" else 0.0)
                    rows.append({
                        "seed": seed,
                        "budget": budget,
                        "policy": policy,
                        "accuracy": acc,
                        "macro_f1": acc - 0.1,
                        "mean_queried_groups": float(budget),
                    })
        result = aggregate(rows)
        delta = result["accuracy_deltas_vs_canonical_fixed"]["value"]["8"]
        self.assertEqual(delta["accuracy_wins"], 4)
        self.assertEqual(delta["accuracy_ties"], 0)
        self.assertAlmostEqual(delta["accuracy_pp"]["mean"], 2.0)
        self.assertAlmostEqual(delta["accuracy_pp"]["sd"], 0.0)
        self.assertAlmostEqual(result["policies"]["canonical_fixed"]["8"]["accuracy"]["mean"], 0.515)


if __name__ == "__main__":
    unittest.main()

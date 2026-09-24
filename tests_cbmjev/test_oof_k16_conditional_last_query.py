import json
import unittest

from cbmjev.contracts import Concept, QueryGroup, Schema, stable_hash
from scripts.evaluate_oof_k16_conditional_last_query import (
    canonicalize_result, score_rows, static_comparisons)


class TinyHead:
    def probabilities_many(self, states):
        # Outcome depends on the first acquired atom, not the hidden last atom.
        return [(0.8, 0.2) if state[0] == 0 else (0.2, 0.8)
                for state in states]


class OofK16LastQueryTests(unittest.TestCase):
    def test_integer_risk_keys_canonicalize_before_report_hash(self):
        result = canonicalize_result({"global_train_ce_by_group":
                                      {2: 1.0, 10: 2.0}, "other": "ok"})
        self.assertEqual(stable_hash(result), stable_hash(json.loads(json.dumps(result))))
        self.assertEqual(result["global_train_ce_by_group"],
                         {"2": 1.0, "10": 2.0})

    def test_prefix_outcome_cannot_see_unqueried_answer(self):
        schema = Schema("tiny", 2,
            tuple(Concept(f"c{i}", f"c{i}", ("no", "yes")) for i in range(16)),
            tuple(QueryGroup(f"g{i}", (i,)) for i in range(16)))
        rows = [{"group_id": "a", "z": [1] + [0] * 15, "y": 1},
                {"group_id": "b", "z": [1] + [0] * 14 + [1], "y": 1}]
        scored = score_rows(rows, TinyHead(), schema, tuple(range(15)),
                            (15,), batch_size=1)
        self.assertEqual([row["outcome"] for row in scored], [1, 1])
        self.assertEqual(scored[0]["ce"][15], scored[1]["ce"][15])
        extra = static_comparisons(scored, {1: 15}, 15, 15, seed=2)
        self.assertEqual(extra["conditional_vs_original_static"]
                         ["adaptive_minus_fixed_ce"], 0)


if __name__ == "__main__":
    unittest.main()

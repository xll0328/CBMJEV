"""Frozen-head repair/damage E1 diagnostic checks."""
import unittest

from cbmjev.contracts import Concept, QueryGroup, Schema
from scripts.analyze_cebab_repair_damage import diagnose_rows


class DeterministicHead:
    def probabilities_many(self, states):
        # Group 0 repairs high-confidence errors; group 1 damages correct
        # predictions. This deliberately decouples confidence from correctness.
        result = []
        for state in states:
            if state[1] == 1:
                result.append((0.9, 0.1))
            elif state[0] == 1:
                result.append((0.1, 0.9))
            else:
                result.append((0.9, 0.1))
        return result


class RepairDamageTests(unittest.TestCase):
    def setUp(self):
        self.schema = Schema("cebab", 2,
            tuple(Concept(str(i), str(i), ("no", "yes")) for i in range(4)),
            tuple(QueryGroup(str(i), (i,)) for i in range(4)))
        self.rows = [{"sample_id": "a", "group_id": "family:a", "split": "validation",
                      "z": [1, 1, 0, 0], "y": 1},
                     {"sample_id": "b", "group_id": "family:b", "split": "test",
                      "z": [1, 1, 0, 0], "y": 0}]

    def test_enumerates_all_partial_states_and_signed_transitions(self):
        transitions, summary = diagnose_rows(self.rows, self.schema, DeterministicHead())
        self.assertEqual(len(transitions), 32)
        self.assertEqual(summary["partial_states"], 15)
        self.assertEqual(summary["samples"], 1)
        self.assertEqual(summary["distinct_families"], 1)
        self.assertEqual({r["split"] for r in transitions}, {"validation"})
        repair = next(r for r in transitions if r["mask"] == 0 and r["action_group"] == 0)
        self.assertEqual(repair["transition"], "repair")
        self.assertEqual(repair["signed_01_gain"], 1)
        self.assertAlmostEqual(repair["gold_probability_gain"], 0.8)
        self.assertGreater(repair["signed_ce_gain"], 0)
        self.assertEqual(repair["confidence_gain"], 0)
        damage = next(r for r in transitions if r["mask"] == 1 and r["action_group"] == 1)
        self.assertEqual(damage["transition"], "damage")
        self.assertEqual(damage["signed_01_gain"], -1)
        self.assertLess(damage["signed_ce_gain"], 0)
        self.assertEqual(sum(summary["all_transitions"]["counts"].values()), 32)
        self.assertEqual(summary["high_confidence"]["wrong_repairable_states"], 4)
        self.assertFalse(summary["high_confidence"]["sufficient_family_count_for_subgroup_claim"])
        self.assertEqual(summary["confidence_distribution"]["all"]["states"], 15)
        self.assertEqual(summary["confidence_distribution"]["wrong"]["min"], 0.9)
        self.assertEqual(summary["confidence_distribution"]["correct"]["max"], 0.9)

    def test_test_split_rejected_and_missing_split_rejected(self):
        with self.assertRaisesRegex(ValueError, "never test"):
            diagnose_rows(self.rows, self.schema, DeterministicHead(), split="test")
        with self.assertRaisesRegex(ValueError, "empty"):
            diagnose_rows(self.rows, self.schema, DeterministicHead(), split="policy_fit")

    def test_limit_and_non_cebab_schema_guard(self):
        rows = self.rows[:1] * 2
        transitions, summary = diagnose_rows(rows, self.schema, DeterministicHead(), limit=1)
        self.assertEqual(len(transitions), 32)
        self.assertEqual(summary["original_split_samples"], 2)
        self.assertEqual(summary["sampled_limit"], 1)
        with self.assertRaisesRegex(ValueError, "positive"):
            diagnose_rows(rows, self.schema, DeterministicHead(), limit=0)


if __name__ == "__main__":
    unittest.main()

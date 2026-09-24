import json
from pathlib import Path
import tempfile
import unittest

from cbmjev.contracts import Concept, QueryGroup, Schema
from scripts.evaluate_cub_paired_gold_intervention import (
    complete_gold_validation, paired_scores, summarize_paired)


class TinyHead:
    def probabilities_many(self, states):
        return [(0.9, 0.1) if state[0] == 0 else (0.1, 0.9)
                for state in states]


class PairedGoldInterventionTests(unittest.TestCase):
    def setUp(self):
        self.schema = Schema("tiny", 2,
            (Concept("a", "a", ("no", "yes")),
             Concept("b", "b", ("no", "yes"))),
            (QueryGroup("a", (0,)), QueryGroup("b", (1,))))
        self.rows = [
            {"sample_id": "x", "group_id": "gx", "y": 1, "z": [0, 0]},
            {"sample_id": "y", "group_id": "gy", "y": 0, "z": [0, 0]}]

    def test_complete_case_pairing_and_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = Path(temporary)
            records = []
            for sid, y, statuses, values in (
                    ("x", 1, ("OBSERVED", "OBSERVED"), (1, 0)),
                    ("y", 0, ("OBSERVED", "NOT_VISIBLE"), (0, None))):
                records.append({"sample_id": sid, "group_id": "g" + sid,
                    "split": "validation", "target": {"status": "OBSERVED", "value": y},
                    "concepts": [{"concept_id": concept, "annotation_status": status,
                                  "value": value} for concept, status, value in
                                 zip(("a", "b"), statuses, values)]})
            records.append({"sample_id": "hidden-test", "split": "test"})
            (prepared / "samples.jsonl").write_text(
                "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
            gold = complete_gold_validation(prepared, self.schema, self.rows)
        self.assertEqual(gold, {"x": (1, 0)})
        outcomes = paired_scores(self.rows, gold, self.schema, TinyHead(),
                                 [0, 1], [1, 2], batch_size=2)
        self.assertEqual(len(outcomes[1]), 1)
        self.assertEqual(outcomes[1][0]["automatic"]["correct"], 0)
        self.assertEqual(outcomes[1][0]["gold"]["correct"], 1)
        self.assertEqual(outcomes[2][0]["gold"]["correct"], 1)
        stats = summarize_paired(outcomes[1], seed=3, replicates=20)
        self.assertEqual(stats["correct"]["gold_minus_automatic"], 1)
        self.assertLess(stats["ce"]["gold_minus_automatic"], 0)

    def test_validation_membership_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = Path(temporary)
            (prepared / "samples.jsonl").write_text(json.dumps({
                "sample_id": "unexpected", "group_id": "other", "split": "validation"}) + "\n")
            with self.assertRaises(ValueError):
                complete_gold_validation(prepared, self.schema, self.rows)


if __name__ == "__main__":
    unittest.main()

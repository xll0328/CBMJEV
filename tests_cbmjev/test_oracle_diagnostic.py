import unittest

from cbmjev.contracts import Concept, QueryGroup, Schema
from tools.oracle_concept_diagnostic import gold_rows


class OracleDiagnosticTests(unittest.TestCase):
    def test_requires_observed_target_and_every_concept(self):
        schema = Schema(dataset="tiny", num_classes=2,
            concepts=(Concept("a", "a", ("n", "y")), Concept("b", "b", ("n", "y"))),
            groups=(QueryGroup("a", (0,)), QueryGroup("b", (1,))))
        records = [
            {"sample_id": "ok", "group_id": "g1", "target": {"status": "OBSERVED", "value": 1},
             "concepts": [{"annotation_status": "OBSERVED", "value": 0}, {"annotation_status": "OBSERVED", "value": 1}]},
            {"sample_id": "missing-c", "group_id": "g2", "target": {"status": "OBSERVED", "value": 0},
             "concepts": [{"annotation_status": "OBSERVED", "value": 1}, {"annotation_status": "MISSING_ANNOTATION", "value": None}]},
            {"sample_id": "missing-y", "group_id": "g3", "target": {"status": "MISSING_ANNOTATION", "value": None},
             "concepts": [{"annotation_status": "OBSERVED", "value": 1}, {"annotation_status": "OBSERVED", "value": 0}]},
        ]
        membership = {r["sample_id"]: {"split": "validation"} for r in records}
        rows, exclusions = gold_rows(records, membership, schema)
        self.assertEqual([row["sample_id"] for row in rows], ["ok"])
        self.assertEqual(exclusions, {"validation:incomplete_concepts": 1,
                                      "validation:missing_target": 1})
        self.assertEqual(rows[0]["response_source"], "gold_concept_diagnostic")


if __name__ == "__main__":
    unittest.main()

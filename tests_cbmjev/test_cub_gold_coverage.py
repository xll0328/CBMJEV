import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_cub_gold_coverage import audit


def row(split, sid, observed):
    return {"split": split, "sample_id": sid, "group_id": sid,
            "target": {"status": "OBSERVED", "value": 0},
            "concepts": [
                {"concept_id": f"c{i}", "annotation_status":
                 "OBSERVED" if good else "NOT_VISIBLE",
                 "value": int(good) if good else None}
                for i, good in enumerate(observed)]}


class CUBGoldCoverageTest(unittest.TestCase):
    def test_group_completeness_and_test_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            schema = {"num_classes": 2,
                      "concepts": [{"id": f"c{i}"} for i in range(3)],
                      "groups": [{"id": "g0", "atoms": [0, 1]},
                                 {"id": "g1", "atoms": [2]}]}
            (root / "schema.json").write_text(json.dumps(schema))
            (root / "audit.json").write_text("{}")
            records = [row("train", "a", [True, True, False]),
                       row("train", "b", [True, False, True]),
                       row("validation", "c", [True, True, True]),
                       {"split": "test", "sample_id": "not-read"}]
            (root / "samples.jsonl").write_text("\n".join(
                json.dumps(record) for record in records) + "\n")
            result = audit(root, ("train", "validation"))
        self.assertEqual(result["splits"]["train"]["cases"], 2)
        self.assertEqual(result["splits"]["train"]["observed_atoms_per_case"]["mean"], 2)
        self.assertEqual(result["splits"]["train"]["group_complete_fraction"],
                         {"g0": .5, "g1": .5})
        self.assertEqual(result["splits"]["train"]["all_groups_complete_fraction"], 0)
        self.assertEqual(result["splits"]["validation"]["all_groups_complete_fraction"], 1)
        self.assertEqual(result["splits"]["validation"]["complete_case_target_classes"], 1)
        self.assertEqual(result["splits"]["validation"]["target_class_total_variation_all_vs_complete"], 0)
        self.assertNotIn("test", result["splits"])


if __name__ == "__main__":
    unittest.main()

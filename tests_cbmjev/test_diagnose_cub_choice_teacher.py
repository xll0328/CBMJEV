from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cbmjev.contracts import Concept, QueryGroup, Schema
from cbmjev.io import write_json, write_jsonl
from scripts.diagnose_cub_choice_teacher import diagnose


class ChoiceTeacherDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.schema = Schema("tiny", 2,
            tuple(Concept(str(i), "feature", ("0", "1")) for i in range(2)),
            tuple(QueryGroup(str(i), (i,)) for i in range(2)))

    def test_bounded_empty_teacher(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            write_json(root / "schema.json", self.schema.to_dict())
            rows = [{"observed": [-1, -1], "action": action, "target": error}
                    for action, error in (([], 1), ([0], 0), ([1], 1))]
            rows += [{"observed": [0, -1], "action": [], "target": 0}]
            write_jsonl(root / "shard.jsonl", rows)
            report = diagnose(root / "schema.json", root / "shard.jsonl",
                              max_histories=2, cost_weight=.03, temperature=.5)
            self.assertEqual(report["histories_scanned"], 2)
            self.assertEqual(report["empty_histories"], 1)
            self.assertEqual(report["stop_argmin_fraction"], 0)
            self.assertEqual(report["teacher_top5"][0]["action"], "0")
            self.assertTrue(report["teacher_top_action_is_lowest_mean_error_action"])

    def test_missing_singleton_rejected(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            write_json(root / "schema.json", self.schema.to_dict())
            write_jsonl(root / "shard.jsonl", [{"observed": [-1, -1],
                "action": [], "target": 1}])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                diagnose(root / "schema.json", root / "shard.jsonl",
                         max_histories=1, cost_weight=.03, temperature=.5)


if __name__ == "__main__":
    unittest.main()

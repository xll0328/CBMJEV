"""Focused role and information-contract checks for the second-source diagnostic."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_cebab_second_source import read_rows, posterior
from scripts.evaluate_cebab_second_source import _read_jsonl, aggregate


def row(sample, group, role):
    return {"sample_id": sample, "group_id": group, "split": role, "y": 1,
            "gold": [0, 1, None, 2], "source_a": [0, 1, 0, 2],
            "source_b": [1, 1, 2, 2]}


class SecondSourceTests(unittest.TestCase):
    def test_role_boundary_and_family_disjointness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in (
                row("x", "family1", "head_fit"), row("y", "family2", "policy_fit"),
                row("z", "family3", "validation"))) + "\n")
            self.assertEqual(len(read_rows(path)), 3)
            contaminated = row("z", "family1", "validation")
            path.write_text("\n".join(json.dumps(r) for r in (
                row("x", "family1", "head_fit"), row("y", "family2", "policy_fit"),
                contaminated)) + "\n")
            with self.assertRaisesRegex(ValueError, "family crosses roles"):
                read_rows(path)
            contaminated["split"] = "test"
            path.write_text("\n".join(json.dumps(r) for r in (
                row("x", "family1", "head_fit"), row("y", "family2", "policy_fit"),
                contaminated)) + "\n")
            with self.assertRaisesRegex(ValueError, "test or unknown role"):
                read_rows(path)

    def test_joint_channel_uses_only_requested_hard_answers(self):
        record = {"prior": [2, 2, 2], "a": [[1, 1, 0]]*3,
                  "b": [[1, 1, 0]]*3, "joint": [[1]*9 for _ in range(3)],
                  "preferred": "a", "labeled": 6}
        channels = [record]*4
        example = row("x", "g", "head_fit")
        before = posterior(example, 0, channels, "joint", 1.0)
        example["gold"] = [2, 2, 2, 2]
        example["y"] = 4
        example["hidden_logits"] = [1000, -1000, 0]
        self.assertEqual(before, posterior(example, 0, channels, "joint", 1.0))

    def test_pilot_metrics_exclude_missing_gold(self):
        metrics = aggregate([row("x", "g", "policy_fit")])["overall"]
        self.assertEqual(metrics["labeled"], 3)
        self.assertEqual(metrics["source_a_correct"], 3)
        self.assertEqual(metrics["source_b_correct"], 2)

    def test_mixed_role_reader_does_not_parse_test_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.jsonl"
            path.write_text('{"split":"test","y":invalid}\n'
                            '{"split":"head_fit","sample_id":"a","y":1}\n')
            self.assertEqual(list(_read_jsonl(path, roles={"head_fit"})),
                             [{"split": "head_fit", "sample_id": "a", "y": 1}])


if __name__ == "__main__":
    unittest.main()

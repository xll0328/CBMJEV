import copy
import unittest

from scripts.compare_choice_device_replays import compare_rows


class ChoiceDeviceReplayTests(unittest.TestCase):
    def setUp(self):
        self.row = {"policy_id": "structured_choice_attention", "sample_id": "bird-1",
            "group_id": "bird-1", "y": 1, "split": "validation", "mode": "offline_replay",
            "steps": [{"action": [2]}, {"action": []}], "queried_groups": [2],
            "prediction": 1, "final_state": [0, -1], "probabilities": [.25, .75]}

    def test_exact_behavior_with_minor_numeric_roundoff(self):
        gpu = copy.deepcopy(self.row)
        gpu["probabilities"] = [.2500001, .7499999]
        result = compare_rows([self.row], [gpu])
        self.assertTrue(result["behavior_equal"])
        self.assertEqual(result["traces"], 1)
        self.assertGreater(result["by_policy"][self.row["policy_id"]]["max_probability_abs_delta"], 0)

    def test_action_change_is_reported_even_if_prediction_agrees(self):
        gpu = copy.deepcopy(self.row)
        gpu["steps"][0]["action"] = [1]
        gpu["queried_groups"] = [1]
        result = compare_rows([self.row], [gpu])
        self.assertFalse(result["behavior_equal"])
        self.assertEqual(result["by_policy"][self.row["policy_id"]]["action_sequence_mismatches"], 1)

    def test_duplicate_or_different_sample_coverage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare_rows([self.row, self.row], [self.row])
        other = copy.deepcopy(self.row)
        other["sample_id"] = "bird-2"
        with self.assertRaisesRegex(ValueError, "coverage"):
            compare_rows([self.row], [other])


if __name__ == "__main__":
    unittest.main()

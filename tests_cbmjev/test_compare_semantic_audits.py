import copy
import unittest

from tools.compare_semantic_audits import compare


def fixture():
    concept = {"concept_id": "a", "per_class_gold_support": [3, 0],
               "response_column_labels": ["semantic:absent", "semantic:present", "runtime:UNCERTAIN", "runtime:NOT_APPLICABLE"],
               "per_class_recall_missing_support_null": [1.0, None],
               "accuracy_runtime_nonanswers_count_as_errors": 1.0,
               "macro_f1_fixed_semantic_classes": 0.5,
               "coverage_semantic_answers_on_observed_gold": 1.0,
               "balanced_accuracy_supported_semantic_classes": 1.0}
    return {"format": "cbmjev-semantic-response-audit-v1", "status": "COMPLETED",
            "split": "validation", "test_evaluated": False, "dataset": "fixture", "schema_hash": "s",
            "source_hashes": {"prepared_samples_sha256": "p", "membership_sha256": "m", "prepared_schema_file_sha256": "s"},
            "selection": {"audited_sample_ids_hash": "ids", "audited_cached_split_cases": 3},
            "concepts": [concept]}


class CompareSemanticAuditsTest(unittest.TestCase):
    def test_null_support_not_zero_or_both_class_mean(self):
        before = fixture()
        after = copy.deepcopy(before)
        after["concepts"][0]["accuracy_runtime_nonanswers_count_as_errors"] = 0.5
        result = compare(before, after)
        self.assertIsNone(result["concepts"][0]["recalls"][1]["delta"])
        self.assertEqual(result["summary"]["observed_support"]["accuracy_runtime_nonanswers_count_as_errors"]["delta"], -0.5)
        self.assertEqual(result["summary"]["all_classes_supported"]["balanced_accuracy_supported_semantic_classes"]["concept_count"], 0)
        self.assertIsNone(result["class_recall_summary_equal_concept_weight"]["semantic:present"]["before"])
        self.assertEqual(result["class_recall_summary_equal_concept_weight"]["semantic:present"]["zero_recall_before"], 0)

    def test_population_mismatch_rejected(self):
        for section, key in (("selection", "audited_sample_ids_hash"), ("source_hashes", "membership_sha256")):
            before, after = fixture(), fixture()
            after[section][key] = "changed"
            with self.assertRaises(ValueError):
                compare(before, after)

    def test_test_audit_and_duplicate_ids_rejected(self):
        for change in (lambda r: r.update(split="test", test_evaluated=True),
                       lambda r: r["concepts"].append(copy.deepcopy(r["concepts"][0]))):
            before, after = fixture(), fixture()
            change(after)
            with self.assertRaises(ValueError):
                compare(before, after)

    def test_support_and_null_mismatch_rejected(self):
        for change in (lambda r: r["concepts"][0]["per_class_gold_support"].__setitem__(1, 2),
                       lambda r: r["concepts"][0]["per_class_recall_missing_support_null"].__setitem__(1, 0)):
            before, after = fixture(), fixture()
            change(after)
            with self.assertRaises(ValueError):
                compare(before, after)


if __name__ == "__main__":
    unittest.main()

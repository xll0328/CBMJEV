"""Public-label/cache audit fixtures; no real dataset/model quality evidence."""

import tempfile
import unittest
from pathlib import Path

from cbmjev.audit import audit_responses
from cbmjev.contracts import Concept, QueryGroup, Schema
from cbmjev.io import file_hash, read_json, write_json, write_jsonl


def fixture(root, *, all_missing=False, all_correct=False, empty_test=False,
            schema_mismatch=False, target_mismatch=False, bad_gold=False,
            bad_data_hash=False, response_source="automatic_model"):
    prepared, cache = root / "prepared", root / "cache"
    prepared.mkdir()
    cache.mkdir()
    schema = Schema(
        "cebab", 2,
        (
            Concept("food", "food sentiment", ("negative", "positive", "unknown")),
            Concept("noise", "noise sentiment", ("negative", "positive", "unknown")),
            Concept("attribute", "binary attribute", ("absent", "present")),
        ),
        (QueryGroup("food", (0,)), QueryGroup("combined", (1, 2))),
    )
    rows, membership, responses = [], [], []

    def add(sample, split, labels, z, *, task_missing=False, include_cache=True):
        concepts = [
            {"concept_id": concept.id, "value": value,
             "annotation_status": "OBSERVED" if value is not None else "MISSING_ANNOTATION"}
            for concept, value in zip(schema.concepts, labels)
        ]
        row = {
            "sample_id": sample, "group_id": "family:" + sample, "dataset": "cebab",
            "input": {"text": "synthetic content"},
            "target": {"status": "MISSING_ANNOTATION" if task_missing else "OBSERVED",
                       "value": None if task_missing else 0},
            "concepts": concepts,
        }
        rows.append(row)
        membership.append({"sample_id": sample, "group_id": row["group_id"], "split": split})
        if include_cache:
            responses.append({
                "sample_id": sample, "group_id": row["group_id"], "split": split,
                "z": z, "y": 0, "response_source": response_source,
            })

    add("h", "head_fit", [0, 0, 0], [0, 0, 0])
    add("p", "policy_fit", [0, 0, 0], [0, 0, 0])
    labels = [[2, 0, 0], [1, 1, 1], [0, 0, 0], [1, None, 1]]
    predicted = [[2, 0, 0], [3, 1, 0], [4, 0, 0], [1, 0, 1]]
    for index, (gold, z) in enumerate(zip(labels, predicted)):
        if all_correct:
            z = [value if value is not None else 0 for value in gold]
        add("v{}".format(index), "validation", [None] * 3 if all_missing else gold, z)
    add("missing_y", "validation", [2, 2, 1], [2, 2, 1], task_missing=True, include_cache=False)
    add("t", "test", [None, None, None], [3, 4, 2],
        task_missing=empty_test, include_cache=not empty_test)
    if target_mismatch:
        responses[2]["y"] = 1
    if bad_gold:
        rows[2]["concepts"][0]["value"] = len(schema.concepts[0].values)
    write_json(prepared / "schema.json", schema.to_dict())
    write_jsonl(prepared / "samples.jsonl", rows)
    write_jsonl(prepared / "membership.jsonl", membership)
    cache_schema = schema.to_dict()
    if schema_mismatch:
        cache_schema["dataset"] = "different-dataset"
    write_json(cache / "schema.json", cache_schema)
    write_jsonl(cache / "responses.jsonl", responses)
    manifest = {
        "format": "cbmjev-cache-v1",
        "schema_hash": Schema.from_dict(cache_schema).hash,
        "responses_sha256": file_hash(cache / "responses.jsonl"),
        "response_source": response_source,
        "batch_independent": True,
        "responder_checkpoint_sha256": "a" * 64,
        "prepared_samples_sha256": (
            "b" * 64 if bad_data_hash else file_hash(prepared / "samples.jsonl")
        ),
        "membership_sha256": file_hash(prepared / "membership.jsonl"),
    }
    write_json(cache / "manifest.json", manifest)
    return prepared, cache


class SemanticAuditTests(unittest.TestCase):
    def test_imbalance_metrics_count_nonanswers_as_false_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, cache = fixture(root)
            result = audit_responses(prepared, cache, root / "audit")
            food, noise, _ = result["concepts"]
            self.assertEqual(food["per_class_recall_missing_support_null"], [0, .5, 1])
            self.assertEqual(food["balanced_accuracy_supported_semantic_classes"], .5)
            self.assertEqual(food["descriptive_majority_fraction_on_audited_gold"], .5)
            self.assertEqual(noise["per_class_recall_missing_support_null"], [1, 1, None])
            self.assertEqual(noise["num_supported_semantic_classes"], 2)
            self.assertEqual(noise["balanced_accuracy_supported_semantic_classes"], 1)

    def test_imbalance_metrics_without_gold_remain_null(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, cache = fixture(root, all_missing=True)
            result = audit_responses(prepared, cache, root / "audit")
            for concept in result["concepts"]:
                self.assertIsNone(concept["balanced_accuracy_supported_semantic_classes"])
                self.assertIsNone(concept["descriptive_majority_fraction_on_audited_gold"])
                self.assertEqual(concept["num_supported_semantic_classes"], 0)

    def test_semantic_unknown_is_answer_runtime_unknown_is_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, cache = fixture(root)
            result = audit_responses(prepared, cache, root / "audit")
            food = result["concepts"][0]
            self.assertEqual(food["semantic_values"][2], "unknown")
            self.assertEqual(food["confusion_gold_rows_response_columns"][2][2], 1)
            self.assertEqual(food["cached_observed_gold_count"], 4)
            self.assertEqual(food["correct_semantic_count"], 2)
            self.assertEqual(food["runtime_uncertain_count_on_observed_gold"], 1)
            self.assertEqual(food["runtime_not_applicable_count_on_observed_gold"], 1)
            self.assertEqual(food["accuracy_runtime_nonanswers_count_as_errors"], 0.5)
            self.assertEqual(food["coverage_semantic_answers_on_observed_gold"], 0.5)
            self.assertEqual(food["conditional_accuracy_among_semantic_answers"], 1)
            self.assertAlmostEqual(food["macro_f1_fixed_semantic_classes"], 5 / 9)
            self.assertEqual(result["overall"]["macro_concept_accuracy"], 0.75)
            self.assertEqual(read_json(root / "audit" / "audit.json"), result)

    def test_missing_task_target_selection_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, cache = fixture(root)
            result = audit_responses(prepared, cache, root / "audit")
            selection = result["selection"]
            self.assertEqual(selection["canonical_total_cases_all_roles"], 8)
            self.assertEqual(selection["cache_total_cases_all_roles"], 7)
            self.assertEqual(selection["canonical_split_cases"], 5)
            self.assertEqual(selection["audited_cached_split_cases"], 4)
            self.assertEqual(selection["canonical_split_missing_task_target_cases"], 1)
            self.assertEqual(selection["canonical_split_not_cached"], 1)
            self.assertFalse(selection["is_whole_dataset_concept_rate"])
            self.assertEqual(result["concepts"][0]["canonical_split_observed_gold_count"], 5)
            self.assertEqual(result["concepts"][0]["cached_observed_gold_count"], 4)

    def test_missing_concept_gold_not_scored_and_pair_n_is_complete_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, cache = fixture(root)
            result = audit_responses(prepared, cache, root / "audit")
            noise = result["concepts"][1]
            self.assertEqual(noise["cached_observed_gold_count"], 3)
            self.assertEqual(noise["cached_missing_gold_count"], 1)
            # Three fixed semantic classes, even though unknown has no support.
            self.assertAlmostEqual(noise["macro_f1_fixed_semantic_classes"], 2 / 3)
            matrix = result["group_error_coupling"]["matrix"]
            pair = matrix[0][1]
            self.assertEqual(pair["paired_n"], 3)
            self.assertEqual(pair["n_both_correct"], 1)
            self.assertEqual(pair["n_both_error"], 1)
            self.assertEqual(pair["n_first_error_second_correct"], 1)
            self.assertAlmostEqual(pair["phi"], 0.5)
            self.assertEqual(matrix[1][0]["phi"], pair["phi"])
            self.assertEqual(result["query_groups"][1]["cached_incomplete_gold_cases_not_scored"], 1)
            self.assertFalse(result["group_error_coupling"]["is_causal_relationship"])

    def test_zero_variance_phi_is_null_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, cache = fixture(root, all_correct=True)
            result = audit_responses(prepared, cache, root / "audit")
            pair = result["group_error_coupling"]["matrix"][0][1]
            self.assertEqual(pair["paired_n"], 3)
            self.assertIsNone(pair["phi"])
            self.assertEqual(pair["phi_undefined_reason"], "ZERO_ERROR_INDICATOR_VARIANCE")

    def test_all_missing_gold_has_null_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, cache = fixture(root, all_missing=True)
            result = audit_responses(prepared, cache, root / "audit")
            self.assertEqual(result["status"], "NO_OBSERVED_CONCEPT_GOLD")
            self.assertIsNone(result["overall"]["macro_concept_accuracy"])
            self.assertIsNone(result["overall"]["macro_concept_f1"])
            self.assertIsNone(result["overall"]["micro_concept_coverage"])
            self.assertEqual(result["overall"]["num_scored_case_concept_pairs"], 0)
            self.assertIsNone(result["group_error_coupling"]["matrix"][0][1]["phi"])
            self.assertEqual(result["group_error_coupling"]["matrix"][0][1]["paired_n"], 0)

    def test_test_guard_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, cache = fixture(root)
            with self.assertRaisesRegex(ValueError, "evaluate_test"):
                audit_responses(prepared, cache, root / "audit", split="test")
            self.assertFalse((root / "audit").exists())
            result = audit_responses(prepared, cache, root / "audit", split="test", evaluate_test=True)
            self.assertTrue(result["test_evaluated"])
            self.assertEqual(result["status"], "NO_OBSERVED_CONCEPT_GOLD")

    def test_empty_cached_requested_split_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, cache = fixture(root, empty_test=True)
            result = audit_responses(prepared, cache, root / "audit", split="test", evaluate_test=True)
            self.assertEqual(result["status"], "NO_CACHED_CASES")
            self.assertEqual(result["selection"]["canonical_split_cases"], 1)
            self.assertEqual(result["selection"]["audited_cached_split_cases"], 0)
            self.assertIsNone(result["overall"]["macro_concept_coverage"])

    def test_schema_data_hash_target_and_gold_guards(self):
        for option in ("schema_mismatch", "bad_data_hash", "target_mismatch", "bad_gold"):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                prepared, cache = fixture(root, **{option: True})
                with self.assertRaises(ValueError):
                    audit_responses(prepared, cache, root / "audit")
                self.assertFalse((root / "audit").exists())

    def test_no_overwrite_and_inputs_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, cache = fixture(root)
            before = (file_hash(prepared / "samples.jsonl"), file_hash(cache / "responses.jsonl"))
            audit_responses(prepared, cache, root / "audit")
            with self.assertRaises(ValueError):
                audit_responses(prepared, cache, root / "audit")
            after = (file_hash(prepared / "samples.jsonl"), file_hash(cache / "responses.jsonl"))
            self.assertEqual(before, after)

    def test_synthetic_source_never_claims_real_dataset_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, cache = fixture(root, response_source="synthetic")
            result = audit_responses(prepared, cache, root / "audit")
            self.assertEqual(result["evidence_status"], "SYNTHETIC_FIXTURE_NOT_RESEARCH_EVIDENCE")

    def test_other_role_not_silently_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, cache = fixture(root)
            with self.assertRaises(ValueError):
                audit_responses(prepared, cache, root / "audit", split="head_fit")


if __name__ == "__main__":
    unittest.main()

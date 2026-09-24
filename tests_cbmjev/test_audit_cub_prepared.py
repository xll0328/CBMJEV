"""Independent audit tests use only tiny synthetic official-format fixtures."""
import json
from pathlib import Path
import tempfile
import unittest

from cbmjev.data import prepare_dataset
from tests_cbmjev.test_data import cub_fixture
from tools.audit_cub_prepared import audit_cub


class IndependentCUBAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.prepared = self.root / "source", self.root / "prepared"
        cub_fixture(self.source)
        annotations = self.source / "attributes/image_attribute_labels.txt"
        # The general adapter allows missing labels; this official audit requires
        # a complete Cartesian join, so complete the intentionally missing cell.
        with annotations.open("a") as stream:
            stream.write("1 3 1 6 0.5\n")
        prepare_dataset("cub", self.source, self.prepared, seed=17)

    def run_audit(self):
        return audit_cub(self.source, self.prepared, self.root / "audit.json", expected_images=24,
                         expected_attributes=3, expected_classes=2, expected_groups=2,
                         expected_train=22, expected_test=2, expected_anomalies=0)

    def mutate(self, filename, operation):
        path = self.prepared / filename
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        operation(records)
        path.write_text("".join(json.dumps(row) + "\n" for row in records))

    def assert_failure_without_output(self):
        with self.assertRaises(ValueError):
            self.run_audit()
        self.assertFalse((self.root / "audit.json").exists())

    def test_success_has_counts_and_hashes_not_sample_data(self):
        result = self.run_audit()
        self.assertEqual(result["prepared_samples"], 24)
        self.assertEqual(result["raw_annotation_pairs"], 72)
        self.assertEqual(result["prepared_null_duration_count"], 0)
        self.assertEqual(sum(result["membership_role_counts"].values()), 24)
        self.assertEqual(result["status"], "PREPARED_MECHANICALLY_VERIFIED_NOT_PAPER_ACCEPTANCE")
        serialized = json.dumps(result)
        for forbidden in ("cub:1", "color::red", "pic1.ppm", "certainty_annotations"):
            self.assertNotIn(forbidden, serialized)
        with self.assertRaises(FileExistsError):
            self.run_audit()

    def test_changed_class_rejected(self):
        self.mutate("samples.jsonl", lambda rows: rows[0]["target"].update(value=17))
        self.assert_failure_without_output()

    def test_changed_certainty_rejected(self):
        self.mutate("samples.jsonl", lambda rows: rows[0]["audit_metadata"]["certainty_annotations"]["1"].__setitem__(1, 7))
        self.assert_failure_without_output()

    def test_illegal_null_concept_rejected(self):
        self.mutate("samples.jsonl", lambda rows: rows[0]["concepts"][0].update(value=None))
        self.assert_failure_without_output()

    def test_changed_source_split_rejected(self):
        self.mutate("samples.jsonl", lambda rows: rows[0]["provenance"].update(source_split="test"))
        self.assert_failure_without_output()

    def test_missing_raw_annotation_rejected(self):
        path = self.source / "attributes/image_attribute_labels.txt"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
        self.assert_failure_without_output()

    def test_missing_prepared_without_exclusion_rejected(self):
        self.mutate("samples.jsonl", lambda rows: rows.pop())
        self.assert_failure_without_output()

    def test_missing_membership_rejected(self):
        self.mutate("membership.jsonl", lambda rows: rows.pop())
        self.assert_failure_without_output()

    def test_fabricated_anomaly_sidecar_rejected(self):
        (self.prepared / "annotation_anomalies.jsonl").write_text(
            json.dumps({"image_id": 1, "attribute_id": 1}) + "\n")
        self.assert_failure_without_output()

    def test_fabricated_duration_null_rejected(self):
        self.mutate("samples.jsonl", lambda rows: rows[0]["audit_metadata"]["certainty_annotations"]["1"].__setitem__(2, None))
        self.assert_failure_without_output()

    def test_boolean_target_not_accepted_as_integer(self):
        self.mutate("samples.jsonl", lambda rows: rows[0]["target"].update(value=False))
        self.assert_failure_without_output()

    def test_role_crossing_rejected_even_if_sample_and_membership_agree(self):
        path = self.prepared / "membership.jsonl"
        membership = [json.loads(line) for line in path.read_text().splitlines()]
        a = next(row for row in membership if row["split"] == "head_fit")
        b = next(row for row in membership if row["split"] == "policy_fit")
        self.mutate("samples.jsonl", lambda rows: [r.update(group_id=a["group_id"]) for r in rows if r["sample_id"] == b["sample_id"]])
        self.mutate("membership.jsonl", lambda rows: [r.update(group_id=a["group_id"]) for r in rows if r["sample_id"] == b["sample_id"]])
        self.assert_failure_without_output()

    def test_valid_duplicate_exclusion_covers_source_ids(self):
        import shutil
        # Rebuild in a fresh output; exact pixel duplicate crosses train/test.
        shutil.copyfile(self.source / "images/pic12.ppm", self.source / "images/pic1.ppm")
        self.prepared = self.root / "prepared_with_exclusion"
        prepare_dataset("cub", self.source, self.prepared, seed=17)
        result = self.run_audit()
        self.assertEqual(result["prepared_samples"], 23)
        self.assertEqual(result["excluded_samples"], 1)


if __name__ == "__main__":
    unittest.main()

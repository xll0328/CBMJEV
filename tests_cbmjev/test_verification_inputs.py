"""Synthetic role/label joins; no real dataset is read by this test module."""
import tempfile
import unittest
from pathlib import Path

from cbmjev.contracts import Concept, QueryGroup, Schema
from cbmjev.io import file_hash, write_json, write_jsonl
from cbmjev.verification_inputs import load_verification_inputs


class VerificationInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.schema = Schema("synthetic", 2,
            (Concept("aspect", "An explicit test aspect", ("negative", "positive", "unknown")),),
            (QueryGroup("aspect", (0,)),))
        self.members, self.samples, self.a, self.b = [], [], [], []
        for i, role in enumerate(("head_fit", "head_fit", "policy_fit", "test", "responder_fit")):
            sid, gid = str(i), "group" + str(i)
            self.members.append({"sample_id": sid, "group_id": gid, "split": role})
            self.samples.append({"sample_id": sid, "group_id": gid, "dataset": "synthetic",
                "split": "train" if role != "test" else "test",
                "target": {"status": "MISSING_ANNOTATION" if i == 1 else "OBSERVED", "value": None if i == 1 else 0},
                "concepts": [{"concept_id": "aspect", "value": None if i == 0 else 2,
                              "annotation_status": "MISSING_ANNOTATION" if i == 0 else "OBSERVED"}]})
            self.a.append({"sample_id": sid, "group_id": gid, "split": role, "A": [2], "z": [2], "y": 1})
            self.b.append({"sample_id": sid, "group_id": gid, "split": role,
                           "typed_values": [0], "shared_values": [1]})

    def save(self):
        write_json(self.root / "schema.json", self.schema.to_dict())
        write_jsonl(self.root / "membership.jsonl", self.members)
        write_jsonl(self.root / "samples.jsonl", self.samples)
        write_jsonl(self.root / "a.jsonl", self.a)
        write_jsonl(self.root / "b.jsonl", self.b)

    def load(self, **kw):
        return load_verification_inputs(self.root, self.root / "a.jsonl", self.root / "b.jsonl",
                                        roles=kw.pop("roles", ("head_fit", "policy_fit")), **kw)

    def test_missing_is_not_unknown_and_cache_y_ignored(self):
        self.save()
        result = self.load()
        self.assertEqual(result.rows["head_fit"][0]["C"], (None,))
        self.assertEqual(result.rows["policy_fit"][0]["C"], (2,))
        self.assertEqual(result.rows["head_fit"][0]["Y"], 0)
        self.assertEqual(result.report["roles"]["head_fit"]["excluded_missing_task"], 1)

    def test_protected_and_source_roles_rejected(self):
        self.save()
        with self.assertRaisesRegex(ValueError, "protected"):
            self.load(roles=("test",))
        with self.assertRaisesRegex(ValueError, "source supervision"):
            self.load(roles=("responder_fit",))

    def test_group_cross_role_rejected(self):
        self.members[2]["group_id"] = "group0"
        self.save()
        with self.assertRaisesRegex(ValueError, "group crosses"):
            self.load()

    def test_missing_prediction_is_not_silently_dropped(self):
        self.b = self.b[1:]
        self.save()
        with self.assertRaisesRegex(ValueError, "missing source prediction"):
            self.load()

    def test_prediction_role_mismatch_rejected(self):
        self.b[0]["split"] = "policy_fit"
        self.save()
        with self.assertRaisesRegex(ValueError, "join mismatch"):
            self.load()

    def test_prediction_duplicate_rejected(self):
        self.a.append(dict(self.a[0]))
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicate prediction"):
            self.load()

    def test_legacy_cache_manifest_and_hash_required(self):
        self.save()
        with self.assertRaisesRegex(ValueError, "original cache manifest"):
            self.load(legacy_a=True)
        manifest = {"format": "cbmjev-cache-v1", "schema_hash": self.schema.hash,
                    "membership_sha256": file_hash(self.root / "membership.jsonl"),
                    "responses_sha256": file_hash(self.root / "a.jsonl"),
                    "provenance": [{"supervised_group_ids": ["group4"]}]}
        write_json(self.root / "manifest.json", manifest)
        loaded = self.load(legacy_a=True, a_metadata=self.root / "manifest.json")
        self.assertEqual(loaded.rows["head_fit"][0]["A"], (2,))

    def test_prediction_metadata_hash_mismatch_rejected(self):
        self.save()
        write_json(self.root / "b.metadata.json", {"predictions_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            self.load(b_metadata=self.root / "b.metadata.json")

    def test_native_runtime_status_is_not_accepted_as_semantic_value(self):
        self.a[0]["A"] = [3]
        self.save()
        with self.assertRaisesRegex(ValueError, "native semantic"):
            self.load()


if __name__ == "__main__":
    unittest.main()

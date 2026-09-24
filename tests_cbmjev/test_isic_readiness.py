import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from tools.prepare_isic_readiness import ATTRIBUTES, assign, prepare


class ISICReadinessTests(unittest.TestCase):
    def fixture(self, root, count=24):
        source = root / "raw"
        mask_root = source / "groundtruth_extracted_v1/ISIC2018_Task2_Training_GroundTruth_v3"
        mask_root.mkdir(parents=True)
        (source / "images").mkdir()
        records = {}
        for i in range(count):
            sid = "ISIC_{:07d}".format(i)
            Image.new("RGB", (3, 2), (i * 8 % 256, 40 + i // 32, 90)).save(source / "images" / (sid + ".jpg"))
            for j, attr in enumerate(ATTRIBUTES):
                Image.new("L", (3, 2), 255 if i % 5 == j else 0).save(mask_root / (sid + "_attribute_" + attr + ".png"))
            records[sid] = {"isic_id": sid, "copyright_license": "CC-0", "metadata": {
                "clinical": {"diagnosis_1": "Benign", "patient_id": "p0" if i < 2 else None,
                             "lesion_id": "l1" if i in (1, 2) else None}}}
        metadata = root / "snapshot.json"
        metadata.write_text(json.dumps({"records": records}))
        return source, metadata

    def test_determinism_identity_union_and_loader_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, metadata = self.fixture(root)
            a = prepare(source, metadata, root / "a", expected_ids=24)
            b = prepare(source, metadata, root / "b", expected_ids=24)
            self.assertEqual(a["split_hash"], b["split_hash"])
            self.assertTrue(all(a["observed_counts"].values()))
            rows = [json.loads(s) for s in (root / "a/membership.jsonl").read_text().splitlines()]
            self.assertEqual(len({r["group_id"] for r in rows[:3]}), 1)
            self.assertEqual(len({r["split"] for r in rows[:3]}), 1)
            self.assertEqual(a["checks"]["patient_independence_missing_ids"], "NOT_CHECKED")
            samples = [json.loads(s) for s in (root / "a/samples.jsonl").read_text().splitlines()]
            self.assertTrue(all(s["provenance"]["source_split"] == "official_task2_training" for s in samples))
            self.assertNotIn("patient_id", str(samples[0]["input"]))
            with self.assertRaises(FileExistsError):
                prepare(source, metadata, root / "a", expected_ids=24)

    def test_missing_mask_is_error_not_absence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, metadata = self.fixture(root)
            next(source.rglob("*_attribute_globules.png")).unlink()
            with self.assertRaisesRegex(ValueError, "incomplete mask"):
                prepare(source, metadata, root / "out", expected_ids=24)

    def test_unknown_target_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, metadata = self.fixture(root)
            data = json.loads(metadata.read_text())
            data["records"]["ISIC_0000000"]["metadata"]["clinical"]["diagnosis_1"] = "unrecognized"
            metadata.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "unknown diagnosis"):
                prepare(source, metadata, root / "out", expected_ids=24)

    def test_missing_image_explicit_exclusion_and_exact_pixels_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, metadata = self.fixture(root)
            (source / "images/ISIC_0000023.jpg").unlink()
            (source / "images/ISIC_0000009.jpg").write_bytes((source / "images/ISIC_0000008.jpg").read_bytes())
            audit = prepare(source, metadata, root / "out", expected_ids=24)
            self.assertEqual(audit["exclusions_by_reason"], {"missing_image": 1})
            self.assertIn(["ISIC_0000008", "ISIC_0000009"], audit["exact_pixel_duplicate_pairs"])

    def test_verified_decode_reuse_requires_unchanged_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, metadata = self.fixture(root)
            original = prepare(source, metadata, root / "first", expected_ids=24)
            reused = prepare(source, metadata, root / "second", expected_ids=24, reuse_prepared=root / "first")
            self.assertEqual(original["split_hash"], reused["split_hash"])
            self.assertEqual(reused["verified_decode_reuse"]["reused_ids"], 24)
            Image.new("RGB", (3, 2), (255, 255, 255)).save(source / "images/ISIC_0000000.jpg")
            with self.assertRaisesRegex(ValueError, "raw files changed"):
                prepare(source, metadata, root / "third", expected_ids=24, reuse_prepared=root / "first")

    def test_missing_image_still_provides_identity_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, metadata = self.fixture(root)
            (source / "images/ISIC_0000001.jpg").unlink()
            prepare(source, metadata, root / "out", expected_ids=24)
            rows = [json.loads(s) for s in (root / "out/membership.jsonl").read_text().splitlines()]
            connected = [r for r in rows if r["sample_id"] in ("isic2018_task2:ISIC_0000000", "isic2018_task2:ISIC_0000002")]
            self.assertEqual(len({r["group_id"] for r in connected}), 1)

    def test_binary_exclusion_mapping_all_roles_and_identity_pixel_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, metadata = self.fixture(root, count=42)
            data = json.loads(metadata.read_text())
            for i, record in enumerate(data["records"].values()):
                record["metadata"]["clinical"]["diagnosis_1"] = "Benign" if i < 21 else "Malignant"
            data["records"]["ISIC_0000001"]["metadata"]["clinical"]["diagnosis_1"] = "Indeterminate"
            metadata.write_text(json.dumps(data))
            # Excluded case1 bridges cases0/2 by identity and case8 by pixels.
            (source / "images/ISIC_0000001.jpg").write_bytes((source / "images/ISIC_0000008.jpg").read_bytes())
            original = prepare(source, metadata, root / "three", expected_ids=42)
            audit = prepare(source, metadata, root / "binary", expected_ids=42,
                            target_mode="binary", require_complete=True, reuse_prepared=root / "three")
            self.assertEqual(audit["target_classes"], ["Benign", "Malignant"])
            self.assertEqual(audit["source_audited_images"], 42)
            self.assertEqual(sum(audit["observed_counts"].values()), 41)
            self.assertEqual(audit["exclusions_by_reason"], {"indeterminate_excluded_from_binary_task": 1})
            self.assertEqual(json.loads((root / "binary/schema.json").read_text())["num_classes"], 2)
            self.assertTrue(all(set(v) == {"Benign", "Malignant"} for v in audit["role_target_counts"].values()))
            samples = [json.loads(s) for s in (root / "binary/samples.jsonl").read_text().splitlines()]
            self.assertTrue(all(s["target"]["value"] == int(s["audit_metadata"]["target_class"] == "Malignant") for s in samples))
            self.assertNotIn("ISIC_0000001", {s["provenance"]["source_id"] for s in samples})
            connected = [s for s in samples if s["provenance"]["source_id"] in ("ISIC_0000000", "ISIC_0000002", "ISIC_0000008")]
            self.assertEqual(len({s["group_id"] for s in connected}), 1)
            fresh = prepare(source, metadata, root / "fresh", expected_ids=42, target_mode="binary", require_complete=True)
            self.assertEqual(audit["split_hash"], fresh["split_hash"])
            self.assertNotEqual(audit["schema_hash"], original["schema_hash"])

    def test_complete_binary_rejects_missing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, metadata = self.fixture(root)
            (source / "images/ISIC_0000023.jpg").unlink()
            with self.assertRaisesRegex(ValueError, "complete preparation required"):
                prepare(source, metadata, root / "out", expected_ids=24,
                        target_mode="binary", require_complete=True)
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()

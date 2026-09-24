"""Tiny official-format fixtures only; these are not real-data experiments."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cbmjev.data import DERM_VALUES, prepare_dataset
from cbmjev.contracts import Schema


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def review(i, family=None, original=True, text=None):
    return {"id": str(i), "original_id": str(i) if family is None else family,
            "is_original": original, "edit_id": "0" if original else "1",
            "description": text or "A distinct review number {}.".format(i),
            "review_majority": "3", "food_aspect_majority": "Positive",
            "food_aspect_label_distribution": "{'Positive': 4, 'Negative': 1}",
            "noise_aspect_majority": "unknown", "ambiance_aspect_majority": "",
            "service_aspect_majority": "no majority"}


def cebab_fixture(root, layout="records"):
    tables = {"train_exclusive": [review(i) for i in range(10)],
              "dev": [review(i) for i in range(20, 24)], "test": [review(30), review(31)]}
    for name, rows in tables.items():
        if layout == "jsonl":
            write(root / (name + ".jsonl"), "\n".join(json.dumps(row) for row in rows))
        else:
            if layout == "columns":
                data = {key: [row[key] for row in rows] for key in rows[0]}
            elif layout == "indexed":
                data = {key: {str(i): row[key] for i, row in enumerate(rows)} for key in rows[0]}
            else:
                data = rows
            write(root / (name + ".json"), json.dumps(data))
    return tables


def image(root, name, number):
    # A one-pixel binary PPM: readable by optional Pillow, distinct exact pixels.
    path = root / "images" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"P6\n1 1\n255\n" + bytes([number % 256, number // 256, 19]))


def cub_fixture(root, n=24):
    write(root / "classes.txt", "2 bird_two\n1 bird_one\n")
    write(root / "images.txt", "".join("{} pic{}.ppm\n".format(i, i) for i in reversed(range(1, n + 1))))
    write(root / "image_class_labels.txt", "".join("{} {}\n".format(i, 1 if i <= 12 else 2) for i in range(1, n + 1)))
    write(root / "train_test_split.txt", "".join("{} {}\n".format(i, 0 if i in (12, 24) else 1) for i in range(1, n + 1)))
    write(root / "attributes/attributes.txt", "3 shape::round\n1 color::red\n2 color::blue\n")
    write(root / "attributes/certainties.txt", "9 definitely\n7 not visible\n6 guessing\n8 probably\n")
    annotations = []
    for i in range(1, n + 1):
        image(root, "pic{}.ppm".format(i), i)
        annotations.append("{} 1 1 9 0.2\n".format(i))
        annotations.append("{} 2 0 7 1.3\n".format(i))
        if i != 1:
            annotations.append("{} 3 1 6 0.5\n".format(i))
    write(root / "attributes/image_attribute_labels.txt", "".join(reversed(annotations)))


def cub_suffix_anomaly_fixture(root):
    """Small synthetic image set, but the entire verified 606-key exception shape."""
    cub_fixture(root)
    write(root / "attributes/attributes.txt",
          "".join("{} group::value{}\n".format(i, i) for i in range(1, 313)))
    for name, suffix in (("images.txt", "2275 extra2275.ppm\n9364 extra9364.ppm\n"),
                         ("image_class_labels.txt", "2275 1\n9364 2\n"),
                         ("train_test_split.txt", "2275 1\n9364 1\n")):
        path = root / name
        write(path, path.read_text() + suffix)
    path = root / "attributes/image_attribute_labels.txt"
    lines = path.read_text()
    for image_id in (2275, 9364):
        image(root, "extra{}.ppm".format(image_id), image_id)
        for attr in range(10, 313):
            lines += "{} {} 1 9 0 1.509\n".format(image_id, attr)
    write(path, lines)
    return path


def derm_fixture(root):
    rows = []
    for i in range(12):
        row = {key: "absent" for key in DERM_VALUES}
        row.update(case_num=str(500 - i), diagnosis="clark nevus", derm="d{}.ppm".format(i),
                   clinic="c{}.ppm".format(i), vascular_structures="linear irregular")
        rows.append(row)
        image(root, row["derm"], i + 1)
        image(root, row["clinic"], i + 101)
    root.joinpath("meta").mkdir(parents=True)
    with (root / "meta/meta.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for split, indexes in (("train", range(8)), ("valid", range(8, 10)), ("test", range(10, 12))):
        write(root / "meta" / (split + "_indexes.csv"), "indexes\n" + "\n".join(map(str, indexes)) + "\n")
    return rows


def skincon_fixture(root):
    concepts = ["Papule", "Plaque", "Brown(Hyperpigmentation)"]
    ann_rows, fitz_rows = [], []
    labels = ["benign", "malignant", "non-neoplastic"]
    for i in range(30):
        md5 = "{:032x}".format(i + 1)
        image_id = md5 + ".jpg"
        row = {"": str(i), "ImageID": image_id, "Do not consider this image": "0"}
        row.update({concept: str((i + j) % 2) for j, concept in enumerate(concepts)})
        ann_rows.append(row)
        fitz_rows.append({"md5hash": md5, "fitzpatrick_scale": str(i % 6 + 1),
                          "fitzpatrick_centaur": str(i % 6 + 1), "label": "label_{}".format(i % 5),
                          "nine_partition_label": labels[i % 3],
                          "three_partition_label": labels[i % 3], "qc": "",
                          "url": "https://example.test/{}".format(image_id),
                          "url_alphanum": image_id})
        image(root, image_id, i + 1)
    ann_rows[0]["Do not consider this image"] = "1"
    with (root / "annotations_fitzpatrick17k.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["", "ImageID"] + concepts
                                + ["Do not consider this image"])
        writer.writeheader()
        writer.writerows(ann_rows)
    with (root / "fitzpatrick17k.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fitz_rows[0]))
        writer.writeheader()
        writer.writerows(fitz_rows)
    return ann_rows, fitz_rows


def isic_fixture(root):
    from PIL import Image
    mask_root = root / "groundtruth_extracted_v1" / "ISIC2018_Task2_Training_GroundTruth_v3"
    mask_root.mkdir(parents=True)
    attrs = ["globules", "milia_like_cyst", "negative_network", "pigment_network", "streaks"]
    records = {}
    for i in range(140):
        image_id = "ISIC_{:07d}".format(i)
        diagnosis = "Malignant" if i % 5 == 0 else "Benign"
        records[image_id] = {"isic_id": image_id, "copyright_license": "CC-0",
                             "attribution": "fixture",
                             "metadata": {"clinical": {"diagnosis_1": diagnosis}}}
        if i != 0:
            image(root, image_id + ".jpg", i + 1)
        for j, attr in enumerate(attrs):
            im = Image.new("L", (2, 2), 0)
            if (i + j) % 3 == 0:
                im.putpixel((0, 0), 255)
            im.save(mask_root / "{}_attribute_{}.png".format(image_id, attr))
    write(root / "isic_archive_metadata_task2.json", json.dumps({"records": records}))


class DataAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "raw"
        self.source.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, dataset, **kwargs):
        return prepare_dataset(dataset, self.source, self.root / "prepared", **kwargs)

    def test_cebab_all_local_json_layouts_and_missingness(self):
        for layout in ("records", "columns", "indexed", "jsonl"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                raw = root / "raw"
                tables = cebab_fixture(raw, layout)
                result = prepare_dataset("cebab", raw, root / "out")
                samples = read_jsonl(result["samples"])
                schema = Schema.from_dict(json.loads(result["schema"].read_text()))
                self.assertEqual(schema.num_atoms, 4)
                self.assertEqual(schema.num_groups, 4)
                self.assertEqual(len(samples), sum(map(len, tables.values())))
                first = next(s for s in samples if s["sample_id"] == "cebab:0")
                self.assertEqual(first["group_id"], "cebab:family:0")
                self.assertEqual([c["value"] for c in first["concepts"]], [1, 2, None, None])
                self.assertEqual(first["concepts"][2]["annotation_status"], "MISSING_ANNOTATION")
                self.assertEqual(first["concepts"][3]["annotation_status"], "NO_MAJORITY")
                self.assertEqual(first["concepts"][0]["distribution"], [0.2, 0.8, 0.0])
                self.assertEqual(result["report"]["role_counts"]["responder_fit"], 6)
                self.assertEqual(result["report"]["role_counts"]["head_fit"], 2)
                self.assertEqual(result["report"]["role_counts"]["policy_fit"], 2)
                self.assertEqual(result["report"]["observed_counts"],
                                 {"train": 10, "validation": 2, "calibration": 2, "test": 2})

    def test_combined_split_json_and_determinism(self):
        tables = cebab_fixture(self.source)
        combined = self.root / "all.json"
        write(combined, json.dumps(tables))
        result1 = self.prepare("cebab")
        result2 = prepare_dataset("cebab", combined, self.root / "out2")
        self.assertEqual(result1["membership"].read_bytes(), result2["membership"].read_bytes())
        self.assertEqual(result1["samples"].read_bytes(), result2["samples"].read_bytes())

    def test_family_and_exact_text_overlap_holdout_priority(self):
        tables = cebab_fixture(self.source)
        tables["train_exclusive"].append(review("edit", family="30", original=False))
        tables["train_exclusive"].append(review("duplicate", text=tables["dev"][0]["description"]))
        for name, rows in tables.items():
            write(self.source / (name + ".json"), json.dumps(rows))
        result = self.prepare("cebab")
        excluded = read_jsonl(result["exclusions"])
        self.assertEqual({s["sample_id"] for s in excluded}, {"cebab:edit", "cebab:duplicate"})
        self.assertEqual(result["report"]["checks"]["group_cross_split_overlap"], 0)
        membership = read_jsonl(result["membership"])
        groups = {}
        for row in membership:
            groups.setdefault(row["group_id"], set()).add(row["split"])
        self.assertTrue(all(len(roles) == 1 for roles in groups.values()))

    def test_target_missing_retained_not_counted_effective(self):
        tables = cebab_fixture(self.source)
        tables["test"][0]["review_majority"] = "no majority"
        tables["test"][1]["review_majority"] = None
        write(self.source / "test.json", json.dumps(tables["test"]))
        result = self.prepare("cebab")
        self.assertEqual(result["report"]["observed_counts"]["test"], 2)
        self.assertEqual(result["report"]["effective_task_counts"]["test"], 0)
        self.assertEqual(result["report"]["target_status_counts"]["test"],
                         {"NO_MAJORITY": 1, "MISSING_ANNOTATION": 1})

    def test_unknown_labels_fail_before_output(self):
        tables = cebab_fixture(self.source)
        tables["test"][0]["food_aspect_majority"] = "not annotated"
        write(self.source / "test.json", json.dumps(tables["test"]))
        with self.assertRaisesRegex(ValueError, "unknown food"):
            self.prepare("cebab")
        self.assertFalse((self.root / "prepared").exists())

    def test_invalid_distribution_and_duplicate_ids_fail(self):
        tables = cebab_fixture(self.source)
        tables["test"][0]["food_aspect_label_distribution"] = {"other": 5}
        write(self.source / "test.json", json.dumps(tables["test"]))
        with self.assertRaisesRegex(ValueError, "unknown key"):
            self.prepare("cebab")
        tables["test"][0] = review(0)
        write(self.source / "test.json", json.dumps(tables["test"]))
        with self.assertRaisesRegex(ValueError, "duplicate sample"):
            self.prepare("cebab")

    def test_cub_joins_certainty_and_group_expansion(self):
        cub_fixture(self.source)
        result = self.prepare("cub")
        schema = Schema.from_dict(json.loads(result["schema"].read_text()))
        self.assertEqual(schema.num_classes, 2)
        self.assertEqual(schema.num_groups, 2)
        self.assertEqual(schema.expand((0,)), (0, 1))
        samples = read_jsonl(result["samples"])
        one = next(s for s in samples if s["sample_id"] == "cub:1")
        two = next(s for s in samples if s["sample_id"] == "cub:2")
        self.assertEqual(one["target"]["value"], 0)
        self.assertEqual(one["concepts"][0]["value"], 1)
        self.assertEqual(one["concepts"][1]["annotation_status"], "NOT_VISIBLE")
        self.assertIsNone(one["concepts"][1]["value"])
        self.assertEqual(one["concepts"][2]["annotation_status"], "MISSING_ANNOTATION")
        self.assertEqual(two["concepts"][2]["annotation_status"], "UNCERTAIN_ANNOTATION")
        self.assertEqual(one["audit_metadata"]["certainty_annotations"]["1"], [1, 9, 0.2])
        self.assertNotIn("annotation_anomalies", result)
        self.assertNotIn("annotation_suffix_anomalies", result["report"]["adapter"])
        self.assertEqual(result["report"]["observed_counts"],
                         {"train": 18, "validation": 2, "calibration": 2, "test": 2})
        self.assertTrue(all(s["fold_id"] is None for s in samples if s["split"] != "train"))

    def test_skincon_joins_fitzpatrick_metadata_and_splits(self):
        skincon_fixture(self.source)
        result = self.prepare("skincon")
        schema = Schema.from_dict(json.loads(result["schema"].read_text()))
        samples = read_jsonl(result["samples"])
        self.assertEqual(schema.num_classes, 3)
        self.assertEqual(schema.num_atoms, 3)
        self.assertEqual(len(samples), 29)
        self.assertEqual(result["report"]["adapter"]["ignored_do_not_consider_count"], 1)
        sample = next(s for s in samples if s["sample_id"].endswith("{:032x}".format(2)))
        self.assertEqual(sample["input"]["image_paths"], ["images/{:032x}.jpg".format(2)])
        self.assertEqual(sample["audit_metadata"]["md5hash"], "{:032x}".format(2))
        self.assertEqual(sample["audit_metadata"]["source_url"],
                         "https://example.test/{:032x}.jpg".format(2))
        self.assertEqual([c["annotation_status"] for c in sample["concepts"]],
                         ["OBSERVED", "OBSERVED", "OBSERVED"])
        self.assertEqual(result["report"]["observed_counts"],
                         {"train": 25, "validation": 2, "calibration": 2, "test": 0})
        self.assertEqual(result["report"]["role_counts"]["responder_fit"], 13)
        self.assertEqual(result["report"]["role_counts"]["head_fit"], 3)
        self.assertEqual(result["report"]["role_counts"]["policy_fit"], 9)

    def test_skincon_bad_join_or_concept_fails(self):
        skincon_fixture(self.source)
        text = (self.source / "fitzpatrick17k.csv").read_text()
        (self.source / "fitzpatrick17k.csv").write_text(text.replace("{:032x}".format(2), "bad", 1))
        with self.assertRaisesRegex(ValueError, "md5hash"):
            self.prepare("skincon")
        self.temp.cleanup()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "raw"
        self.source.mkdir()
        skincon_fixture(self.source)
        rows, fields = [], None
        with (self.source / "annotations_fitzpatrick17k.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = list(reader)
        rows[1]["Papule"] = "maybe"
        with (self.source / "annotations_fitzpatrick17k.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "binary"):
            self.prepare("skincon")

    def test_isic2018_task2_masks_targets_and_partial_images(self):
        isic_fixture(self.source)
        result = self.prepare("isic2018_task2")
        schema = Schema.from_dict(json.loads(result["schema"].read_text()))
        samples = read_jsonl(result["samples"])
        self.assertEqual(schema.num_classes, 2)
        self.assertEqual(schema.num_atoms, 5)
        self.assertEqual(len(samples), 139)
        self.assertEqual(result["report"]["adapter"]["complete_mask_ids"], 140)
        self.assertEqual(result["report"]["adapter"]["skipped_missing_image_count"], 1)
        sample = next(s for s in samples if s["sample_id"] == "isic2018_task2:ISIC_0000001")
        self.assertEqual(sample["input"]["image_paths"], ["images/ISIC_0000001.jpg"])
        self.assertEqual(sample["target"]["value"], 0)
        self.assertEqual(sample["audit_metadata"]["target_class"], "Benign")
        self.assertEqual([c["value"] for c in sample["concepts"]], [0, 0, 1, 0, 0])
        self.assertEqual(result["report"]["observed_counts"],
                         {"train": 113, "validation": 13, "calibration": 13, "test": 0})

    def test_isic2018_task2_bad_one_hot_fails(self):
        isic_fixture(self.source)
        payload = json.loads((self.source / "isic_archive_metadata_task2.json").read_text())
        for record in payload["records"].values():
            record["metadata"]["clinical"]["diagnosis_1"] = "OnlyClass"
        write(self.source / "isic_archive_metadata_task2.json", json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "fewer than two"):
            self.prepare("isic2018_task2")

    def test_cub_verified_suffix_exception_preserves_semantics_and_audit(self):
        path = cub_suffix_anomaly_fixture(self.source)
        original = path.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        with patch("cbmjev.data.CUB_KNOWN_SIX_COLUMN_ANNOTATIONS_SHA256", digest):
            result = self.prepare("cub")
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(result["report"]["status"], "PREPARED_NOT_ACCEPTED")
        summary = result["report"]["adapter"]["annotation_suffix_anomalies"]
        self.assertEqual(summary["count"], 606)
        self.assertEqual(summary["source_sha256"], digest)
        self.assertEqual(summary["suffix_semantics"], "UNINTERPRETED")
        sidecar = read_jsonl(result["annotation_anomalies"])
        self.assertEqual(len(sidecar), 606)
        self.assertEqual({(r["image_id"], r["attribute_id"]) for r in sidecar},
                         {(i, a) for i in (2275, 9364) for a in range(10, 313)})
        source_lines = original.decode().splitlines()
        for row in sidecar:
            self.assertEqual(row["source_sha256"], digest)
            self.assertEqual(row["raw_line"], source_lines[row["line_number"] - 1])
            self.assertEqual(row["suffix_tokens"], ["0", "1.509"])
            self.assertIsNone(row["duration"])
            self.assertEqual(row["duration_status"], "AMBIGUOUS_TRAILING_FIELDS")
        samples = read_jsonl(result["samples"])
        for iid in (2275, 9364):
            sample = next(s for s in samples if s["sample_id"] == "cub:" + str(iid))
            self.assertEqual(len(sample["audit_metadata"]["annotation_suffix_anomaly_lines"]), 303)
            self.assertEqual(sample["audit_metadata"]["annotation_suffix_anomaly_sidecar"],
                             result["annotation_anomalies"].name)
            for attr in range(10, 313):
                self.assertEqual(sample["audit_metadata"]["certainty_annotations"][str(attr)], [1, 9, None])
                self.assertEqual(sample["concepts"][attr - 1]["value"], 1)
                self.assertEqual(sample["concepts"][attr - 1]["annotation_status"], "OBSERVED")
        ordinary = next(s for s in samples if s["sample_id"] == "cub:1")
        self.assertEqual(ordinary["audit_metadata"]["certainty_annotations"]["1"], [1, 9, 0.2])

    def test_cub_unknown_six_column_source_hash_is_rejected(self):
        cub_suffix_anomaly_fixture(self.source)
        with self.assertRaisesRegex(ValueError, "unrecognized six-column CUB annotation file SHA256"):
            self.prepare("cub")
        self.assertFalse((self.root / "prepared").exists())

    def test_cub_known_hash_cannot_bypass_suffix_key_or_semantic_validation(self):
        path = cub_suffix_anomaly_fixture(self.source)
        original = path.read_text()
        for replacement, error in (
                ("2275 10 1 9 0.0 1.509", "suffix/key pattern"),
                ("2275 10 1 9 0 -1", "suffix/key pattern"),
                ("2275 10 1 9 0 nan", "suffix/key pattern"),
                ("2275 10 1 9 0 unknown", "invalid known CUB annotation suffix"),
                ("2275 9 1 9 0 1.509", "suffix/key pattern"),
                ("2276 10 1 9 0 1.509", "suffix/key pattern"),
                ("2275 10 2 9 0 1.509", "invalid CUB annotation key/value"),
                ("2275 10 1 99 0 1.509", "invalid CUB annotation key/value"),
                ("2275 10 1 9 1.509", "complete 606-key pattern"),
                ("2275 10 1 9 0 1.509 extra", "must contain")):
            with self.subTest(replacement=replacement):
                write(path, original.replace("2275 10 1 9 0 1.509", replacement, 1))
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                with patch("cbmjev.data.CUB_KNOWN_SIX_COLUMN_ANNOTATIONS_SHA256", digest):
                    with self.assertRaisesRegex(ValueError, error):
                        self.prepare("cub")
        self.assertFalse((self.root / "prepared").exists())

    def test_cub_five_column_duration_remains_strict(self):
        cub_fixture(self.source)
        path = self.source / "attributes/image_attribute_labels.txt"
        original = path.read_text()
        for duration in ("-1", "nan", "inf", "not_a_number"):
            with self.subTest(duration=duration):
                write(path, original.replace("1 1 1 9 0.2\n", "1 1 1 9 " + duration + "\n", 1))
                with self.assertRaises(ValueError):
                    self.prepare("cub")

    def test_cub_reject_join_and_unsafe_path(self):
        cub_fixture(self.source)
        labels = self.source / "image_class_labels.txt"
        write(labels, labels.read_text().replace("24 2\n", ""))
        with self.assertRaisesRegex(ValueError, "joins"):
            self.prepare("cub")
        cub_fixture(self.source)
        images = self.source / "images.txt"
        write(images, images.read_text().replace("pic1.ppm", "../../outside.ppm"))
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.prepare("cub")

    def test_unknown_certainty_and_duplicate_annotation(self):
        cub_fixture(self.source)
        write(self.source / "attributes/certainties.txt", "9 confident\n")
        with self.assertRaisesRegex(ValueError, "unknown CUB certainty"):
            self.prepare("cub")
        cub_fixture(self.source)
        path = self.source / "attributes/image_attribute_labels.txt"
        write(path, path.read_text() + "1 1 1 9 0.2\n")
        with self.assertRaisesRegex(ValueError, "duplicate CUB"):
            self.prepare("cub")

    def test_image_absolute_path_and_escaping_symlink_rejected(self):
        cub_fixture(self.source)
        images = self.source / "images.txt"
        original = images.read_text()
        write(images, original.replace("pic1.ppm", "/pic1.ppm"))
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.prepare("cub")
        write(images, original)
        outside = self.root / "outside.ppm"
        outside.write_bytes((self.source / "images/pic1.ppm").read_bytes())
        (self.source / "images/link.ppm").symlink_to(outside)
        write(images, original.replace("pic1.ppm", "link.ppm"))
        with self.assertRaisesRegex(ValueError, "symlink escapes"):
            self.prepare("cub")

    def test_valid_family_zero_and_unverified_sentinel(self):
        tables = cebab_fixture(self.source)
        tables["train_exclusive"][0]["original_id"] = "000000"
        write(self.source / "train_exclusive.json", json.dumps(tables["train_exclusive"]))
        result = self.prepare("cebab")
        first = next(s for s in read_jsonl(result["samples"]) if s["sample_id"] == "cebab:0")
        self.assertEqual(first["group_id"], "cebab:family:000000")
        tables["train_exclusive"][0]["original_id"] = "-1"
        write(self.source / "train_exclusive.json", json.dumps(tables["train_exclusive"]))
        with self.assertRaisesRegex(ValueError, "sentinel"):
            prepare_dataset("cebab", self.source, self.root / "out2")

    def test_derm_zero_based_row_join_and_categorical_values(self):
        derm_fixture(self.source)
        result = self.prepare("derm7pt")
        schema = Schema.from_dict(json.loads(result["schema"].read_text()))
        self.assertEqual(schema.num_atoms, 7)
        self.assertEqual(sum(schema.value_counts), 19)
        samples = read_jsonl(result["samples"])
        first = next(s for s in samples if s["sample_id"] == "derm7pt:0")
        self.assertEqual(first["input"]["image_paths"], ["images/d0.ppm"])
        self.assertEqual(first["audit_metadata"]["case_num"], "500")
        self.assertEqual(first["audit_metadata"]["clinic_image_path"], "images/c0.ppm")
        self.assertEqual(first["concepts"][2]["value"], 2)
        self.assertEqual(first["target"]["value"], 1)
        self.assertEqual(result["report"]["observed_counts"],
                         {"train": 8, "validation": 1, "calibration": 1, "test": 2})

    def test_derm_reject_bad_header_or_overlapping_positions(self):
        derm_fixture(self.source)
        write(self.source / "meta/test_indexes.csv", "index\n10\n11\n")
        with self.assertRaisesRegex(ValueError, "indexes.*header"):
            self.prepare("derm7pt")
        write(self.source / "meta/test_indexes.csv", "indexes\n9\n11\n")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.prepare("derm7pt")

    def test_derm_missing_concept_is_not_absent(self):
        derm_fixture(self.source)
        path = self.source / "meta/meta.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["blue_whitish_veil"] = ""
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = self.prepare("derm7pt", decode_images=False)
        first = next(s for s in read_jsonl(result["samples"]) if s["sample_id"] == "derm7pt:0")
        self.assertIsNone(first["concepts"][1]["value"])
        self.assertEqual(first["concepts"][1]["annotation_status"], "MISSING_ANNOTATION")
        self.assertEqual(result["report"]["duplicates"]["image_decode_and_pixel_hash"], "NOT_CHECKED")

    def test_unchecked_audits_are_explicit_and_no_overwrite(self):
        cebab_fixture(self.source)
        result = self.prepare("cebab", source_revision="fixture-only")
        self.assertEqual(result["report"]["status"], "PREPARED_NOT_ACCEPTED")
        for key in ("license_and_access", "near_duplicate_phash", "source_authenticity"):
            self.assertEqual(result["report"]["checks"][key], "NOT_CHECKED")
        before = result["samples"].read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare("cebab")
        self.assertEqual(before, result["samples"].read_bytes())


if __name__ == "__main__":
    unittest.main()

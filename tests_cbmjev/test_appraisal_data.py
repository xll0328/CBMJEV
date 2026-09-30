"""Synthetic native-format fixtures; never download or inspect real row texts."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from cbmjev.appraisal_data import (
    APPRAISAL_FIELDS, EMOTIONS, GENERATION_FILE, TARGET_SEMANTICS,
    VALIDATION_FILE, appraisal_schema, inspect_crowd_envent, load_role_records,
    normalize_text, prepare_appraisal_data, preprocess_text,
)
from cbmjev.contracts import Schema
from cbmjev.io import read_jsonl


def fixture_rows(n=130):
    return [dict({field: str((i + j) % 5 + 1) for j, field in enumerate(APPRAISAL_FIELDS)},
                 text_id=str(i), prolific_id="writer-" + str(i // 2), emotion=EMOTIONS[i % 13],
                 generated_text="I felt joy when the synthetic event number {} happened.".format(i),
                 hidden_emo_text="DO NOT USE LABEL-AWARE TEXT", age="FORBIDDEN DEMOGRAPHICS")
            for i in range(n)]


def write_fixture(root, rows, validated=(0, 2, 4)):
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    with (root / GENERATION_FILE).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with (root / VALIDATION_FILE).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text_id", "emotion", "prolific_id"], delimiter="\t")
        writer.writeheader()
        for sid in validated:
            for reader in range(5):
                writer.writerow({"text_id": str(sid), "emotion": "READER_TARGET_NOT_USED",
                                 "prolific_id": "reader-" + str(reader)})
    (root / "readme.txt").write_text("Synthetic fixture; emotion is the prompting emotion.\n", encoding="utf-8")


class AppraisalDataTests(unittest.TestCase):
    def test_native_schema_and_question_direction(self):
        schema = Schema.from_dict(appraisal_schema())
        self.assertEqual(schema.num_classes, 13)
        self.assertEqual(schema.value_counts, (5,) * 21)
        self.assertIn("violate", schema.concepts[17].description)
        self.assertTrue(all("1 = Not at all; 5 = Extremely" in c.description for c in schema.concepts))

    def test_text_masking_is_fixed_and_label_independent(self):
        text, count, prefixes = preprocess_text("I felt JOY when sadness, anger and no emotion appeared.")
        self.assertEqual(text, "[EMOTION], [EMOTION] and [EMOTION] appeared.")
        self.assertEqual((count, prefixes), (4, 1))
        self.assertEqual(preprocess_text("Enjoyment, trusted and angry remain synonyms.")[1], 0)
        self.assertEqual(normalize_text("Ｆｏｏ —  BAR!"), "foo bar")

    def test_native_labels_and_input_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = fixture_rows()
            rows[0]["emotion"] = "fear"
            rows[0]["effort"] = "5"
            rows[1]["effort"] = ""
            write_fixture(root / "raw", rows)
            result = prepare_appraisal_data(root / "raw", root / "out")
            samples = {s["provenance"]["source_id"]: s for s in read_jsonl(result["samples"])}
            self.assertEqual(samples["0"]["target"]["value"], EMOTIONS.index("fear"))
            self.assertEqual(samples["0"]["concepts"][-1]["value"], 4)
            self.assertIsNone(samples["1"]["concepts"][-1]["value"])
            self.assertEqual(samples["1"]["concepts"][-1]["annotation_status"], "MISSING_ANNOTATION")
            self.assertEqual(samples["0"]["input"]["text"], "the synthetic event number 0 happened.")
            input_json = json.dumps([s["input"] for s in samples.values()])
            self.assertNotIn("FORBIDDEN", input_json)
            self.assertNotIn("writer-", input_json)
            self.assertNotIn("LABEL-AWARE", input_json)
            self.assertEqual(result["report"]["target_semantics"], TARGET_SEMANTICS)

    def test_author_text_transitive_components_and_disjoint_roles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = fixture_rows()
            # A duplicate text joins two author groups; each author's other row
            # must follow into the same component, even with different targets.
            rows[2]["generated_text"] = rows[0]["generated_text"].upper()
            write_fixture(root / "raw", rows)
            result = prepare_appraisal_data(root / "raw", root / "out")
            members = read_jsonl(result["membership"])
            by_id = {m["sample_id"]: m for m in members}
            first_four = [by_id["crowd_envent:" + str(i)] for i in range(4)]
            self.assertEqual(len({m["group_id"] for m in first_four}), 1)
            self.assertEqual(len({m["split"] for m in first_four}), 1)
            roles = {}
            for member in members:
                self.assertEqual(roles.setdefault(member["group_id"], member["split"]), member["split"])
            self.assertEqual(set(m["split"] for m in members),
                             {"responder_fit", "head_fit", "policy_fit", "policy_tune", "confirmation"})

    def test_identical_masked_inputs_connect_even_when_raw_words_differ(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = fixture_rows()
            rows[2]["generated_text"] = rows[0]["generated_text"].replace("joy", "fear")
            write_fixture(root / "raw", rows)
            result = prepare_appraisal_data(root / "raw", root / "out")
            by_id = {m["sample_id"]: m for m in read_jsonl(result["membership"])}
            self.assertEqual(by_id["crowd_envent:0"]["group_id"], by_id["crowd_envent:2"]["group_id"])

    def test_membership_independent_of_source_row_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = fixture_rows()
            write_fixture(root / "raw1", rows)
            write_fixture(root / "raw2", list(reversed(rows)))
            one = prepare_appraisal_data(root / "raw1", root / "one")
            two = prepare_appraisal_data(root / "raw2", root / "two")
            self.assertEqual(one["report"]["split_hash"], two["report"]["split_hash"])

    def test_official_test_author_overlap_is_described_not_hidden(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root, fixture_rows())
            info = inspect_crowd_envent(root)
            self.assertEqual(info["official_test_rows"], 3)
            self.assertEqual(info["official_author_intersection"], 3)
            self.assertEqual(info["official_test_rows_with_non_test_author"], 3)
            self.assertNotIn("synthetic event number", json.dumps(info))
            self.assertNotIn("writer-", json.dumps(info))

    def test_missing_author_column_declares_text_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = fixture_rows()
            for row in rows:
                del row["prolific_id"]
            write_fixture(root, rows)
            info = inspect_crowd_envent(root)
            self.assertEqual(info["author_grouping"], "unavailable_text_groups_only")
            self.assertIsNone(info["unique_authors"])
            self.assertEqual(info["components"], 130)

    def test_partial_missing_authors_and_invalid_native_values_fail(self):
        for field, invalid in (("prolific_id", ""), ("effort", "0"), ("emotion", "invented")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                rows = fixture_rows()
                rows[0][field] = invalid
                write_fixture(Path(temp), rows)
                with self.assertRaises(ValueError):
                    inspect_crowd_envent(Path(temp))

    def test_locked_role_default_and_mutated_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root / "raw", fixture_rows())
            result = prepare_appraisal_data(root / "raw", root / "out")
            members = read_jsonl(result["membership"])
            expected = {m["sample_id"] for m in members if m["split"] == "responder_fit"}
            loaded = load_role_records(root / "out", "responder_fit")
            self.assertEqual({s["sample_id"] for s in loaded}, expected)
            with self.assertRaisesRegex(ValueError, "locked"):
                load_role_records(root / "out", ["responder_fit", "confirmation"])
            with self.assertRaises(ValueError):
                load_role_records(root / "out", "train")
            self.assertTrue(load_role_records(root / "out", "confirmation", allow_confirmation=True))
            members[0]["split"] = "head_fit" if members[0]["split"] == "confirmation" else "confirmation"
            result["membership"].write_text("\n".join(json.dumps(m) for m in members), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                load_role_records(root / "out", "responder_fit")

    def test_zip_reader_never_uses_packaged_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root / "raw", fixture_rows())
            with zipfile.ZipFile(root / "release.zip", "w") as archive:
                for name in (GENERATION_FILE, VALIDATION_FILE, "readme.txt"):
                    archive.write(root / "raw" / name, name)
                archive.writestr("predictions/do_not_read.tsv", "MALFORMED MODEL OUTCOMES")
            info = inspect_crowd_envent(root / "release.zip")
            self.assertEqual(info["generation_rows"], 130)
            self.assertNotIn("MALFORMED", json.dumps(info))

    def test_no_overwrite_and_duplicate_ids_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = fixture_rows()
            write_fixture(root / "raw", rows)
            prepare_appraisal_data(root / "raw", root / "out")
            with self.assertRaisesRegex(ValueError, "new or empty"):
                prepare_appraisal_data(root / "raw", root / "out")
            rows[1]["text_id"] = rows[0]["text_id"]
            write_fixture(root / "bad", rows)
            with self.assertRaisesRegex(ValueError, "unique"):
                inspect_crowd_envent(root / "bad")


if __name__ == "__main__":
    unittest.main()

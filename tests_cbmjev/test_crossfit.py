"""Prepared-fold nested crossfit planning tests; fixtures are synthetic only."""

import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from cbmjev.contracts import stable_hash
from cbmjev.crossfit import plan_crossfit_prepared, verify_crossfit_prepared
from cbmjev.io import file_hash
from cbmjev.cli import main as cli_main


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


def fixture(root):
    prepared = root / "prepared"
    prepared.mkdir()
    schema = {
        "schema_version": "cbmjev-schema-v1", "dataset": "fixture", "num_classes": 2,
        "concepts": [{"id": "c", "description": "concept", "values": ["no", "yes"]}],
        "groups": [{"id": "c", "atoms": [0]}],
    }
    rows, membership = [], []
    # Twelve train groups: two target strata, four groups per frozen outer fold.
    for index in range(12):
        group = "g{:02d}".format(index)
        role = ("responder_fit", "head_fit", "policy_fit")[index % 3]
        rows.append({
            "schema_version": "acam-data-v1", "dataset": "fixture",
            "sample_id": "s{:02d}".format(index), "group_id": group,
            "split": "train", "fold_id": index % 3,
            "input": {"modality": "text", "text": "row {}".format(index), "image_paths": []},
            "target": {"value": index % 2, "status": "OBSERVED"},
            "concepts": [{"concept_id": "c", "value": index % 2,
                          "annotation_status": "OBSERVED", "distribution": None}],
            "provenance": {"source_id": str(index), "source_split": "train", "raw_sha256": "x"},
            "audit_metadata": {},
        })
        membership.append({"sample_id": "s{:02d}".format(index), "group_id": group,
                           "split": role, "outer_split": "train"})
    for offset, split in enumerate(("validation", "calibration", "test"), 12):
        rows.append({
            "schema_version": "acam-data-v1", "dataset": "fixture",
            "sample_id": "s{:02d}".format(offset), "group_id": "g{:02d}".format(offset),
            "split": split, "fold_id": None,
            "input": {"modality": "text", "text": split, "image_paths": []},
            "target": {"value": offset % 2, "status": "OBSERVED"},
            "concepts": [{"concept_id": "c", "value": 0,
                          "annotation_status": "OBSERVED", "distribution": None}],
            "provenance": {"source_id": str(offset), "source_split": split, "raw_sha256": "x"},
            "audit_metadata": {},
        })
        membership.append({"sample_id": "s{:02d}".format(offset),
                           "group_id": "g{:02d}".format(offset),
                           "split": split, "outer_split": split})
    write_json(prepared / "schema.json", schema)
    write_jsonl(prepared / "samples.jsonl", rows)
    write_jsonl(prepared / "membership.jsonl", membership)
    audit = {"dataset": "fixture", "seed": 17, "fold_count": 3,
             "schema_hash": stable_hash(schema), "split_hash": stable_hash(membership)}
    write_json(prepared / "audit.json", audit)
    return prepared


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class CrossfitPreparedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.prepared = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def plan(self, name="plan", inner_folds=2):
        return plan_crossfit_prepared(self.prepared, self.root / name,
                                      inner_folds=inner_folds)

    def rewrite_samples(self, mutate):
        path = self.prepared / "samples.jsonl"
        rows = load_rows(path)
        mutate(rows)
        write_jsonl(path, rows)

    def rewrite_members(self, mutate, update_audit=True):
        path = self.prepared / "membership.jsonl"
        rows = load_rows(path)
        mutate(rows)
        write_jsonl(path, rows)
        if update_audit:
            audit = json.loads((self.prepared / "audit.json").read_text())
            audit["split_hash"] = stable_hash(rows)
            write_json(self.prepared / "audit.json", audit)

    def test_stable_bytes_hashes_and_verification(self):
        first = self.plan("a")
        second = self.plan("b")
        self.assertEqual(first["plan_hash"], second["plan_hash"])
        self.assertEqual((self.root / "a/plan.json").read_bytes(),
                         (self.root / "b/plan.json").read_bytes())
        self.assertEqual((self.root / "a/fold_manifest.jsonl").read_bytes(),
                         (self.root / "b/fold_manifest.jsonl").read_bytes())
        receipt = json.loads((self.root / "a/plan.json").read_text())["receipt"]
        self.assertEqual(receipt["prepared_hash"], receipt["prepared_files_hash"])
        self.assertEqual(receipt["fold_hash"], receipt["fold_manifest_hash"])
        self.assertEqual(receipt["group_ids_hash"], receipt["eligible_group_ids_hash"])
        self.assertEqual(verify_crossfit_prepared(self.prepared, self.root / "a")["status"],
                         "PASS")

    def test_frozen_outer_and_recomputed_inner_assignments_have_no_leakage(self):
        self.plan()
        plan = json.loads((self.root / "plan/plan.json").read_text())
        eligible = set(plan["eligible_group_ids"])
        excluded = {g for values in plan["excluded_group_ids_by_outer_split"].values()
                    for g in values}
        self.assertEqual(excluded, {"g12", "g13", "g14"})
        self.assertFalse(eligible & excluded)
        outer_targets = set()
        manifest = {row["group_id"]: row for row in load_rows(
            self.root / "plan/fold_manifest.jsonl")}
        for outer in plan["folds"]:
            fit, target = set(outer["fit_group_ids"]), set(outer["target_group_ids"])
            self.assertFalse(fit & target)
            self.assertEqual(fit | target, eligible)
            outer_targets.update(target)
            inner_targets = set()
            for inner in outer["inner_folds"]:
                inner_fit, inner_target = set(inner["fit_group_ids"]), set(inner["target_group_ids"])
                self.assertFalse(inner_fit & inner_target)
                self.assertEqual(inner_fit | inner_target, fit)
                self.assertFalse(inner_target & target)
                self.assertFalse(inner_target & excluded)
                inner_targets.update(inner_target)
            self.assertEqual(inner_targets, fit)
        self.assertEqual(outer_targets, eligible)
        prepared = {row["group_id"]: row for row in load_rows(
            self.prepared / "samples.jsonl")}
        for group_id in eligible:
            frozen = prepared[group_id]["fold_id"]
            self.assertEqual(manifest[group_id]["outer_fold_id"], frozen)
            self.assertEqual(set(manifest[group_id]["inner_fold_by_outer_fold"]),
                             {str(fold) for fold in range(3) if fold != frozen})

    def test_cross_fold_group_is_rejected(self):
        def mutate(rows):
            rows[1]["group_id"] = rows[0]["group_id"]
        self.rewrite_samples(mutate)
        self.rewrite_members(lambda rows: rows[1].update(group_id=rows[0]["group_id"],
                                                        split=rows[0]["split"]))
        with self.assertRaisesRegex(ValueError, "crosses prepared fold_id"):
            self.plan()

    def test_heldout_fold_is_rejected(self):
        self.rewrite_samples(lambda rows: rows[12].update(fold_id=0))
        with self.assertRaisesRegex(ValueError, "held-out sample must not have fold_id"):
            self.plan()

    def test_group_crossing_role_or_outer_split_is_rejected(self):
        self.rewrite_samples(lambda rows: rows[1].update(group_id=rows[0]["group_id"], fold_id=0))
        self.rewrite_members(lambda rows: rows[1].update(group_id=rows[0]["group_id"],
                                                        split="head_fit"))
        with self.assertRaisesRegex(ValueError, "crosses prepared split/role"):
            self.plan()

        # Start with a clean fixture for the outer-split case.
        self.temp.cleanup()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.prepared = fixture(self.root)
        self.rewrite_samples(lambda rows: rows[12].update(group_id=rows[0]["group_id"]))
        self.rewrite_members(lambda rows: rows[12].update(group_id=rows[0]["group_id"]))
        with self.assertRaisesRegex(ValueError, "crosses prepared split/role"):
            self.plan()

    def test_missing_train_fold_is_rejected(self):
        self.rewrite_samples(lambda rows: rows[0].update(fold_id=None))
        with self.assertRaisesRegex(ValueError, "train fold_id"):
            self.plan()

    def test_bool_and_noncontiguous_folds_are_rejected(self):
        self.rewrite_samples(lambda rows: rows[0].update(fold_id=True))
        with self.assertRaisesRegex(ValueError, "train fold_id"):
            self.plan()

        self.temp.cleanup()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.prepared = fixture(self.root)
        self.rewrite_samples(lambda rows: [row.update(fold_id=4) for row in rows
                                           if row.get("fold_id") == 2])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.plan()

    def test_missing_target_structure_is_rejected_not_treated_as_null(self):
        corruptions = (
            ("missing-target", lambda rows: rows[0].pop("target")),
            ("non-object-target", lambda rows: rows[0].update(target=None)),
            ("missing-value", lambda rows: rows[0].update(target={"status": "OBSERVED"})),
        )
        for index, (name, corrupt) in enumerate(corruptions):
            with self.subTest(name=name):
                if index:
                    self.temp.cleanup()
                    self.temp = tempfile.TemporaryDirectory()
                    self.root = Path(self.temp.name)
                    self.prepared = fixture(self.root)
                self.rewrite_samples(corrupt)
                with self.assertRaisesRegex(ValueError, "explicit target.value"):
                    self.plan()

    def test_explicit_null_target_value_remains_a_missing_target_stratum(self):
        self.rewrite_samples(lambda rows: rows[0]["target"].update(value=None))
        self.plan()
        manifest = {row["group_id"]: row for row in load_rows(
            self.root / "plan/fold_manifest.jsonl")}
        self.assertEqual(manifest["g00"]["stratum"], "missing-target")

    def test_membership_tamper_against_audit_is_rejected(self):
        self.rewrite_members(lambda rows: rows[0].update(split="head_fit"), update_audit=False)
        with self.assertRaisesRegex(ValueError, "audit.split_hash"):
            self.plan()

    def test_manifest_tamper_is_detected(self):
        self.plan()
        path = self.root / "plan/fold_manifest.jsonl"
        rows = load_rows(path)
        rows[0]["outer_fold_id"] = 99
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "fold_manifest byte hash mismatch"):
            verify_crossfit_prepared(self.prepared, self.root / "plan")

    def test_semantically_equal_noncanonical_manifest_bytes_are_rejected(self):
        self.plan()
        manifest_path = self.root / "plan/fold_manifest.jsonl"
        rows = load_rows(manifest_path)
        # Re-encode with sorted keys, then update all self-reported hashes. An
        # exact deterministic rebuild must still reject these different bytes.
        write_jsonl(manifest_path, rows)
        plan_path = self.root / "plan/plan.json"
        plan = json.loads(plan_path.read_text())
        plan["receipt"]["fold_manifest_sha256"] = file_hash(manifest_path)
        plan.pop("plan_hash")
        plan["plan_hash"] = stable_hash(plan)
        write_json(plan_path, plan)
        with self.assertRaisesRegex(ValueError, "not the canonical deterministic encoding"):
            verify_crossfit_prepared(self.prepared, self.root / "plan")

    def test_plan_tamper_is_detected(self):
        self.plan()
        path = self.root / "plan/plan.json"
        plan = json.loads(path.read_text())
        plan["eligible_group_ids"].append("g14")
        write_json(path, plan)
        with self.assertRaisesRegex(ValueError, "plan_hash mismatch"):
            verify_crossfit_prepared(self.prepared, self.root / "plan")

    def test_prepared_input_tamper_after_planning_is_detected(self):
        self.plan()
        self.rewrite_samples(lambda rows: rows[0]["target"].update(value=1))
        with self.assertRaisesRegex(ValueError, "differs from deterministic"):
            verify_crossfit_prepared(self.prepared, self.root / "plan")

    def test_plan_and_verify_cli_smoke(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cli_main(["plan-crossfit-prepared", "--prepared", str(self.prepared),
                             "--out", str(self.root / "cli-plan"), "--inner-folds", "2"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "PLANNED_NOT_TRAINED")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cli_main(["verify-crossfit-prepared", "--prepared", str(self.prepared),
                             "--planned", str(self.root / "cli-plan")])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "PASS")


if __name__ == "__main__":
    unittest.main()

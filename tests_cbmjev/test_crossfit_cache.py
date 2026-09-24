"""Synthetic real-backend OOF cache roundtrips and adversarial artifact checks."""
import tempfile
import unittest
from pathlib import Path
import shutil
from unittest.mock import patch

from cbmjev.contracts import stable_hash
from cbmjev.crossfit import plan_crossfit_prepared
from cbmjev.crossfit_cache import cache_crossfit_responses, load_crossfit_cache
from cbmjev.io import file_hash, read_json, read_jsonl
from cbmjev.pipeline import train_crossfit_responder
from cbmjev.provenance import make_fit_record
from tests_cbmjev.test_crossfit import fixture, write_json, write_jsonl


class CrossfitCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = fixture(self.root)
        audit = read_json(self.prepared / "audit.json")
        audit["source_revision"] = "SYNTHETIC_CROSSFIT_CACHE_FIXTURE"
        write_json(self.prepared / "audit.json", audit)
        self.plan = self.root / "plan"

    def train(self, **selectors):
        if not self.plan.exists():
            plan_crossfit_prepared(self.prepared, self.plan, inner_folds=2)
        responder = self.root / ("responder" + str(len(list(self.root.iterdir()))))
        train_crossfit_responder(self.prepared, self.plan, responder,
                                epochs=1, batch_size=4, **selectors)
        return responder

    def cache(self, responder, **kwargs):
        out = self.root / ("cache" + str(len(list(self.root.iterdir()))))
        cache_crossfit_responses(self.prepared, self.plan, responder, out, **kwargs)
        return out

    def load(self, cache, responder):
        return load_crossfit_cache(cache, self.prepared, self.plan, responder)

    def rebind(self, cache):
        manifest = read_json(cache / "manifest.json")
        manifest["files_sha256"] = {name: file_hash(cache / name)
                                    for name in manifest["files_sha256"]}
        manifest.pop("manifest_hash")
        manifest["manifest_hash"] = stable_hash(manifest)
        write_json(cache / "manifest.json", manifest)

    def test_real_outer_inner_and_final_roundtrip(self):
        for selectors, split in (({"outer_fold": 0}, None),
                                  ({"outer_fold": 0, "inner_fold": 1}, None),
                                  ({"final": True}, "validation"),
                                  ({"final": True}, "calibration")):
            with self.subTest(selectors=selectors, split=split):
                responder = self.train(**selectors)
                cache = self.cache(responder, split=split)
                schema, rows, manifest = self.load(cache, responder)
                self.assertTrue(rows)
                self.assertEqual(schema.num_classes, 2)
                self.assertEqual(set(manifest["response_artifact_by_group"]),
                                 {r["group_id"] for r in rows})
                self.assertEqual(manifest["evidence_status"], "OFFLINE_RESPONSES_NOT_LATENCY_EVIDENCE")
                self.assertFalse(set(manifest["target_group_ids"]) &
                                 set(manifest["crossfit"]["fit_group_ids"]))
                with self.assertRaises(ValueError):
                    cache_crossfit_responses(self.prepared, self.plan, responder, cache, split=split)

    def test_missing_target_is_explicit_exclusion(self):
        records = read_jsonl(self.prepared / "samples.jsonl")
        records[0]["target"] = {"value": None, "status": "UNANNOTATED"}
        write_jsonl(self.prepared / "samples.jsonl", records)
        responder = self.train(outer_fold=0)
        cache = self.cache(responder)
        _, rows, manifest = self.load(cache, responder)
        self.assertEqual(len(rows), 3)
        self.assertEqual(manifest["num_excluded_targets"], 1)
        self.assertEqual(read_jsonl(cache / "exclusions.jsonl")[0]["sample_id"], "s00")
        write_jsonl(cache / "exclusions.jsonl", [])
        self.rebind(cache)
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.load(cache, responder)

    def test_final_test_and_implicit_split_rejected(self):
        responder = self.train(final=True)
        for split in (None, "test", "train"):
            with self.assertRaisesRegex(ValueError, "explicit"):
                self.cache(responder, split=split)

    def test_rehashed_omission_label_and_category_tamper_rejected(self):
        responder = self.train(outer_fold=0)
        for mutation in (lambda rows: rows.pop(),
                         lambda rows: rows[0].update(y=1 - rows[0]["y"]),
                         lambda rows: rows[0].update(z=[True]),
                         lambda rows: rows[0].update(z=[99]),
                         lambda rows: rows[0].update(responder_artifact_id="fake")):
            cache = self.cache(responder)
            rows = read_jsonl(cache / "responses.jsonl")
            mutation(rows)
            write_jsonl(cache / "responses.jsonl", rows)
            self.rebind(cache)
            with self.assertRaises(ValueError):
                self.load(cache, responder)

    def test_changed_receipt_and_checkpoint_rejected(self):
        responder = self.train(outer_fold=0)
        cache = self.cache(responder)
        receipt = read_json(responder / "receipt.json")
        receipt["seed"] += 1
        write_json(responder / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "source binding"):
            self.load(cache, responder)
        with (responder / "responder.pt").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.load(cache, responder)

    def test_empty_or_false_fit_provenance_rejected(self):
        responder = self.train(outer_fold=0)
        receipt = read_json(responder / "receipt.json")
        for records in ([], [make_fit_record(receipt["artifact_id"], supervised_group_ids=[],
                                            fit_kind="frozen_external")]):
            receipt["provenance"] = records
            write_json(responder / "receipt.json", receipt)
            with self.assertRaises(ValueError):
                self.cache(responder)

    def test_inner_ancestor_leak_into_outer_targets_rejected(self):
        responder = self.train(outer_fold=0, inner_fold=1)
        receipt = read_json(responder / "receipt.json")
        for leaked_group in ("g00", "g12", "g13", "g14"):
            receipt["provenance"] = [
                make_fit_record("leaking-parent", supervised_group_ids=[leaked_group]),
                make_fit_record(receipt["artifact_id"], supervised_group_ids=receipt["supervised_group_ids"],
                                parent_ids=["leaking-parent"])]
            write_json(responder / "receipt.json", receipt)
            with self.assertRaisesRegex(ValueError, "leakage"):
                self.cache(responder)

    def test_rehashed_source_assignment_and_batch_evidence_rejected(self):
        responder = self.train(outer_fold=0)
        for key, replacement in (("response_artifact_by_group", {}),
                                 ("batch_invariance_check", {"passed": True})):
            cache = self.cache(responder)
            manifest = read_json(cache / "manifest.json")
            manifest[key] = replacement
            write_json(cache / "manifest.json", manifest)
            self.rebind(cache)
            with self.assertRaises(ValueError):
                self.load(cache, responder)

    def test_declared_or_observed_batch_dependence_rejected(self):
        responder = self.train(outer_fold=0)
        with patch("cbmjev.crossfit_cache.verify_batch_invariance", return_value={"passed": False}):
            with self.assertRaisesRegex(ValueError, "batch-dependent"):
                self.cache(responder)
        receipt = read_json(responder / "receipt.json")
        receipt["batch_independent"] = False
        write_json(responder / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "batch_independent"):
            self.cache(responder)

    def test_explicit_historical_source_bridge_is_read_only_and_verified(self):
        responder = self.train(outer_fold=0)
        cache = self.cache(responder)
        release = self.root / "release"
        shutil.copytree(Path(__file__).parents[1] / "cbmjev", release / "cbmjev",
                        ignore=shutil.ignore_patterns("__pycache__"))
        with patch("cbmjev.crossfit_cache.responder_code_fingerprint", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "semantic_code_hash"):
                self.load(cache, responder)
            _, bridged_rows, bridged_manifest = load_crossfit_cache(
                cache, self.prepared, self.plan, responder, responder_source_dir=release)
        self.assertEqual(len(bridged_rows), 4)
        self.assertNotIn("response_source_binding", bridged_manifest)
        with (release / "cbmjev/responders.py").open("ab") as handle:
            handle.write(b"\n# changed historical source\n")
        with self.assertRaisesRegex(ValueError, "source_code_hash"):
            load_crossfit_cache(cache, self.prepared, self.plan, responder,
                                responder_source_dir=release)


if __name__ == "__main__":
    unittest.main()

"""Synthetic integration checks; no scientific performance claims."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from cbmjev import pipeline
from cbmjev.io import read_json, read_jsonl, file_hash
from cbmjev.static_mask_artifacts import load_static_mask


class StaticMaskArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="static-mask-artifacts-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        pipeline.smoke(cls.root / "smoke", seed=17)
        cls.models, cls.cache = cls.root / "smoke/models", cls.root / "smoke/cache"
        for k in (0, 2, 4):
            pipeline.train_static_mask(cls.cache, cls.models, cls.root / str(k), config=dict(k=k, epochs=1))

    def test_roundtrip_one_batch_and_endpoint_parity(self):
        for k in (0, 2, 4):
            artifact = self.root / str(k)
            mask, report = load_static_mask(artifact, self.models, self.cache)
            self.assertTrue(report["head_exclusion_provenance_checked"])
            out = self.root / ("eval" + str(k))
            methods = ["learned_static_mask"] + (["all"] if k == 4 else [])
            pipeline.evaluate_models(self.models, self.cache, out, methods=methods,
                                     static_mask_dir=artifact, max_groups=k)
            rows = read_jsonl(out / "traces.jsonl")
            actual = [r for r in rows if r["method"] == "learned_static_mask"]
            for r in actual:
                self.assertEqual(tuple(r["queried_groups"]), mask.selected_groups())
                self.assertEqual(r["calls"], int(k > 0))
            if k == 4:
                reference = {r["sample_id"]: r for r in rows if r["method"] == "all"}
                self.assertTrue(all(r["prediction"] == reference[r["sample_id"]]["prediction"] for r in actual))
            self.assertIsNotNone(read_json(out / "metrics.json")["static_mask_identity"])

    def test_budget_and_formal_paths_rejected(self):
        for args in (dict(max_groups=1), dict(split="test", evaluate_test=True), dict(certification_run=True)):
            with self.assertRaises(ValueError):
                pipeline.evaluate_models(self.models, self.cache, self.root / "bad-eval", methods=["learned_static_mask"],
                                         static_mask_dir=self.root / "2", **args)
        with self.assertRaises(ValueError):
            pipeline.freeze_family(self.models, self.cache, self.root / "bad-freeze", methods=["learned_static_mask"])

    def test_missing_receipt_hash_and_ancestry_tampering(self):
        for mode in ("missing", "hash", "ancestry"):
            out = self.root / mode
            shutil.copytree(self.root / "2", out)
            if mode == "missing":
                (out / "receipt.json").unlink()
                expected = FileNotFoundError
            elif mode == "hash":
                (out / "training.json").write_text("{}")
                expected = ValueError
            else:
                r = read_json(out / "receipt.json")
                r["provenance"][-1]["parent_ids"] = []
                (out / "receipt.json").write_text(json.dumps(r))
                expected = ValueError
            with self.assertRaises(expected):
                load_static_mask(out, self.models, self.cache)

    def test_source_mutation_leaves_no_completion_receipt(self):
        out = self.root / "source-mutated"
        with patch.object(pipeline, "code_fingerprint", side_effect=["before", "after"]):
            with self.assertRaisesRegex(ValueError, "source or parent changed"):
                pipeline.train_static_mask(self.cache, self.models, out, config=dict(k=2, epochs=1))
        self.assertFalse((out / "receipt.json").exists())

    def test_failed_parent_exclusion_prevents_fit(self):
        out = self.root / "invalid-exclusion"
        with patch("cbmjev.static_mask_artifacts.validate_target_exclusion", side_effect=ValueError("ancestor supervision overlap")), \
             patch("cbmjev.static_mask_artifacts.fit_static_mask") as fit:
            with self.assertRaisesRegex(ValueError, "overlap"):
                pipeline.train_static_mask(self.cache, self.models, out, config=dict(k=2))
            fit.assert_not_called()
        self.assertFalse(out.exists())

    def test_cardinality_roundtrip_and_old_missing_key(self):
        out = self.root / "cardinality"
        pipeline.train_static_mask(self.cache, self.models, out,
                                   config=dict(k=2, epochs=2, relaxation="cardinality"))
        mask, report = load_static_mask(out, self.models, self.cache)
        self.assertEqual(mask.relaxation, "cardinality")
        self.assertTrue(report["backward_preserves_cardinality"])
        old = self.root / "legacy-missing-key"
        shutil.copytree(self.root / "2", old)
        r = read_json(old / "training.json")
        del r["config"]["relaxation"]
        (old / "training.json").write_text(json.dumps(r))
        receipt = read_json(old / "receipt.json")
        receipt["report_sha256"] = file_hash(old / "training.json")
        (old / "receipt.json").write_text(json.dumps(receipt))
        mask, _ = load_static_mask(old, self.models, self.cache)
        self.assertEqual(mask.relaxation, "sigmoid")


if __name__ == "__main__":
    unittest.main()

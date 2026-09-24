"""Lossless storage, fail-closed training and bounded-memory regression tests."""
import copy
import json
from pathlib import Path
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

import torch

from cbmjev.contracts import stable_hash
from cbmjev.crossfit_targets import (DiskRecords, TargetWriter, open_target_package,
                                     package_hash, records_hash)
from cbmjev.crossfit_training import (construct_action_targets,
                                     fit_controller_from_targets,
                                     _validate_target_package)
from cbmjev.learning import normalize_config
from tests_cbmjev import test_crossfit_training as fixtures

CONFIG = fixtures.CONFIG


class TargetStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NestedOOFTrainingPrimitiveTests.setUpClass()
        cls.fixture = fixtures.NestedOOFTrainingPrimitiveTests

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "targets.json"

    def produce(self):
        f = self.fixture
        return construct_action_targets(f.target_rows, f.head, f.schema, CONFIG,
            head_report=f.head_report, response_artifact_by_group=f.target_assignments,
            provenance_records=[f.outer_record], manifest_path=self.path)

    def test_exact_records_hashes_and_cpu_training(self):
        stored = self.produce()
        self.assertIsInstance(stored["records"], DiskRecords)
        materialized = {**stored, "records": list(stored["records"])}
        self.assertEqual(materialized, self.fixture.package)
        reopened = open_target_package(self.path)
        self.assertEqual(package_hash(reopened), self.fixture.package["package_sha256"])
        left, lr = fit_controller_from_targets(self.fixture.package, self.fixture.schema, CONFIG)
        right, rr = fit_controller_from_targets(reopened, self.fixture.schema, CONFIG)
        self.assertEqual(stable_hash(lr), stable_hash(rr))
        for key, tensor in left.network.state_dict().items():
            self.assertTrue(torch.equal(tensor, right.network.state_dict()[key]))

    def test_legacy_and_canonical_encoding(self):
        package = self.fixture.package
        self.path.write_text(json.dumps(package), encoding="utf-8")
        self.assertEqual(open_target_package(self.path), json.loads(json.dumps(package)))
        records = [{"id": "鸟é", "target": -0.0, "state": [-1, 2]},
                   {"id": "second", "target": 1e-20}]
        example = {"records": records, "unicode": "测试"}
        self.assertEqual(records_hash(iter(records)), stable_hash(records))
        self.assertEqual(package_hash(example), stable_hash(example))

    def test_tamper_rejected_before_optimizer(self):
        package = self.produce()
        shard = package["records"]._path(package["records"].shards[0])
        original = shard.read_bytes()
        for changed in (original[:-10], original + original.splitlines(keepends=True)[0],
                        original.replace(b'"target":', b'"extra":', 1)):
            shard.write_bytes(changed)
            with patch("torch.optim.AdamW") as optimizer:
                with self.assertRaises((ValueError, KeyError)):
                    fit_controller_from_targets(open_target_package(self.path), self.fixture.schema, CONFIG)
                optimizer.assert_not_called()
        shard.write_bytes(original)

    def test_cross_package_batches_remain_identical(self):
        f = self.fixture
        inline, stored = [], []
        for index, rows in enumerate((f.target_rows[:3], f.target_rows[3:])):
            options = dict(head_report=f.head_report,
                response_artifact_by_group={r["group_id"]: f.outer_responder_id for r in rows},
                provenance_records=[f.outer_record])
            inline.append(construct_action_targets(rows, f.head, f.schema, CONFIG, **options))
            stored.append(construct_action_targets(rows, f.head, f.schema, CONFIG,
                manifest_path=Path(self.tmp.name) / (str(index) + ".json"), **options))
        left, lr = fit_controller_from_targets(inline, f.schema, CONFIG)
        right, rr = fit_controller_from_targets(stored, f.schema, CONFIG)
        self.assertEqual(lr, rr)
        for key, tensor in left.network.state_dict().items():
            self.assertTrue(torch.equal(tensor, right.network.state_dict()[key]))

    def test_sharded_provenance_is_not_hash_only(self):
        self.produce()
        manifest = json.loads(self.path.read_text())
        metadata = manifest["metadata"]
        metadata["response_artifact_by_group"][metadata["target_group_ids"][0]] = "unknown"
        # Rehashing package bytes must not bypass source ancestry validation.
        candidate = {**metadata, "records": open_target_package(self.path)["records"]}
        metadata["package_sha256"] = package_hash(candidate)
        self.path.write_text(json.dumps(manifest))
        with patch("torch.optim.AdamW") as optimizer:
            with self.assertRaisesRegex(ValueError, "unknown artifact"):
                fit_controller_from_targets(open_target_package(self.path), self.fixture.schema, CONFIG)
            optimizer.assert_not_called()

    def test_paths_epochs_and_missing_shards(self):
        self.produce()
        manifest = json.loads(self.path.read_text())
        for name in ("../escape.jsonl", "/tmp/escape.jsonl"):
            changed = copy.deepcopy(manifest)
            changed["shards"][0]["path"] = name
            self.path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                open_target_package(self.path)
        self.path.write_text(json.dumps(manifest))
        source = open_target_package(self.path)
        shard = source["records"]._path(source["records"].shards[0])
        shard.unlink()
        with self.assertRaises(FileNotFoundError):
            _validate_target_package(source, self.fixture.schema,
                                     normalize_config(CONFIG, self.fixture.schema))

    def test_failure_does_not_publish_manifest_or_overwrite(self):
        with patch("cbmjev.crossfit_training.action_targets", side_effect=RuntimeError("failed")):
            with self.assertRaises(RuntimeError):
                self.produce()
        self.assertFalse(self.path.exists())
        with self.assertRaisesRegex(ValueError, "exists"):
            self.produce()

    def test_logical_validation_failure_prevents_manifest_publication(self):
        with patch("cbmjev.crossfit_training._validate_target_package",
                   side_effect=ValueError("logical validation failed")):
            with self.assertRaisesRegex(ValueError, "logical validation"):
                self.produce()
        self.assertFalse(self.path.exists())

    def test_duplicate_and_external_symlink_shards_rejected(self):
        source = self.produce()["records"]
        duplicate = copy.deepcopy(source.shards)
        duplicate[1]["path"] = duplicate[0]["path"]
        with self.assertRaises(ValueError):
            DiskRecords(source.root, duplicate)
        shard = source._path(source.shards[0])
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "outside.jsonl"
            target.write_bytes(shard.read_bytes())
            shard.unlink()
            shard.symlink_to(target)
            with self.assertRaises(ValueError):
                list(source)

    def test_streaming_memory_does_not_scale_with_event_count(self):
        def peak(count, name):
            tracemalloc.start()
            writer = TargetWriter(Path(self.tmp.name) / name)
            for i in range(count):
                writer.append({"epoch": 0, "observed": [-1] * 312, "target": i * 0.01})
            records = writer.records()
            records_hash(records)
            package_hash({"records": records})
            _, result = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return result
        small, large = peak(100, "small.json"), peak(10000, "large.json")
        self.assertLess(large, small + 2 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()

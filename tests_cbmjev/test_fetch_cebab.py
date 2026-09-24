"""Offline downloader tests: synthetic bytes, mocked Parquet, no network."""
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import fetch_cebab as fetch


def fake_reader(records, metadata_rows=None):
    class Reader:
        def __init__(self, path):
            self.metadata = SimpleNamespace(num_rows=len(records) if metadata_rows is None else metadata_rows)

        def iter_batches(self, batch_size):
            assert batch_size == 512
            yield SimpleNamespace(to_pylist=lambda: records)
    return Reader


class FetchCEBaBTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b"mock parquet content"
        self.spec = dict(split="test", filename="test.parquet", bytes=len(self.payload), rows=1,
                         sha256=hashlib.sha256(self.payload).hexdigest())
        self.records = [{"id": "00001", "original_id": "0", "is_original": True,
                         "description": "café 好", "food_distribution": "{'Positive': 3}",
                         "review_majority": "5", "empty": "", "missing": None}]

    def opener(self, request, timeout):
        self.assertEqual(timeout, 60)
        self.assertTrue(request.full_url.startswith(fetch.BASE_URL))
        self.assertFalse(request.has_header("Authorization"))
        return io.BytesIO(self.payload)

    def test_download_verifies_and_preserves_raw(self):
        target = fetch.download_verified(self.spec, self.root, self.opener)
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertEqual(fetch.file_hash(target), self.spec["sha256"])
        self.assertFalse(target.with_name(target.name + ".partial").exists())
        with self.assertRaises(FileExistsError):
            fetch.download_verified(self.spec, self.root, self.opener)

    def test_sha_mismatch(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            fetch.download_verified(dict(self.spec, sha256="0" * 64), self.root, self.opener)
        self.assertFalse((self.root / "test.parquet").exists())
        self.assertTrue((self.root / "test.parquet.partial").exists())

    def test_undersize_and_oversize(self):
        for adjustment in (-1, 1):
            directory = self.root / str(adjustment)
            directory.mkdir()
            with self.assertRaisesRegex(ValueError, "Byte count"):
                fetch.download_verified(dict(self.spec, bytes=len(self.payload) + adjustment), directory, self.opener)

    def test_lossless_conversion_and_hash(self):
        target = self.root / "test.jsonl"
        stats = fetch.convert_parquet("unused", target, 1, fake_reader(self.records))
        self.assertEqual(json.loads(target.read_text()), self.records[0])
        self.assertEqual(stats["sha256"], fetch.file_hash(target))
        self.assertEqual(stats["bytes"], target.stat().st_size)
        with self.assertRaises(FileExistsError):
            fetch.convert_parquet("unused", target, 1, fake_reader(self.records))

    def test_metadata_count_rejected_before_output(self):
        target = self.root / "test.jsonl"
        with self.assertRaisesRegex(ValueError, "metadata row count"):
            fetch.convert_parquet("unused", target, 2, fake_reader(self.records))
        self.assertFalse(target.exists())

    def test_actual_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Decoded row count mismatch"):
            fetch.convert_parquet("unused", self.root / "test.jsonl", 2, fake_reader(self.records, metadata_rows=2))

    def test_three_files_and_receipt(self):
        manifest = tuple(dict(self.spec, split=name, filename=name + ".parquet")
                         for name in ("train_exclusive", "validation", "test"))
        out = self.root / "dataset"
        with patch.object(fetch, "MANIFEST", manifest), patch.object(fetch, "urlopen", side_effect=self.opener) as opener:
            receipt = fetch.fetch_cebab(out, reader=fake_reader(self.records))
        self.assertEqual(opener.call_count, 3)
        self.assertEqual(receipt["total_rows"], 3)
        self.assertEqual(receipt["revision"], fetch.REVISION)
        self.assertIn("NOT_VERIFIED", receipt["hf_snapshot_equivalence_to_zip_v1_1"])
        self.assertIn("CC BY-NC", receipt["license"]["caveat"])
        self.assertEqual(len(list((out / "raw").glob("*.parquet"))), 3)
        self.assertEqual(len(list(out.glob("*.jsonl"))), 3)
        self.assertEqual(json.loads((out / "source_receipt.json").read_text()), receipt)

    def test_nonempty_output_refused_without_network(self):
        sentinel = self.root / "user_file"
        sentinel.write_text("preserve me")
        with patch.object(fetch, "urlopen") as opener:
            with self.assertRaises(FileExistsError):
                fetch.fetch_cebab(self.root, reader=fake_reader(self.records))
            opener.assert_not_called()
        self.assertEqual(sentinel.read_text(), "preserve me")

    def test_missing_dependency_before_download_or_directory(self):
        out = self.root / "new"
        with patch.object(fetch, "parquet_reader", side_effect=RuntimeError("requires pyarrow")), patch.object(fetch, "urlopen") as opener:
            with self.assertRaisesRegex(RuntimeError, "pyarrow"):
                fetch.fetch_cebab(out)
            opener.assert_not_called()
        self.assertFalse(out.exists())

    def test_locked_manifest_invariants(self):
        self.assertEqual(sum(x["rows"] for x in fetch.MANIFEST), 5117)
        self.assertEqual(sum(x["bytes"] for x in fetch.MANIFEST), 846457)
        self.assertEqual([x["split"] for x in fetch.MANIFEST], ["train_exclusive", "validation", "test"])

    def test_inclusive_stream_replaces_train_and_reuses_verified_heldout(self):
        manifest = tuple(dict(self.spec, split=name, filename=name + ".parquet")
                         for name in ("train_exclusive", "validation", "test"))
        inclusive = dict(self.spec, split="train_inclusive", filename="train_inclusive.parquet")
        local = self.root / "local"
        local.mkdir()
        for spec in manifest[1:]:
            (local / spec["filename"]).write_bytes(self.payload)
        with patch.object(fetch, "MANIFEST", manifest), patch.object(fetch, "INCLUSIVE", inclusive), patch.object(fetch, "urlopen") as opener:
            receipt = fetch.fetch_cebab(self.root / "new", reader=fake_reader(self.records),
                                       train_variant="train_inclusive", inclusive_bytes=self.payload, local_raw=local)
        opener.assert_not_called()
        self.assertEqual([a["split"] for a in receipt["artifacts"]], ["train_inclusive", "validation", "test"])
        self.assertFalse((self.root / "new/train_exclusive.jsonl").exists())
        self.assertEqual(receipt["artifacts"][0]["acquisition"], "verified_stream_relay")

    def test_inclusive_bad_stream_rejected(self):
        with self.assertRaisesRegex(ValueError, "pinned SHA256/bytes"):
            fetch.fetch_cebab(self.root / "new", reader=fake_reader(self.records),
                              train_variant="train_inclusive", inclusive_bytes=b"corrupt")


if __name__ == "__main__":
    unittest.main()

"""Small offline fixtures only; no official CUB data or scientific results."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools import prepare_cub_official as cub


def tar_bytes(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, kind, value in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.REGTYPE:
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
            else:
                member.linkname = value
                archive.addfile(member)
    return buffer.getvalue()


class OfficialCUBPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_hashes_and_size(self):
        payload = b"fixture"
        result = cub.verify_stream(io.BytesIO(payload), len(payload), hashlib.md5(payload).hexdigest())
        self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())
        for size in (len(payload) - 1, len(payload) + 1):
            with self.assertRaisesRegex(ValueError, "byte count"):
                cub.verify_stream(io.BytesIO(payload), size, "0" * 32)
        with self.assertRaisesRegex(ValueError, "MD5"):
            cub.verify_stream(io.BytesIO(payload), len(payload), "0" * 32)

    def test_normal_files_and_directories(self):
        payload = tar_bytes([("CUB_200_2011", tarfile.DIRTYPE, ""),
                             ("CUB_200_2011/images/a.jpg", tarfile.REGTYPE, b"fixture"),
                             ("attributes.txt", tarfile.REGTYPE, b"1 color::red\n")])
        result = cub.safe_extract_verified(io.BytesIO(payload), self.root / "out")
        self.assertEqual(result["regular_files"], 2)
        self.assertEqual((self.root / "out/attributes.txt").read_bytes(), b"1 color::red\n")

    def test_unsafe_paths_rejected_before_output(self):
        for index, name in enumerate(("/absolute", "../escape", "a/../../escape", "C:/drive", "a\\escape")):
            with self.subTest(name=name):
                payload = tar_bytes([("valid", tarfile.REGTYPE, b"ok"), (name, tarfile.REGTYPE, b"bad")])
                out = self.root / str(index)
                with self.assertRaisesRegex(ValueError, "unsafe archive path"):
                    cub.safe_extract_verified(io.BytesIO(payload), out)
                self.assertFalse(out.exists())

    def test_links_and_special_members_rejected(self):
        for index, kind in enumerate((tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE)):
            with self.subTest(kind=kind):
                payload = tar_bytes([("special", kind, "/outside")])
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    cub.safe_extract_verified(io.BytesIO(payload), self.root / str(index))

    def test_duplicate_and_file_parent_rejected(self):
        for index, names in enumerate((("a", "a"), ("a", "a/b"), ("a/b", "a"))):
            payload = tar_bytes([(name, tarfile.REGTYPE, b"x") for name in names])
            with self.assertRaises(ValueError):
                cub.safe_extract_verified(io.BytesIO(payload), self.root / str(index))

    def test_existing_output_refused(self):
        out = self.root / "out"
        out.mkdir()
        with self.assertRaises(FileExistsError):
            cub.safe_extract_verified(io.BytesIO(b"not a tar"), out)
        target = self.root / "link"
        target.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            cub.safe_extract_verified(io.BytesIO(b"not a tar"), target)

    def test_bad_official_archive_creates_no_outputs(self):
        archive = self.root / "bad.tgz"
        archive.write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "byte count"):
            cub.prepare_cub(archive, self.root / "out", self.root / "prepared")
        self.assertFalse((self.root / "out").exists())
        self.assertFalse((self.root / "prepared").exists())

    def test_nested_outputs_rejected(self):
        archive = self.root / "bad.tgz"
        archive.write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            cub.prepare_cub(archive, self.root / "out", self.root / "out/prepared")

    def test_resource_limits(self):
        payload = tar_bytes([("a", tarfile.REGTYPE, b"xx")])
        with patch.object(cub, "MAX_EXTRACTED_BYTES", 1), self.assertRaisesRegex(ValueError, "byte limit"):
            cub.safe_extract_verified(io.BytesIO(payload), self.root / "out")
        with patch.object(cub, "MAX_MEMBERS", 0), self.assertRaisesRegex(ValueError, "member limit"):
            cub.safe_extract_verified(io.BytesIO(payload), self.root / "out")

    def test_real_adapter_on_tiny_fixture_with_patched_official_fingerprint(self):
        from tests_cbmjev.test_data import cub_fixture
        raw = self.root / "fixture/CUB_200_2011"
        cub_fixture(raw)
        (raw / "attributes/attributes.txt").rename(raw.parent / "attributes.txt")
        archive = self.root / "fixture.tgz"
        with tarfile.open(archive, "w:gz") as handle:
            for path in sorted(raw.parent.rglob("*")):
                handle.add(path, arcname=path.relative_to(raw.parent), recursive=False)
        payload = archive.read_bytes()
        with patch.object(cub, "OFFICIAL_BYTES", len(payload)), patch.object(cub, "OFFICIAL_MD5", hashlib.md5(payload).hexdigest()):
            result = cub.prepare_cub(archive, self.root / "out", self.root / "prepared", seed=23)
        self.assertEqual(result["status"], "PREPARED_NOT_ACCEPTED")
        self.assertEqual(result["source_counts"], {"train": 22, "test": 2})
        receipt = json.loads((self.root / "out/source_receipt.json").read_text())
        self.assertEqual(receipt["verified_archive"]["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(receipt["attributes_file"], str((self.root / "out/attributes.txt").resolve()))
        self.assertEqual(json.loads((self.root / "prepared/audit.json").read_text())["seed"], 23)
        self.assertEqual(result["scientific_acceptance"], "NOT_EVALUATED")


if __name__ == "__main__":
    unittest.main()

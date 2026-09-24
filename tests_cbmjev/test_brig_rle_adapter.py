import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, write_json
from scripts import train_evaluate_cub_brig_rle_crossfit as rle


class BRiGRLEAdapterTests(unittest.TestCase):
    def test_fold_sidecar_binds_all_sources_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in ("receipt.json", "fast_variant_receipt.json",
                         "training.json", "brig.pt"):
                (directory / name).write_bytes(name.encode())
            receipt = {
                "format": "cbmjev-brig-rle-fold-sidecar-v1", "status": "COMPLETE",
                "sources_sha256": rle.source_hashes(),
                "base_receipt_sha256": file_hash(directory / "receipt.json"),
                "fast_sidecar_sha256": file_hash(directory / "fast_variant_receipt.json"),
                "checkpoint_sha256": file_hash(directory / "brig.pt"),
                "training_sha256": file_hash(directory / "training.json")}
            receipt["receipt_hash"] = stable_hash(receipt)
            write_json(directory / "rle_variant_receipt.json", receipt)
            with patch.object(rle.fast, "verify_fast_fold", return_value={}):
                self.assertEqual(rle.verify_rle_fold(directory), receipt)
                (directory / "brig.pt").write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "RLE fold sidecar"):
                    rle.verify_rle_fold(directory)


if __name__ == "__main__":
    unittest.main()

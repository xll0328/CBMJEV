#!/usr/bin/env python3
"""Source-bound run-length acceleration of the existing BRiG-fast adapter.

The fast trainer and the original crossfit adapter are left untouched while
their live runs train. This wrapper changes only GroupQ's repeated-state
encoding, which has exact forward/gradient and tiny-fit parity tests. The
ordinary and fast receipts remain valid; an additional sidecar identifies
the RLE implementation used to produce a checkpoint or evaluation.
"""

from pathlib import Path
from unittest.mock import patch

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, read_json, write_json
from scripts import brig_fast_fit, train_evaluate_cub_brig_fast_crossfit as fast
from scripts.brig_q_rle import GroupQRunLength


SOURCE_NAMES = ("train_evaluate_cub_brig_rle_crossfit.py", "brig_q_rle.py")


def source_hashes():
    root = Path(__file__).parent
    return {**fast.source_hashes(), **{name: file_hash(root / name) for name in SOURCE_NAMES}}


def _sidecar(directory):
    return Path(directory) / "rle_variant_receipt.json"


def verify_rle_fold(directory):
    directory = Path(directory)
    receipt = read_json(_sidecar(directory))
    unsigned = dict(receipt)
    if (unsigned.pop("receipt_hash", None) != stable_hash(unsigned)
            or receipt.get("format") != "cbmjev-brig-rle-fold-sidecar-v1"
            or receipt.get("status") != "COMPLETE"
            or receipt.get("sources_sha256") != source_hashes()
            or receipt.get("base_receipt_sha256") != file_hash(directory / "receipt.json")
            or receipt.get("fast_sidecar_sha256") != file_hash(directory / "fast_variant_receipt.json")
            or receipt.get("checkpoint_sha256") != file_hash(directory / "brig.pt")
            or receipt.get("training_sha256") != file_hash(directory / "training.json")):
        raise ValueError("BRiG RLE fold sidecar/source/checkpoint mismatch")
    fast.verify_fast_fold(directory)
    return receipt


def main(argv=None):
    args = fast.base.parse_args(argv)
    before = source_hashes()
    fold_sidecars = None
    if args.command == "evaluate":
        fold_sidecars = [verify_rle_fold(path) for path in args.fold_models]
    # The fitter creates budget-specific GroupQ instances through this module
    # global. Previous-budget rollout models consequently use RLE as well.
    with patch.object(brig_fast_fit, "GroupQ", GroupQRunLength):
        fast.main(argv)
    if source_hashes() != before:
        raise ValueError("BRiG RLE source changed during run")
    directory = Path(args.out)
    if args.command == "train-fold":
        receipt = {"format": "cbmjev-brig-rle-fold-sidecar-v1", "status": "COMPLETE",
            "outer_fold": args.outer_fold,
            "algorithm": "same-BRiG-Q-objective-run-length-state-encoding",
            "sources_sha256": before,
            "base_receipt_sha256": file_hash(directory / "receipt.json"),
            "fast_sidecar_sha256": file_hash(directory / "fast_variant_receipt.json"),
            "checkpoint_sha256": file_hash(directory / "brig.pt"),
            "training_sha256": file_hash(directory / "training.json")}
        receipt["receipt_hash"] = stable_hash(receipt)
        write_json(_sidecar(directory), receipt)
        verify_rle_fold(directory)
    else:
        if fold_sidecars != [verify_rle_fold(path) for path in args.fold_models]:
            raise ValueError("BRiG RLE folds changed during evaluation")
        receipt = {"format": "cbmjev-brig-rle-eval-sidecar-v1", "status": "COMPLETE",
            "sources_sha256": before,
            "fold_sidecars_sha256": [file_hash(_sidecar(path)) for path in args.fold_models],
            "base_receipt_sha256": file_hash(directory / "receipt.json"),
            "fast_sidecar_sha256": file_hash(directory / "fast_variant_receipt.json")}
        receipt["receipt_hash"] = stable_hash(receipt)
        write_json(_sidecar(directory), receipt)


if __name__ == "__main__":
    main()

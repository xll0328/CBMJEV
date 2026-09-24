#!/usr/bin/env python3
"""Provenance-bound GroupQ encoding optimization for the CUB BRiG adapter."""

from pathlib import Path

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, read_json, write_json
from scripts import train_evaluate_cub_brig_fast_crossfit as fast
from scripts.brig_accelerated_ops import accelerated_groupq_forward


def source_hashes():
    project = Path(fast.base.__file__).resolve().parent.parent
    return {
        "adapter": file_hash(Path(__file__)),
        "accelerated_ops": file_hash(Path(__file__).with_name("brig_accelerated_ops.py")),
        "fast_adapter": file_hash(Path(fast.__file__)),
        "fast_sources": fast.source_hashes(),
        "brig_core": file_hash(project / "cbmjev" / "brig.py"),
        "learning_core": file_hash(project / "cbmjev" / "learning.py"),
    }


def _sidecar(directory):
    return Path(directory) / "accelerated_variant_receipt.json"


def verify_accelerated_fold(directory):
    directory = Path(directory)
    receipt = read_json(_sidecar(directory))
    unsigned = dict(receipt)
    if (unsigned.pop("receipt_hash", None) != stable_hash(unsigned)
            or receipt.get("format") != "cbmjev-brig-unique-state-fold-v1"
            or receipt.get("status") != "COMPLETE"
            or receipt.get("source_hashes") != source_hashes()
            or receipt.get("base_receipt_sha256") != file_hash(directory / "receipt.json")
            or receipt.get("fast_sidecar_sha256") != file_hash(directory / "fast_variant_receipt.json")
            or receipt.get("checkpoint_sha256") != file_hash(directory / "brig.pt")):
        raise ValueError("accelerated BRiG fold sidecar/source/checkpoint mismatch")
    return receipt


def _write_receipt(directory, command, source_binding, fold_bindings=()):
    directory = Path(directory)
    receipt = {
        "format": "cbmjev-brig-unique-state-fold-v1" if command == "train-fold"
                  else "cbmjev-brig-unique-state-eval-v1",
        "status": "COMPLETE",
        "algorithm": "same-BRiG-objective-deduplicated-validated-state-encoding",
        "source_hashes": source_binding,
        "base_receipt_sha256": file_hash(directory / "receipt.json"),
        "fast_sidecar_sha256": file_hash(directory / "fast_variant_receipt.json")
                               if command == "train-fold" else None,
        "checkpoint_sha256": file_hash(directory / "brig.pt") if command == "train-fold" else None,
        "fold_sidecars_sha256": list(fold_bindings) if command != "train-fold" else None,
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    write_json(_sidecar(directory), receipt)
    return receipt


def main(argv=None):
    args = fast.base.parse_args(argv)
    before = source_hashes()
    fast.base.GroupQ.forward = accelerated_groupq_forward

    fold_bindings = []
    if args.command == "evaluate":
        fold_bindings = [verify_accelerated_fold(path) for path in args.fold_models]

    fast.main(argv)

    if source_hashes() != before:
        raise ValueError("accelerated BRiG sources changed during execution")
    if args.command == "train-fold":
        _write_receipt(args.out, args.command, before)
        verify_accelerated_fold(args.out)
    else:
        _write_receipt(args.out, args.command, before,
                       [file_hash(_sidecar(path)) for path in args.fold_models])


if __name__ == "__main__":
    main()

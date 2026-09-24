#!/usr/bin/env python3
"""Crossfit BRiG adapter with cached rollout targets and unique-state Q."""

from pathlib import Path

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, read_json, write_json
from scripts import train_evaluate_cub_brig_fast_crossfit as fast
from scripts.brig_accelerated_ops import accelerated_groupq_forward
from scripts.brig_cached_fit import fit_brig_cached


def source_hashes():
    root = Path(__file__).parent
    project = Path(fast.base.__file__).resolve().parent.parent
    names = ("train_evaluate_cub_brig_cached_crossfit.py",
             "brig_cached_fit.py", "brig_cached_rollout_targets.py",
             "brig_batched_rollout.py", "brig_accelerated_ops.py",
             "train_evaluate_cub_brig_fast_crossfit.py",
             "train_evaluate_cub_brig_crossfit.py")
    return {**{name: file_hash(root / name) for name in names},
            "brig_core": file_hash(project / "cbmjev" / "brig.py"),
            "learning_core": file_hash(project / "cbmjev" / "learning.py")}


def _sidecar(directory):
    return Path(directory) / "cached_variant_receipt.json"


def verify_cached_fold(directory):
    directory = Path(directory)
    receipt = read_json(_sidecar(directory))
    unsigned = dict(receipt)
    if (unsigned.pop("receipt_hash", None) != stable_hash(unsigned)
            or receipt.get("format") != "cbmjev-brig-cached-rollout-fold-v1"
            or receipt.get("status") != "COMPLETE"
            or receipt.get("source_hashes") != source_hashes()
            or receipt.get("base_receipt_sha256") != file_hash(directory / "receipt.json")
            or receipt.get("checkpoint_sha256") != file_hash(directory / "brig.pt")):
        raise ValueError("cached-rollout BRiG fold receipt/source/checkpoint mismatch")
    return receipt


def _write_sidecar(directory, *, command, source_binding, fold_bindings=()):
    directory = Path(directory)
    receipt = {
        "format": "cbmjev-brig-cached-rollout-fold-v1" if command == "train-fold"
                  else "cbmjev-brig-cached-rollout-eval-v1",
        "status": "COMPLETE",
        "algorithm": "exact-cached-terminal-risk-plus-unique-state-encoding",
        "source_hashes": source_binding,
        "base_receipt_sha256": file_hash(directory / "receipt.json"),
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
    fast.base.fit_brig = fit_brig_cached

    if args.command == "train-fold":
        result = fast.base.train_fold(args)
        if source_hashes() != before:
            raise ValueError("cached BRiG sources changed during training")
        _write_sidecar(args.out, command=args.command, source_binding=before)
        verify_cached_fold(args.out)
    else:
        bindings = [verify_cached_fold(path) for path in args.fold_models]
        result = fast.base.evaluate(args)
        if (source_hashes() != before or bindings != [verify_cached_fold(path)
                                                       for path in args.fold_models]):
            raise ValueError("cached BRiG sources or fold receipts changed during evaluation")
        _write_sidecar(args.out, command=args.command, source_binding=before,
                       fold_bindings=[file_hash(_sidecar(path)) for path in args.fold_models])
    print(result, flush=True)


if __name__ == "__main__":
    main()

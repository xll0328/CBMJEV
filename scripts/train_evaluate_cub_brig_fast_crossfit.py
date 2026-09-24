#!/usr/bin/env python3
"""Provenance-bound batched-rollout variant of the CUB BRiG crossfit adapter.

The original adapter is deliberately unchanged while its serial jobs are live.
Its ordinary receipt is supplemented with a fast-variant sidecar that binds
the monkeypatched fitter source; only the pair of receipts identifies this run.
"""

from pathlib import Path

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, read_json, write_json
from scripts import train_evaluate_cub_brig_crossfit as base
from scripts.brig_fast_fit import fit_brig_fast


SOURCE_NAMES = ("train_evaluate_cub_brig_fast_crossfit.py", "brig_fast_fit.py",
                "brig_batched_rollout.py", "train_evaluate_cub_brig_crossfit.py")


def source_hashes():
    root = Path(__file__).parent
    return {name: file_hash(root / name) for name in SOURCE_NAMES}


def _sidecar_path(directory):
    return Path(directory) / "fast_variant_receipt.json"


def verify_fast_fold(directory):
    path = _sidecar_path(directory)
    receipt = read_json(path)
    unsigned = dict(receipt)
    if (unsigned.pop("receipt_hash", None) != stable_hash(unsigned)
            or receipt.get("format") != "cbmjev-brig-batched-rollout-fold-sidecar-v1"
            or receipt.get("status") != "COMPLETE"
            or receipt.get("sources_sha256") != source_hashes()
            or receipt.get("base_receipt_sha256") != file_hash(Path(directory) / "receipt.json")
            or receipt.get("checkpoint_sha256") != file_hash(Path(directory) / "brig.pt")):
        raise ValueError("BRiG fast fold sidecar/source/checkpoint mismatch")
    return receipt


def main(argv=None):
    args = base.parse_args(argv)
    before = source_hashes()
    if args.command == "train-fold":
        # Original adapter performs the same exclusion, source, model and
        # checkpoint checks. Only its low-level trainer is substituted.
        base.fit_brig = fit_brig_fast
        result = base.train_fold(args)
        if source_hashes() != before:
            raise ValueError("BRiG fast source changed during training")
        directory = Path(args.out)
        sidecar = {"format": "cbmjev-brig-batched-rollout-fold-sidecar-v1",
            "status": "COMPLETE", "outer_fold": args.outer_fold,
            "algorithm": "same-BRiG-Q-objective-batched-empty-rollout-targets",
            "sources_sha256": before,
            "base_receipt_sha256": file_hash(directory / "receipt.json"),
            "checkpoint_sha256": file_hash(directory / "brig.pt")}
        sidecar["receipt_hash"] = stable_hash(sidecar)
        write_json(_sidecar_path(directory), sidecar)
        verify_fast_fold(directory)
    else:
        fold_bindings = [verify_fast_fold(path) for path in args.fold_models]
        result = base.evaluate(args)
        if source_hashes() != before or fold_bindings != [verify_fast_fold(path)
                                                        for path in args.fold_models]:
            raise ValueError("BRiG fast source or fold sidecar changed during evaluation")
        directory = Path(args.out)
        sidecar = {"format": "cbmjev-brig-batched-rollout-eval-sidecar-v1",
            "status": "COMPLETE", "sources_sha256": before,
            "fold_sidecars_sha256": [file_hash(_sidecar_path(path)) for path in args.fold_models],
            "base_receipt_sha256": file_hash(directory / "receipt.json")}
        sidecar["receipt_hash"] = stable_hash(sidecar)
        write_json(_sidecar_path(directory), sidecar)
    print(result, flush=True)


if __name__ == "__main__":
    main()

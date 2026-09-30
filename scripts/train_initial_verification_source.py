#!/usr/bin/env python3
"""Train/predict source-only DistilRoBERTa native concepts on the data server.

Loads an existing local snapshot; never downloads models or data. Training uses
only responder_fit concepts, with an internal group split shared by typed B.
Test and confirmation prediction require an explicit frozen-evaluation override.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbmjev.initial_verifier import (MODEL_REVISION, TRAINING_SEEDS, predict_initial_source,
                                    train_initial_source)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="select on source groups and refit all responder_fit concepts")
    train.add_argument("--prepared", required=True)
    train.add_argument("--output", required=True, help="new private runtime directory")
    train.add_argument("--backbone", required=True, help="existing pinned local DistilRoBERTa directory")
    train.add_argument("--model-revision", default=MODEL_REVISION)
    train.add_argument("--seed", type=int, choices=TRAINING_SEEDS, default=40)
    train.add_argument("--split-seed", type=int, default=20260930)
    train.add_argument("--epochs", type=int, default=5, help="at most 5; lower counts are engineering pilots")
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--device", default="cuda:0")
    predict = commands.add_parser("predict", help="emit role-filtered IDs, hard A, and genuine probabilities")
    predict.add_argument("--prepared", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--output", required=True, help="new private JSONL; metadata written alongside")
    predict.add_argument("--roles", nargs="+", default=["head_fit", "policy_fit", "policy_tune"])
    predict.add_argument("--allow-protected-roles", action="store_true",
                         help="explicit test/confirmation inference override; never allows source fitting")
    predict.add_argument("--batch-size", type=int, default=16)
    predict.add_argument("--device", default="cuda:0")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "train":
        metadata, report = train_initial_source(args.prepared, args.output, args.backbone,
            model_revision=args.model_revision, seed=args.seed, epochs=args.epochs,
            batch_size=args.batch_size, split_seed=args.split_seed, device=args.device)
        selected = report["selected"]
        result = {"checkpoint": str(Path(args.output)), "seed": args.seed,
                  "fit_n": len(metadata["fit_sample_ids"]),
                  "selected": {"learning_rate": selected["learning_rate"], "epochs": selected["epoch"],
                               "concept_macro_f1": selected["metrics"]["concept_macro_f1"],
                               "n_samples": selected["metrics"]["n_samples"]},
                  "checkpoint_identity": metadata["checkpoint_identity"]}
    else:
        report = predict_initial_source(args.prepared, args.checkpoint, args.output, roles=args.roles,
            allow_protected=args.allow_protected_roles, batch_size=args.batch_size, device=args.device)
        result = {"checkpoint": str(Path(args.checkpoint)), "predictions": str(Path(args.output)),
                  "n": report["n"], "seed": report["seed"],
                  "predictions_sha256": report["predictions_sha256"]}
    result["truncation"] = {key: report["truncation"][key] for key in
                            ("n_samples", "max_length", "truncated_samples", "truncation_rate")}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

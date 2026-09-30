"""Train concept-only typed verification and the fair shared-encoder reference.

All transformer loads are local-only. Run extraction on the server, then fit
small readouts from cached frozen features. Cache replay is never reported as
end-to-end deployment latency. Source quality is selected on a source-internal
group split; protected test/confirmation roles are rejected by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cbmjev.io import write_json
from cbmjev.typed_verifier import (audit_source_conditions, extract_feature_cache,
    load_frozen_encoder, load_source_records, predict_cached_sources, train_cached_sources)


def _model_arguments(parser):
    parser.add_argument("--backbone", required=True, help="existing local Qwen3-4B directory; no download")
    parser.add_argument("--model-revision", required=True, help="immutable Hugging Face commit SHA")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16, help="candidate/text paths per backbone forward")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="cache both frozen feature sources in bounded shards")
    extract.add_argument("--prepared", required=True)
    extract.add_argument("--roles", nargs="+", default=["responder_fit"])
    extract.add_argument("--allow-protected-roles", action="store_true",
                         help="explicit confirmation/test inference override; NEVER permits fitting")
    extract.add_argument("--cache", required=True)
    extract.add_argument("--shard-size", type=int, default=32, help="samples per server-side feature shard")
    extract.add_argument("--padding-side", choices=("left", "right"), default="right")
    extract.add_argument("--max-cache-gib", type=float, default=64.)
    _model_arguments(extract)
    train = commands.add_parser("train", help="freeze SOURCE_GATE, select readouts, refit all source groups")
    train.add_argument("--cache", required=True, help="ONLY source/responder_fit cache rows allowed")
    train.add_argument("--output", required=True)
    train.add_argument("--seed", type=int, default=40)
    train.add_argument("--split-seed", type=int, default=20260930)
    train.add_argument("--epochs", type=int, default=10, help="at most 10; lower counts are engineering pilots")
    train.add_argument("--batch-size", type=int, default=64, help="samples per cached readout optimizer step")
    predict = commands.add_parser("predict", help="emit genuine typed/shared probabilities from frozen readouts")
    predict.add_argument("--cache", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--output", required=True, help="JSONL; metadata written alongside")
    predict.add_argument("--allow-protected-roles", action="store_true",
                         help="explicit confirmation/test inference override; NEVER permits fitting")
    audit = commands.add_parser("audit", help="fixed source-tune batch/padding/candidate-order consistency")
    audit.add_argument("--prepared", required=True)
    audit.add_argument("--source-role", default="responder_fit", choices=("responder_fit", "source_fit", "source-fit"))
    audit.add_argument("--checkpoint", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--limit", type=int, default=100)
    _model_arguments(audit)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "extract":
        if args.max_cache_gib <= 0:
            raise ValueError("max-cache-gib must be positive")
        schema, records, provenance = load_source_records(args.prepared, args.roles,
            allow_protected=args.allow_protected_roles)
        encoder = load_frozen_encoder(args.backbone, schema, model_revision=args.model_revision,
                                      device=args.device, max_length=args.max_length)
        index = extract_feature_cache(encoder, records, args.cache, provenance=provenance,
            batch_size=args.batch_size, shard_size=args.shard_size, padding_side=args.padding_side,
            max_cache_bytes=int(args.max_cache_gib * 1024**3),
            progress=lambda value: print(json.dumps(value), flush=True))
        result = {"cache": str(Path(args.cache)), "n": len(index["rows"]),
                  "estimated_feature_bytes": index["estimated_feature_bytes"],
                  "index_sha256": index["index_sha256"], "binding": index["binding"]}
    elif args.command == "train":
        metadata, report = train_cached_sources(args.cache, args.output, seed=args.seed,
            split_seed=args.split_seed, epochs=args.epochs, batch_size=args.batch_size)
        result = {"checkpoint": str(Path(args.output)), "fit_n": len(metadata["fit_sample_ids"]),
                  "seed": args.seed, "selected": {kind: {
                      "learning_rate": value["selected"]["learning_rate"],
                      "epochs": value["selected"]["epoch"],
                      "tune_concept_macro_f1": value["selected"]["metrics"]["concept_macro_f1"]}
                      for kind, value in report.items()},
                  "source_gate": {key: report["typed"]["gate"][key] for key in
                                  ("upgrade_triggered", "macro_trigger", "collapse_trigger")},
                  "backbone_forwards_during_training": 0}
    elif args.command == "predict":
        report = predict_cached_sources(args.cache, args.checkpoint, args.output,
                                         allow_protected=args.allow_protected_roles)
        result = {"output": str(Path(args.output)), "n": report["n"],
                  "fit_n": len(report["fit_sample_ids"]), **{key: report[key] for key in (
                      "source_checkpoint_sha256", "readout_weights_sha256", "predictions_sha256",
                      "cached_readout_replay_seconds", "timing_scope")}}
    else:
        if args.limit < 1:
            raise ValueError("audit limit must be positive")
        output = Path(args.output)
        if output.exists():
            raise FileExistsError("audit output already exists")
        schema, records, _ = load_source_records(args.prepared, (args.source_role,))
        encoder = load_frozen_encoder(args.backbone, schema, model_revision=args.model_revision,
                                      device=args.device, max_length=args.max_length)
        report = audit_source_conditions(encoder, records, args.checkpoint,
                                         batch_size=args.batch_size, limit=args.limit)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
        result = {"audit": str(output), "n": report["n"], "passed": report["passed"],
                  "readout_weights_sha256": report["readout_weights_sha256"],
                  "conditions": [{key: condition[key] for key in
                                  ("condition", "changed_hard_answers", "max_probability_delta")}
                                 for condition in report["conditions"]]}
        if not result["passed"]:
            print(json.dumps(result, ensure_ascii=False))
            raise SystemExit("source condition audit failed; preserve the report and investigate")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

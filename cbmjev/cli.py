"""Unified CBMJev command line. Defaults never evaluate test or call remote APIs."""
import argparse
import json
from pathlib import Path
import sys

from .io import read_json, read_jsonl, write_json, fresh_dir, environment


def parser():
    main = argparse.ArgumentParser(prog="cbmjev", description=__doc__)
    commands = main.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train-choice-pair", help="development scalar/set heads on visible structured features")
    p.add_argument("--prepared", required=True)
    p.add_argument("--targets", nargs="+", required=True)
    p.add_argument("--config", required=True, help="original target-generation configuration")
    p.add_argument("--out", required=True)
    p.add_argument("--max-questions", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--seed", type=int, default=60)
    p.add_argument("--cost-weight", type=float, default=.03)
    p.add_argument("--temperature", type=float, default=.5)
    p.add_argument("--remaining-groups", type=int)
    p.add_argument("--budget-mode", choices=("fixed", "uniform_remaining"), default="fixed")
    p.add_argument("--capacity-control", action="store_true",
                   help="also fit a parameter-matched candidate-independent MLP head")

    p = commands.add_parser("evaluate-choice-crossfit", help="paired structured Choice validation on bound nested OOF artifacts")
    for name in ("prepared", "planned", "merged", "responder", "cache", "choice", "target-config", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--targets", nargs="+", required=True)
    p.add_argument("--cost-weight", type=float)
    p.add_argument("--max-groups", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--responder-source-dir",
                   help="read-only original release root for historical completed responder caches")
    p = commands.add_parser("doctor", help="inspect environment; optional small device forward")
    p.add_argument("--out")
    p.add_argument("--check-device", help="cpu or cuda:index; does not download models")

    p = commands.add_parser("prepare", help="convert already-downloaded official-format data")
    p.add_argument("--dataset", required=True, choices=("cebab", "cub", "derm7pt"))
    p.add_argument("--source", required=True)
    p.add_argument("--source-revision", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=17)

    responder_options = argparse.ArgumentParser(add_help=False)
    p = responder_options
    p.add_argument("--prepared", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--kind", choices=("hashing_text", "hf_text", "resnet18", "nano_semantic"), default="hashing_text")
    p.add_argument("--raw-root")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, help="default 3e-5 for hf_text, 0.003 otherwise")
    p.add_argument("--backbone", help="explicit HF model source for hf_text; local-only directory/state_dict for other kinds")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--revision", help="hf_text: requested Hub revision; resolved commit is recorded and tokenizer-pinned")
    p.add_argument("--allow-download", action="store_true", help="hf_text only: explicitly permit loading the named Hub model")
    p.add_argument("--max-length", type=int, default=512, help="hf_text token limit; overlength input fails, never silently truncates")
    p.add_argument("--freeze-backbone", action="store_true", help="hf_text: train concept heads only, keep encoder in eval mode")
    p.add_argument("--input-workers", type=int, default=0,
                   help="bounded content-IO threads; zero preserves synchronous loading")
    p.add_argument("--input-prefetch-batches", type=int, default=2)
    p.add_argument("--concept-class-weighting", choices=("none", "inverse_frequency"), default="none",
                   help="resnet18/hf_text: class weights fitted only on responder training concept labels")

    commands.add_parser("train-responder", parents=[responder_options],
                        help="fit semantics only on responder_fit")
    p = commands.add_parser("train-crossfit-responder", parents=[responder_options],
                            help="fit the exact groups of a verified nested fold plan")
    p.add_argument("--planned", required=True)
    stage = p.add_mutually_exclusive_group(required=True)
    stage.add_argument("--outer-fold", type=int)
    stage.add_argument("--final", action="store_true")
    p.add_argument("--inner-fold", type=int)

    p = commands.add_parser("cache-crossfit", help="verified fold-only automatic responses; no test access")
    p.add_argument("--prepared", required=True)
    p.add_argument("--planned", required=True)
    p.add_argument("--responder", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--raw-root")
    p.add_argument("--device", default="cpu")
    p.add_argument("--split", choices=("validation", "calibration"),
                   help="required only for the final responder")

    p = commands.add_parser("cache", help="produce full offline automatic responses, not latency evidence")
    p.add_argument("--prepared", required=True)
    p.add_argument("--responder", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--raw-root")
    p.add_argument("--device", default="cpu")

    p = commands.add_parser("train", help="fit f on head_fit and risk/value on policy_fit")
    p.add_argument("--cache", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--device")

    p = commands.add_parser("audit-responses", help="held-out semantic quality and error coupling; gold is audit-only")
    p.add_argument("--prepared", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--evaluate-test", action="store_true")

    p = commands.add_parser("train-nano-controller", help="fit a local frozen-LM independent risk head")
    p.add_argument("--cache", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-examples", type=int, default=50000)
    p.add_argument("--matched-mlp", action="store_true", help="fit ordinary MLP on the exact same risk events")
    p.add_argument("--cache-nano-features", action="store_true", help="cache bounded frozen backbone features during fitting")
    p.add_argument("--nano-feature-batch-size", type=int)
    p.add_argument("--feature-normalization", choices=("none", "layernorm"), default="none",
                   help="opt-in trainable normalization before Nano scalar head; default preserves old adapter")

    p = commands.add_parser("train-brig-controller", help="fit grouped fixed-budget Bellman risk-to-go baseline")
    p.add_argument("--cache", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--max-budget", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--no-empty-rollout", action="store_true")

    p = commands.add_parser("train-static-mask", help="fit a global fixed-K straight-through concept mask")
    p.add_argument("--cache", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=.01)
    p.add_argument("--temperature", type=float, default=1.)
    p.add_argument("--relaxation", choices=("sigmoid", "cardinality"), default="sigmoid")
    p.add_argument("--seed", type=int)

    p = commands.add_parser("evaluate", help="validation by default; nano_static_risk is a validation-only fixed-order Nano stopping control")
    p.add_argument("--models", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=("validation", "calibration", "test"), default="validation")
    p.add_argument("--evaluate-test", action="store_true")
    p.add_argument("--frozen-family")
    p.add_argument("--certification-run", action="store_true")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--cost-weight", type=float)
    p.add_argument("--max-groups", type=int, help="unfrozen evaluation only: override acquired query-group budget")
    p.add_argument("--device")
    p.add_argument("--prepared", help="required for live evaluation")
    p.add_argument("--responder", help="enables live measurement; omission means offline replay")
    p.add_argument("--raw-root")
    p.add_argument("--nano-controller")
    p.add_argument("--warmup", type=int, default=0, help="validation-case warmup per policy; live only")
    p.add_argument("--brig-controller", help="source-bound validation-only BRiG artifact")
    p.add_argument("--static-mask", help="source-bound validation-only learned static-mask artifact")

    p = commands.add_parser("freeze", help="freeze a finite complete policy family BEFORE calibration")
    p.add_argument("--models", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cost-weights", type=float, nargs="+")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--mode", choices=("offline_replay", "live"), default="offline_replay")
    p.add_argument("--nano-controller")

    p = commands.add_parser("certify", help="terminal group-risk bound for the previously frozen family")
    p.add_argument("--traces", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--expected-manifest-hash", required=True)
    p.add_argument("--alpha", required=True, type=float)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--assumptions", help="explicit attestation JSON; omission cannot grant a certificate")
    p.add_argument("--out", required=True, help="new JSON file, never overwritten")

    p = commands.add_parser("profile", help="validation-only live batching and shared-compute measurement")
    p.add_argument("--prepared", required=True)
    p.add_argument("--responder", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--raw-root")
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--warmup", type=int, default=3)

    p = commands.add_parser("plan-crossfit", help="LEGACY random toy DAG; ignores prepared fold_id; not for formal runs")
    p.add_argument("--prepared", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=17)

    p = commands.add_parser("execute-crossfit-outer", help="train one complete nested outer fold; no final deployment")
    p.add_argument("--prepared", required=True)
    p.add_argument("--planned", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--outer-fold", type=int, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--responder-config", help="JSON object of train-responder keyword options")
    p.add_argument("--raw-root")

    p = commands.add_parser("merge-crossfit-outer", help="fit unified OOF controller/head from all outer folds; no final deployment")
    p.add_argument("--prepared", required=True)
    p.add_argument("--planned", required=True)
    p.add_argument("--outer-dirs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--controller-only", action="store_true", help="skip final OOF task-head fitting")
    p.add_argument("--controller-batch-size", type=int,
                   help="merge-only controller optimizer batch size; target construction config remains unchanged")
    p.add_argument("--responder-source-dir",
                   help="read-only original release root for historical completed responder caches")

    p = commands.add_parser("fit-crossfit-static", help="fit training-only greedy order using each heldout fold's own head")
    p.add_argument("--prepared", required=True)
    p.add_argument("--planned", required=True)
    p.add_argument("--outer-dirs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--responder-source-dir",
                   help="read-only original release root for historical completed responder caches")

    p = commands.add_parser("evaluate-crossfit", help="validation-only offline evaluation of explicit crossfit components")
    p.add_argument("--prepared", required=True)
    p.add_argument("--planned", required=True)
    p.add_argument("--merged", required=True)
    p.add_argument("--responder", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--methods", nargs="+", choices=("stop", "all", "fixed", "random", "value", "value_singleton", "static", "static_value"))
    p.add_argument("--static-order-dir", help="validated OOF static-order artifact from the same outer folds")
    p.add_argument("--cost-weight", type=float)
    p.add_argument("--max-groups", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--responder-source-dir",
                   help="read-only original release root for historical completed responder caches")

    p = commands.add_parser("plan-crossfit-prepared", help="freeze prepared outer folds and plan nested OOF jobs")
    p.add_argument("--prepared", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--inner-folds", type=int, default=3)

    p = commands.add_parser("verify-crossfit-prepared", help="fail closed unless a prepared-fold plan exactly rebuilds")
    p.add_argument("--prepared", required=True)
    p.add_argument("--planned", required=True)

    p = commands.add_parser("summarize", help="export machine-derived policy metrics to a CSV table")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)

    p = commands.add_parser("smoke", help="end-to-end synthetic text train/cache/replay/live; not paper evidence")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=17)
    return main


def dispatch(args):
    from . import pipeline
    if args.command == "train-choice-pair":
        from .choice_artifacts import train_choice_pair
        return train_choice_pair(args.prepared, args.targets, args.config, args.out,
            max_questions=args.max_questions, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, seed=args.seed, cost_weight=args.cost_weight,
            temperature=args.temperature, remaining_groups=args.remaining_groups,
            budget_mode=args.budget_mode, capacity_control=args.capacity_control)
    if args.command == "evaluate-choice-crossfit":
        from .choice_evaluation import evaluate_choice_crossfit_validation
        return evaluate_choice_crossfit_validation(args.prepared, args.planned, args.merged,
            args.responder, args.cache, args.choice, args.targets, args.target_config, args.out,
            cost_weight=args.cost_weight, max_groups=args.max_groups, device=args.device,
            responder_source_dir=args.responder_source_dir)
    if args.command == "doctor":
        result = environment()
        if args.check_device:
            import torch
            device = torch.device(args.check_device)
            if device.type not in ("cpu", "cuda"):
                raise ValueError("device check supports cpu/cuda only")
            value = torch.ones((8, 8), device=device)
            result["small_forward"] = {"device": str(device), "sum": float((value @ value).sum())}
        if args.out:
            write_json(args.out, result)
        return result
    if args.command == "prepare":
        from .data import prepare_dataset
        result = prepare_dataset(args.dataset, args.source, args.out, seed=args.seed,
                                 source_revision=args.source_revision)
        return {key: str(value) if isinstance(value, Path) else value for key, value in result.items()}
    if args.command in ("train-responder", "train-crossfit-responder"):
        options = dict(kind=args.kind, raw_root=args.raw_root,
                   device=args.device, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
                   learning_rate=args.learning_rate, backbone=args.backbone, image_size=args.image_size,
                   concept_class_weighting=args.concept_class_weighting,
                   revision=args.revision, allow_download=args.allow_download, max_length=args.max_length,
                   freeze_backbone=args.freeze_backbone, input_workers=args.input_workers,
                   input_prefetch_batches=args.input_prefetch_batches)
        if args.command == "train-crossfit-responder":
            return pipeline.train_crossfit_responder(
                args.prepared, args.planned, args.out, outer_fold=args.outer_fold,
                inner_fold=args.inner_fold, final=args.final, **options)
        return pipeline.train_responder(args.prepared, args.out, **options)
    if args.command == "cache-crossfit":
        from .crossfit_cache import cache_crossfit_responses
        return cache_crossfit_responses(args.prepared, args.planned, args.responder,
                                       args.out, raw_root=args.raw_root,
                                       device=args.device, split=args.split)
    if args.command == "cache":
        return pipeline.cache_responses(args.prepared, args.responder, args.out,
                                        raw_root=args.raw_root, device=args.device)
    if args.command == "train":
        config = read_json(args.config)
        if args.seed is not None:
            config["seed"] = args.seed
        if args.device is not None:
            config["device"] = args.device
        return pipeline.train_models(args.cache, config, args.out)
    if args.command == "audit-responses":
        from .audit import audit_responses
        return audit_responses(args.prepared, args.cache, args.out, split=args.split,
                                evaluate_test=args.evaluate_test)
    if args.command == "train-nano-controller":
        return pipeline.train_nano_controller(args.cache, args.models, args.backbone, args.out,
                     device=args.device, epochs=args.epochs, batch_size=args.batch_size,
                     learning_rate=args.learning_rate, max_length=args.max_length,
                     max_examples=args.max_examples, matched_mlp=args.matched_mlp,
                     cache_nano_features=args.cache_nano_features,
                     nano_feature_batch_size=args.nano_feature_batch_size,
                     feature_normalization=args.feature_normalization)
    if args.command == "train-brig-controller":
        config = {key: getattr(args, key) for key in
                  ("device", "epochs", "batch_size", "learning_rate", "hidden", "max_budget", "seed")
                  if getattr(args, key) is not None}
        config["empty_rollout"] = not args.no_empty_rollout
        return pipeline.train_brig_controller(args.cache, args.models, args.out, config=config)
    if args.command == "train-static-mask":
        config = {key: getattr(args, key) for key in
                  ("k", "device", "epochs", "batch_size", "learning_rate", "temperature", "seed", "relaxation")
                  if getattr(args, key) is not None}
        return pipeline.train_static_mask(args.cache, args.models, args.out, config=config)
    if args.command == "evaluate":
        return pipeline.evaluate_models(args.models, args.cache, args.out, split=args.split,
                    evaluate_test=args.evaluate_test, frozen_family=args.frozen_family,
                    certification_run=args.certification_run, methods=args.methods,
                    cost_weight=args.cost_weight, max_groups=args.max_groups, device=args.device, prepared=args.prepared,
                    responder_dir=args.responder, raw_root=args.raw_root,
                    nano_dir=args.nano_controller, warmup=args.warmup, brig_dir=args.brig_controller,
                    static_mask_dir=args.static_mask)
    if args.command == "freeze":
        return pipeline.freeze_family(args.models, args.cache, args.out,
                    weights=args.cost_weights, methods=args.methods, mode=args.mode,
                    nano_dir=args.nano_controller)
    if args.command == "certify":
        from .evaluation import certify_policies
        result = certify_policies(read_jsonl(args.traces), read_json(args.manifest),
                   alpha=args.alpha, delta=args.delta, expected_manifest_hash=args.expected_manifest_hash,
                   assumptions=read_json(args.assumptions) if args.assumptions else None)
        write_json(args.out, result)
        return result
    if args.command == "profile":
        from .profiling import profile_backend
        return profile_backend(args.prepared, args.responder, args.out, raw_root=args.raw_root,
                  device=args.device, limit=args.limit, repeats=args.repeats,
                  batch_sizes=tuple(args.batch_sizes), warmup=args.warmup)
    if args.command == "plan-crossfit":
        from .provenance import plan_nested_crossfit
        _, rows, membership = pipeline.load_prepared(args.prepared)
        groups = sorted({r["group_id"] for r in rows if membership[r["sample_id"]]["split"]
                         in ("responder_fit", "head_fit", "policy_fit")})
        result = plan_nested_crossfit(groups, seed=args.seed)
        write_json(args.out, result)
        return {"out": args.out, "status": "PLANNED_NOT_TRAINED"}
    if args.command == "execute-crossfit-outer":
        from .crossfit_executor import execute_outer_fold
        return execute_outer_fold(args.prepared, args.planned, args.out,
                                  outer_fold=args.outer_fold, config=read_json(args.config),
                                  responder_options=read_json(args.responder_config) if args.responder_config else None,
                                  raw_root=args.raw_root)
    if args.command == "merge-crossfit-outer":
        from .crossfit_merge import merge_outer_folds
        return merge_outer_folds(args.prepared, args.planned, args.outer_dirs, args.out,
                                 fit_final_head=not args.controller_only,
                                 responder_source_dir=args.responder_source_dir,
                                 controller_batch_size=args.controller_batch_size)
    if args.command == "fit-crossfit-static":
        from .crossfit_static import fit_crossfit_static_order
        order, report = fit_crossfit_static_order(args.prepared, args.planned, args.outer_dirs,
            args.out, device=args.device, batch_size=args.batch_size,
            responder_source_dir=args.responder_source_dir)
        return {"out": args.out, "order": order, "report": report}
    if args.command == "evaluate-crossfit":
        from .crossfit_evaluation import evaluate_crossfit_validation
        return evaluate_crossfit_validation(args.prepared, args.planned, args.merged,
            args.responder, args.cache, args.out, methods=args.methods,
            cost_weight=args.cost_weight, max_groups=args.max_groups, device=args.device,
            static_order_dir=args.static_order_dir, responder_source_dir=args.responder_source_dir)
    if args.command == "plan-crossfit-prepared":
        from .crossfit import plan_crossfit_prepared
        return plan_crossfit_prepared(args.prepared, args.out, inner_folds=args.inner_folds)
    if args.command == "verify-crossfit-prepared":
        from .crossfit import verify_crossfit_prepared
        return verify_crossfit_prepared(args.prepared, args.planned)
    if args.command == "summarize":
        return pipeline.export_summary(args.runs, args.out)
    if args.command == "smoke":
        return pipeline.smoke(args.out, seed=args.seed)
    raise ValueError("unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        print(json.dumps({"status": "ERROR", "type": type(exc).__name__, "message": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

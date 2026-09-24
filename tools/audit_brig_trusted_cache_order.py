#!/usr/bin/env python3
"""Audit whether fixed-order BRiG risk caching matches stochastic minibatch targets.

This is a development diagnostic for the seed/fold-specific BRiG comparison.
It verifies the source-bound RLE and trusted-cached artifacts, reconstructs
the first budget-2 minibatch using the fitter's Python RNG stream, and compares
the RLE per-minibatch rollout risks to the trusted cache indexed in shuffled
order. It never reads validation/test labels and is not paper evidence.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch

from cbmjev.brig import _terminal
from cbmjev.io import file_hash, read_json, write_json
from cbmjev.learning import training_rows
from scripts.brig_batched_rollout import empty_rollout_targets
from scripts.brig_q_rle import GroupQRunLength
from scripts.brig_trusted_ops import (
    TrustedTrainingGroupQ,
    empty_rollout_targets_trusted,
)
from scripts.train_evaluate_cub_brig_crossfit import _fold_sources
from scripts.train_evaluate_cub_brig_rle_crossfit import verify_rle_fold
from scripts.train_evaluate_cub_brig_trusted_cached_crossfit import (
    verify_trusted_cached_fold,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "planned", "merged", "responder", "cache",
                 "rle-fold", "trusted-fold", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allocator-gib", type=float, default=2.0)
    parser.add_argument("--reserve-gib", type=float, default=8.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if Path(args.out).exists():
        raise FileExistsError("audit output must be new")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("pin exactly one authorized physical GPU")
        free, total = torch.cuda.mem_get_info(0)
        if free < (args.allocator_gib + args.reserve_gib) * 2**30:
            raise RuntimeError("insufficient free GPU memory plus required reserve")
        torch.cuda.set_per_process_memory_fraction(args.allocator_gib * 2**30 / total, 0)
    if args.batch_size < 1 or args.epochs < 1 or args.seed < 0:
        raise ValueError("invalid seed, batch size, or epoch count")

    rle_receipt = verify_rle_fold(args.rle_fold)
    verify_trusted_cached_fold(args.trusted_fold)
    if rle_receipt["outer_fold"] != args.outer_fold:
        raise ValueError("RLE receipt outer fold differs from request")
    base_receipt = read_json(Path(args.rle_fold) / "receipt.json")

    root_args = SimpleNamespace(
        prepared=Path(args.prepared), planned=Path(args.planned),
        merged=Path(args.merged), responder=Path(args.responder),
        cache=Path(args.cache), responder_source_dir=None, device=args.device)
    schema, rows, head, binding, _, _ = _fold_sources(root_args, args.outer_fold)
    fit_rows = [{**row, "split": "policy_fit"} for row in rows]
    examples = training_rows(fit_rows, "policy_fit", schema)
    if len(examples) != base_receipt["binding"]["training_sample_count"]:
        raise ValueError("loaded policy-fit rows differ from fold receipt")

    rle_payload = torch.load(Path(args.rle_fold) / "brig.pt",
                             map_location="cpu", weights_only=True)
    trusted_payload = torch.load(Path(args.trusted_fold) / "brig.pt",
                                 map_location="cpu", weights_only=True)
    if not torch.equal(rle_payload["1"]["net.0.weight"],
                       trusted_payload["1"]["net.0.weight"]):
        raise ValueError("Q1 first-layer weights are not bitwise equal")
    if set(rle_payload["1"]) != set(trusted_payload["1"]) or any(
            not torch.equal(rle_payload["1"][key], trusted_payload["1"][key])
            for key in rle_payload["1"]):
        raise ValueError("Q1 checkpoints are not bitwise equal")

    device = torch.device(args.device)
    q_rle = GroupQRunLength(schema, args.hidden).to(device)
    q_rle.load_state_dict(rle_payload["1"])
    q_rle.eval()
    q_trusted = TrustedTrainingGroupQ(schema, args.hidden).to(device)
    q_trusted.load_state_dict(trusted_payload["1"])
    q_trusted.eval()

    rng = random.Random(args.seed)
    for budget in (1,):
        for _ in range(args.epochs):
            order = list(range(len(examples)))
            rng.shuffle(order)
            for start in range(0, len(order), args.batch_size):
                for index in order[start:start + args.batch_size]:
                    answers, _ = examples[index]
                    kind = rng.randrange(3)
                    visible_count = rng.randrange(schema.num_groups - budget + 1)
                    if kind:
                        rng.sample(range(schema.num_groups), visible_count)

    order = list(range(len(examples)))
    rng.shuffle(order)
    indices = order[:args.batch_size]
    if not indices:
        raise ValueError("empty reconstructed minibatch")
    batch = [examples[index] for index in indices]

    fixed_order_risk_chunks = []
    fixed_order_terminal_states = []
    for cache_start in range(0, len(examples), args.batch_size):
        cache_examples = examples[cache_start:cache_start + args.batch_size]
        _, _, terminal_states, terminal_labels = empty_rollout_targets_trusted(
            schema, {1: q_trusted}, cache_examples, 2)
        fixed_order_terminal_states.extend(
            terminal_states[i * schema.num_groups:(i + 1) * schema.num_groups]
            for i in range(len(cache_examples)))
        risks = _terminal(head, terminal_states, terminal_labels, args.device)
        fixed_order_risk_chunks.append(
            risks.reshape(len(cache_examples), schema.num_groups))
    fixed_order_cache = torch.cat(fixed_order_risk_chunks, dim=0)[indices].reshape(-1)
    cached_terminal_states = [state for index in indices
        for state in fixed_order_terminal_states[index]]
    _, _, rle_states, rle_labels = empty_rollout_targets(
        schema, {1: q_rle}, batch, 2)
    _, _, trusted_states, trusted_labels = empty_rollout_targets_trusted(
        schema, {1: q_trusted}, batch, 2)
    if rle_states != trusted_states or rle_labels != trusted_labels:
        raise ValueError("same-minibatch terminal trajectories/labels differ")
    minibatch_risks = _terminal(head, rle_states, rle_labels, args.device)
    trusted_minibatch_risks = _terminal(
        head, trusted_states, trusted_labels, args.device)
    if not torch.equal(minibatch_risks, trusted_minibatch_risks):
        raise ValueError("RLE and trusted per-minibatch risks differ")
    cached_path_risks_in_batch_order = _terminal(
        head, cached_terminal_states, trusted_labels, args.device)

    sample_ids = [row["sample_id"] for row in rows]
    batch_sample_ids = [sample_ids[index] for index in indices]
    result = {
        "format": "cbmjev-brig-cache-order-audit-v1",
        "status": "COMPLETE",
        "paper_evidence": False,
        "evidence_status": "TRAINING_TARGET_NUMERICS_DIAGNOSTIC_NOT_METHOD_OR_TEST_EVIDENCE",
        "split": "policy_fit",
        "outer_fold": args.outer_fold,
        "seed": args.seed,
        "budget": 2,
        "epoch": 0,
        "batch_size": args.batch_size,
        "q1_checkpoint_bitwise_equal": True,
        "same_minibatch_terminal_states_equal": True,
        "same_minibatch_risk_targets_bitwise_equal": True,
        "fixed_order_cached_terminal_states_match_random_batch":
            cached_terminal_states == rle_states,
        "fixed_order_terminal_state_mismatch_count": sum(
            left != right for left, right in zip(cached_terminal_states, rle_states)),
        "fixed_order_paths_revalued_in_random_batch_order_bitwise_equal": bool(
            torch.equal(cached_path_risks_in_batch_order, minibatch_risks)),
        "fixed_order_paths_revalued_max_abs_difference": float(
            (cached_path_risks_in_batch_order - minibatch_risks).abs().max().item()),
        "fixed_order_cache_vs_random_minibatch_bitwise_equal": bool(
            torch.equal(minibatch_risks, fixed_order_cache)),
        "risk_target_count": int(minibatch_risks.numel()),
        "risk_target_mismatch_count": int(
            (minibatch_risks != fixed_order_cache).sum().item()),
        "risk_target_max_abs_difference": float(
            (minibatch_risks - fixed_order_cache).abs().max().item()),
        "minibatch_indices": indices,
        "minibatch_indices_sha256": hashlib.sha256(
            json.dumps(indices, separators=(",", ":")).encode()).hexdigest(),
        "minibatch_sample_ids": batch_sample_ids,
        "rle_fold_receipt_sha256": file_hash(
            Path(args.rle_fold) / "receipt.json"),
        "trusted_fold_receipt_sha256": file_hash(
            Path(args.trusted_fold) / "receipt.json"),
        "trusted_variant_receipt_sha256": file_hash(
            Path(args.trusted_fold) / "trusted_cached_variant_receipt.json"),
    }
    write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare one explicit official CEBaB training variant; never concatenate."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbmjev.data import prepare_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--train-variant", required=True, choices=("train_exclusive", "train_inclusive"))
    parser.add_argument("--heldout-reference", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.train_variant == "train_inclusive" and args.heldout_reference is None:
        parser.error("inclusive migration requires --heldout-reference to keep existing evaluation membership fixed")
    result = prepare_dataset("cebab", args.source, args.out, seed=args.seed,
                             source_revision=args.source_revision, train_variant=args.train_variant,
                             heldout_reference=args.heldout_reference)
    audit = result["report"]
    print(json.dumps({"out": str(args.out), "observed_counts": audit["observed_counts"],
                      "effective_task_counts": audit["effective_task_counts"],
                      "role_counts": audit["role_counts"], "exclusions": audit["exclusions_by_reason"],
                      "split_hash": audit["split_hash"]}, sort_keys=True))

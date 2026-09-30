#!/usr/bin/env python3
"""Inspect or prepare a local official crowd-enVENT archive; print metadata only."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbmjev.appraisal_data import DEFAULT_SEED, inspect_crowd_envent, prepare_appraisal_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="local 2023 ZIP or extracted release directory")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--source-revision", default="crowd-enVent2023")
    args = parser.parse_args()
    if args.inspect_only:
        if args.out is not None:
            parser.error("--inspect-only does not write an output directory")
        result = inspect_crowd_envent(args.source)
    else:
        if args.out is None:
            parser.error("preparation requires --out")
        result = prepare_appraisal_data(args.source, args.out, args.seed, args.source_revision)["report"]
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare two policies in one source-bound validation evaluation, not causal gains."""
import argparse
import json
from pathlib import Path

if __package__:
    from .analyze_adaptive_branching import load_run, validate_comparison_identity
else:
    from analyze_adaptive_branching import load_run, validate_comparison_identity


def compare(left, right, *, bootstrap_resamples=0):
    if type(bootstrap_resamples) is not int or bootstrap_resamples < 0:
        raise ValueError("bootstrap_resamples must be a nonnegative integer")
    identity = validate_comparison_identity((left, right))
    if set(left["traces"]) != set(right["traces"]):
        raise ValueError("policy sample sets differ")
    differences = {field: 0 for field in ("prediction", "final_state", "queried_groups", "actions")}
    for sid, a in left["traces"].items():
        b = right["traces"][sid]
        if (a["group_id"], a["y"]) != (b["group_id"], b["y"]):
            raise ValueError("policy targets/groups differ")
        for field in ("prediction", "final_state", "queried_groups"):
            differences[field] += a[field] != b[field]
        differences["actions"] += ([step["action"] for step in a["steps"]]
                                   != [step["action"] for step in b["steps"]])
    report = {"format": "cbmjev-policy-trace-comparison-v1",
            "evidence_status": "VALIDATION_DEVELOPMENT_NOT_MATCHED_BUDGET_CAUSAL_COMPARISON",
            "seed": left["metrics"]["seed"], "num_samples": len(left["traces"]),
            "policies": {run["report"]["method"]: {key: run["report"][key]
                for key in ("accuracy", "macro_f1", "mean_queried_groups", "mean_calls")}
                for run in (left, right)},
            "differing_samples": differences, "comparison_artifact_identity": identity,
            "input_hashes": {"left": left["hashes"], "right": right["hashes"]}}
    if bootstrap_resamples:
        from cbmjev.evaluation import paired_group_bootstrap
        paired = paired_group_bootstrap(list(left["traces"].values()),
            list(right["traces"].values()), seed=left["metrics"]["seed"],
            n_resamples=bootstrap_resamples)
        # Keep raw group identities on the server; the digest/count suffice here.
        paired.pop("group_ids")
        report["paired_group_bootstrap"] = paired
        report["inference_warning"] = (
            "Exploratory validation comparison after method development; no selection/multiplicity "
            "adjustment, no training-seed uncertainty, and equal group count does not imply equal atom/call cost.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--left", default="static_value")
    parser.add_argument("--right", default="value_singleton")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    report = compare(load_run(Path(args.run), args.left), load_run(Path(args.run), args.right),
                     bootstrap_resamples=args.bootstrap_resamples)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"out": str(out), "differences": report["differing_samples"]}))


if __name__ == "__main__":
    main()

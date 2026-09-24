#!/usr/bin/env python3
"""Check CPU/GPU replay decisions for the same source-bound Choice model.

This compares task behavior, not latency. Small floating-point differences are
reported separately; an action or prediction mismatch fails parity.
"""

import argparse
from pathlib import Path

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, read_json, write_json
from scripts.summarize_cub_choice_pair import _verified_eval


def _indexed(rows):
    index = {}
    for row in rows:
        key = (row["policy_id"], row["sample_id"])
        if key in index:
            raise ValueError("duplicate policy/sample trace")
        index[key] = row
    return index


def compare_rows(cpu_rows, gpu_rows):
    cpu, gpu = _indexed(cpu_rows), _indexed(gpu_rows)
    if not cpu or set(cpu) != set(gpu):
        raise ValueError("CPU/GPU policy and sample coverage differ")
    result = {"traces": len(cpu), "by_policy": {}}
    for policy in sorted({key[0] for key in cpu}):
        keys = sorted(key for key in cpu if key[0] == policy)
        action_mismatch = prediction_mismatch = final_state_mismatch = 0
        max_probability_abs_delta = 0.0
        for key in keys:
            left, right = cpu[key], gpu[key]
            if ((left["sample_id"], left["group_id"], left["y"], left["split"], left["mode"])
                    != (right["sample_id"], right["group_id"], right["y"],
                        right["split"], right["mode"])):
                raise ValueError("CPU/GPU validation identities differ")
            actions_left = [step["action"] for step in left["steps"]]
            actions_right = [step["action"] for step in right["steps"]]
            action_mismatch += (actions_left != actions_right or
                                left["queried_groups"] != right["queried_groups"])
            prediction_mismatch += left["prediction"] != right["prediction"]
            final_state_mismatch += left["final_state"] != right["final_state"]
            if len(left["probabilities"]) != len(right["probabilities"]):
                raise ValueError("CPU/GPU task-head class count differs")
            max_probability_abs_delta = max(max_probability_abs_delta,
                *(abs(a - b) for a, b in zip(left["probabilities"], right["probabilities"])))
        result["by_policy"][policy] = {
            "samples": len(keys), "action_sequence_mismatches": action_mismatch,
            "prediction_mismatches": prediction_mismatch,
            "final_state_mismatches": final_state_mismatch,
            "max_probability_abs_delta": max_probability_abs_delta}
    result["behavior_equal"] = all(
        not any(values[name] for name in ("action_sequence_mismatches",
                                         "prediction_mismatches", "final_state_mismatches"))
        for values in result["by_policy"].values())
    return result


def compare(cpu_dir, gpu_dir):
    cpu_dir, gpu_dir = Path(cpu_dir), Path(gpu_dir)
    expected_format = read_json(cpu_dir / "receipt.json")["format"]
    cpu_receipt, cpu_metrics, cpu_rows = _verified_eval(cpu_dir, expected_format)
    gpu_receipt, gpu_metrics, gpu_rows = _verified_eval(gpu_dir, expected_format)
    for key in ("source_binding", "source_code_hash", "samples_per_policy", "num_policies"):
        if cpu_receipt[key] != gpu_receipt[key]:
            raise ValueError("CPU/GPU source-bound receipt mismatch: " + key)
    if (cpu_metrics["source_binding"] != gpu_metrics["source_binding"]
            or set(cpu_metrics["policies"]) != set(gpu_metrics["policies"])):
        raise ValueError("CPU/GPU policy metric bindings differ")
    result = {"format": "cbmjev-choice-device-parity-v1",
        "evidence_status": "DEVELOPMENT_REPLAY_CONSISTENCY_NOT_LATENCY_EVIDENCE",
        "cpu_receipt_sha256": file_hash(cpu_dir / "receipt.json"),
        "gpu_receipt_sha256": file_hash(gpu_dir / "receipt.json"),
        "source_binding_verified": True,
        **compare_rows(cpu_rows, gpu_rows)}
    result["report_hash"] = stable_hash(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-eval", required=True)
    parser.add_argument("--gpu-eval", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise ValueError("parity output must be new")
    result = compare(args.cpu_eval, args.gpu_eval)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print({"out": str(out), "behavior_equal": result["behavior_equal"]}, flush=True)
    if not result["behavior_equal"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

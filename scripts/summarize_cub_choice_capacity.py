#!/usr/bin/env python3
"""Receipt-bound CUB K16 capacity control: set attention versus matched MLP."""

import argparse
from pathlib import Path

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, read_json, write_json
from scripts.summarize_cub_choice_pair import (_basic, _compare, _one_policy,
                                                _verified_eval)


MATCHED_FIT_KEYS = ("source_content_sha256", "effective_features_sha256",
    "derived_ids_in_order", "initial_shared_tensors_sha256",
    "initial_active_logits_sha256", "epoch_order_sha256", "order_sha256",
    "questions_exposed_per_head", "candidates_exposed_per_head",
    "optimizer_steps_per_head")


def _verified_fit(directory, expected_format):
    directory = Path(directory)
    receipt = read_json(directory / "receipt.json")
    unsigned = dict(receipt)
    if (unsigned.pop("receipt_sha256", None) != stable_hash(unsigned)
            or receipt.get("format") != expected_format
            or receipt.get("status") != "COMPLETE"):
        raise ValueError("invalid Choice fit receipt")
    expected_files = ({"scalar.pt", "attention.pt", "independent_mlp.pt",
                       "config.json", "report.json"} if "capacity" in expected_format else
                      {"scalar.pt", "attention.pt", "config.json", "report.json"})
    if set(receipt.get("files", {})) != expected_files:
        raise ValueError("unexpected Choice fit files")
    for name in expected_files:
        if receipt["files"][name] != file_hash(directory / name):
            raise ValueError("Choice fit file hash mismatch: " + name)
    return receipt, read_json(directory / "report.json")


def summarize_capacity(capacity_fit, capacity_eval, soft_fit, fixed_eval,
                       *, draws=2000, seed=60):
    if type(draws) is not int or draws < 100 or type(seed) is not int or seed < 0:
        raise ValueError("invalid bootstrap settings")
    capacity_receipt, capacity_report = _verified_fit(
        capacity_fit, "cbmjev-structured-choice-capacity-artifact-v1")
    soft_receipt, soft_report = _verified_fit(
        soft_fit, "cbmjev-structured-choice-pair-artifact-v1")
    control = capacity_report["fit"]
    original = soft_report["fit"]
    if (capacity_report.get("selected_derived_ids") != soft_report.get("selected_derived_ids")
            or capacity_report.get("package_bindings") != soft_report.get("package_bindings")
            or any(control.get(key) != original.get(key) for key in MATCHED_FIT_KEYS)
            or any(control["heads"][name] != original["heads"][name]
                   for name in ("scalar", "attention"))):
        raise ValueError("capacity and original Choice fits are not exactly matched")
    if not control.get("capacity_control", {}).get("same_visible_features_and_log_candidate_count"):
        raise ValueError("capacity-control feature parity not declared")
    capacity_eval_receipt, capacity_metrics, capacity_rows = _verified_eval(
        capacity_eval, "cbmjev-choice-crossfit-validation-v1")
    fixed_receipt, fixed_metrics, fixed_rows = _verified_eval(
        fixed_eval, "cbmjev-crossfit-validation-v1")
    source = capacity_eval_receipt["source_binding"]["nested_system"]
    if (capacity_eval_receipt.get("num_policies") != 3
            or capacity_eval_receipt["source_binding"].get("choice_receipt_sha256") !=
               file_hash(Path(capacity_fit) / "receipt.json")
            or source != fixed_receipt["source_binding"]
            or capacity_metrics["source_binding"] != capacity_eval_receipt["source_binding"]
            or fixed_metrics["source_binding"] != fixed_receipt["source_binding"]
            or {capacity_metrics.get("seed"), fixed_metrics.get("seed")} != {seed}):
        raise ValueError("capacity and fixed evaluations have mismatched sources or seed")
    names = ("scalar", "attention", "independent_mlp")
    policies = {name: _one_policy(capacity_rows, policy_id="structured_choice_" + name)
                for name in names}
    if set(capacity_metrics.get("policies", {})) != {
            "structured_choice_" + name for name in names}:
        raise ValueError("capacity evaluation policy set mismatch")
    policies["fixed_K16"] = _one_policy(fixed_rows, method="fixed")
    if (fixed_metrics["policies"]["fixed"]["mean_queried_groups"] != 16
            or any(len(row["queried_groups"]) != 16 for row in policies["fixed_K16"])):
        raise ValueError("fixed comparator must use exact K16")
    specs = read_json(Path(capacity_eval) / "settings.json")
    if set(specs) != {"structured_choice_" + name for name in names}:
        raise ValueError("capacity evaluation specs mismatch")
    for name in names:
        spec = specs["structured_choice_" + name]
        if (spec.get("max_groups") != 16 or spec.get("split") != "validation"
                or spec.get("mode") != "offline_replay"
                or spec.get("source_binding") != capacity_eval_receipt["source_binding"]
                or spec.get("head_weights_sha256") !=
                   control["heads"][name]["final_weights_sha256"]
                or any(row.get("system_hash") != spec.get("system_hash")
                       for row in policies[name])):
            raise ValueError("capacity trace/spec/weights/budget mismatch")
    pairs = (("attention", "independent_mlp"), ("attention", "scalar"),
             ("independent_mlp", "scalar"), ("attention", "fixed_K16"),
             ("independent_mlp", "fixed_K16"))
    result = {"format": "cbmjev-choice-capacity-control-k16-summary-v1",
        "evidence_status": "SINGLE_SEED_DEVELOPMENT_VALIDATION_NOT_LOCKED_TEST",
        "seed": seed, "split": "validation", "samples": 594,
        "same_training_examples_features_order_and_original_pair_weights_verified": True,
        "same_nested_evaluation_system_verified": True,
        "capacity_fit_receipt_sha256": file_hash(Path(capacity_fit) / "receipt.json"),
        "soft_fit_receipt_sha256": file_hash(Path(soft_fit) / "receipt.json"),
        "capacity_eval_receipt_sha256": file_hash(Path(capacity_eval) / "receipt.json"),
        "fixed_eval_receipt_sha256": file_hash(Path(fixed_eval) / "receipt.json"),
        "parameter_count": control["capacity_control"]["parameter_count"],
        "metrics": {name: _basic(rows) for name, rows in policies.items()},
        "comparisons": {a + "_minus_" + b: _compare(policies[a], policies[b],
            seed=seed + 1000 + i * 1000, draws=draws)
            for i, (a, b) in enumerate(pairs)},
        "interpretation_limit": "One seed and development validation only; matched capacity does not equate optimization geometry, inductive bias or pretraining. Query/cost behavior accompanies accuracy. No JEV-specific or latency claim follows from this alone."}
    result["report_hash"] = stable_hash(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capacity-fit", "capacity-eval", "soft-fit", "fixed-eval", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=60)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise ValueError("summary output must be new")
    result = summarize_capacity(args.capacity_fit, args.capacity_eval,
        args.soft_fit, args.fixed_eval, draws=args.draws, seed=args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print({"status": "COMPLETE", "out": str(out), "metrics": result["metrics"]}, flush=True)


if __name__ == "__main__":
    main()

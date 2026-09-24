#!/usr/bin/env python3
"""Paired seed60 CUB K16 soft-CE/risk-regression/fixed development summary."""

import argparse
from pathlib import Path

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, read_json, write_json
from scripts.summarize_cub_choice_pair import (_basic, _compare, _one_policy,
                                                _verified_eval)


def summarize_objectives(soft_dir, risk_dir, fixed_dir, *, draws=2000, seed=60):
    if type(draws) is not int or draws < 100 or type(seed) is not int or seed < 0:
        raise ValueError("invalid bootstrap settings")
    soft_receipt, soft_metrics, soft_rows = _verified_eval(
        soft_dir, "cbmjev-choice-crossfit-validation-v1")
    risk_receipt, risk_metrics, risk_rows = _verified_eval(
        risk_dir, "cbmjev-choice-risk-crossfit-validation-v1")
    fixed_receipt, fixed_metrics, fixed_rows = _verified_eval(
        fixed_dir, "cbmjev-crossfit-validation-v1")
    source = soft_receipt["source_binding"]["nested_system"]
    if (source != risk_receipt["source_binding"]["nested_system"]
            or source != fixed_receipt["source_binding"]
            or soft_metrics["source_binding"] != soft_receipt["source_binding"]
            or risk_metrics["source_binding"] != risk_receipt["source_binding"]
            or fixed_metrics["source_binding"] != fixed_receipt["source_binding"]
            or {soft_metrics.get("seed"), risk_metrics.get("seed"), fixed_metrics.get("seed")} != {seed}
            or soft_receipt["source_binding"]["choice_receipt_sha256"] !=
               risk_receipt["source_binding"]["soft_choice_receipt_sha256"]):
        raise ValueError("objectives/fixed do not share the same nested system, seed and soft fit")
    soft_specs, risk_specs, fixed_specs = (read_json(Path(directory) / "settings.json")
        for directory in (soft_dir, risk_dir, fixed_dir))
    soft_names = ("structured_choice_scalar", "structured_choice_attention")
    risk_names = ("structured_choice_risk_scalar", "structured_choice_risk_attention")
    comparable = []
    for specs, names, receipt in ((soft_specs, soft_names, soft_receipt),
                                 (risk_specs, risk_names, risk_receipt)):
        if set(specs) != set(names):
            raise ValueError("Choice evaluation policy set mismatch")
        for name in names:
            spec = specs[name]
            if (spec.get("max_groups") != 16 or spec.get("mode") != "offline_replay"
                    or spec.get("split") != "validation"
                    or spec.get("runtime_method") != "structured_choice"
                    or spec.get("source_binding") != receipt["source_binding"]):
                raise ValueError("Choice evaluation budget/source/protocol mismatch")
            comparable.append((spec["cost_weight"], spec["cost"]))
    fixed_spec = fixed_specs.get("fixed")
    if (not isinstance(fixed_spec, dict) or fixed_spec.get("config", {}).get("policy", {}).get("max_groups") != 16
            or fixed_spec.get("config", {}).get("cost") != comparable[0][1]
            or fixed_spec.get("source_binding") != fixed_receipt["source_binding"]):
        raise ValueError("fixed K16 budget/cost/source protocol mismatch")
    if any(item != comparable[0] for item in comparable):
        raise ValueError("soft/risk Choice cost and lambda settings differ")
    policies = {
        "soft_scalar": _one_policy(soft_rows, policy_id="structured_choice_scalar"),
        "soft_attention": _one_policy(soft_rows, policy_id="structured_choice_attention"),
        "risk_scalar": _one_policy(risk_rows, policy_id="structured_choice_risk_scalar"),
        "risk_attention": _one_policy(risk_rows, policy_id="structured_choice_risk_attention"),
        "fixed_K16": _one_policy(fixed_rows, method="fixed")}
    fixed = policies["fixed_K16"]
    if (fixed_metrics["policies"]["fixed"]["mean_queried_groups"] != 16
            or any(len(row["queried_groups"]) != 16 for row in fixed)):
        raise ValueError("fixed comparator must use exact K16")
    for name, rows in policies.items():
        spec = fixed_spec if name == "fixed_K16" else (
            soft_specs[name.replace("soft_", "structured_choice_")] if name.startswith("soft_")
            else risk_specs[name.replace("risk_", "structured_choice_risk_")])
        if any(row.get("system_hash") != spec.get("system_hash")
               or len(row["queried_groups"]) > 16 for row in rows):
            raise ValueError("trace/spec identity or K16 budget mismatch")
    pairs = (("risk_scalar", "soft_scalar"), ("risk_attention", "soft_attention"),
             ("risk_attention", "risk_scalar"), ("soft_attention", "soft_scalar"),
             ("risk_scalar", "fixed_K16"), ("risk_attention", "fixed_K16"),
             ("soft_scalar", "fixed_K16"), ("soft_attention", "fixed_K16"))
    result = {"format": "cbmjev-choice-objective-control-k16-summary-v1",
        "evidence_status": "SINGLE_SEED_DEVELOPMENT_VALIDATION_NOT_LOCKED_TEST",
        "seed": seed, "split": "validation", "samples": 594,
        "same_nested_system_and_soft_fit_verified": True,
        "soft_eval_receipt_sha256": file_hash(Path(soft_dir) / "receipt.json"),
        "risk_eval_receipt_sha256": file_hash(Path(risk_dir) / "receipt.json"),
        "fixed_eval_receipt_sha256": file_hash(Path(fixed_dir) / "receipt.json"),
        "metrics": {name: _basic(rows) for name, rows in policies.items()},
        "comparisons": {a + "_minus_" + b: _compare(policies[a], policies[b],
            seed=seed + 1000 + i * 1000, draws=draws)
            for i, (a, b) in enumerate(pairs)},
        "interpretation_limit": "One seed on development validation, fixed trained heads and group-resampling uncertainty only. Soft CE and utility MSE use identical OOF examples and initializations, but different objectives; neither demonstrates JEV-specific or multi-step optimality. Query/cost counts must accompany accuracy. Shared-GPU timing is not paper evidence."}
    result["report_hash"] = stable_hash(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("soft-eval", "risk-eval", "fixed-eval", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=60)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise ValueError("summary output must be new")
    result = summarize_objectives(args.soft_eval, args.risk_eval,
                                  args.fixed_eval, draws=args.draws, seed=args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print({"status": "COMPLETE", "out": str(out), "metrics": result["metrics"]},
          flush=True)


if __name__ == "__main__":
    main()

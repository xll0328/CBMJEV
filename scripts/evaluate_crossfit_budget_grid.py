#!/usr/bin/env python3
"""Evaluate a CUB/CBMJev crossfit validation budget grid with one source load.

This is a throughput helper around the same audited crossfit evaluation
components.  It avoids reloading and revalidating large outer-fold artifacts for
every budget point, while preserving machine-readable metrics, traces, settings
and a receipt.
"""
import argparse
import json
from pathlib import Path

from cbmjev.config import resolve_config
from cbmjev.contracts import DeclaredCost, stable_hash
from cbmjev.crossfit_evaluation import _load_sources, _verify_binding_files_unchanged
from cbmjev.evaluation import summarize_traces
from cbmjev.io import file_hash, fresh_dir, write_json, write_jsonl
from cbmjev.pipeline import code_fingerprint
from cbmjev.runtime import ReplayEnvironment, run_episode


def order_for_method(method, static_order):
    """Only static policies receive the training-fitted prefix.

    ``fixed`` intentionally uses run_episode's canonical schema order. Passing
    the fitted prefix to every method makes fixed and static identical while
    incorrectly presenting them as separate baselines.
    """
    return static_order if method in {"static", "static_value"} else None


def verify_expected_grid_binding(expected, binding, seed):
    """Fail before replay if a canonical-fixed rerun changes the fitted system."""
    if expected.get("split") != "validation" or expected.get("mode") != "offline_replay":
        raise ValueError("expected grid must be validation offline replay")
    if expected.get("seed") != seed:
        raise ValueError("expected grid seed differs from current run")
    old = expected.get("source_binding")
    if not isinstance(old, dict) or "static_order" not in old or "static_order" in binding:
        raise ValueError("expected grid must have fitted static order; canonical replay must not")
    keys = ("merged_receipt_sha256", "responder_receipt_sha256",
            "cache_manifest_sha256", "outer_sources", "plan_binding",
            "head_component_sha256", "controller_component_sha256",
            "responder_checkpoint_sha256")
    if any(key not in old or key not in binding or old[key] != binding[key] for key in keys):
        raise ValueError("canonical fixed replay does not match historical fitted system")
    policies = expected.get("policies", {})
    shared = [key for key in policies if key.startswith("fixed_K")
              and "static_K" + key[len("fixed_K"):] in policies]
    if not shared or any(
            not policies[key].get("group_metrics") or
            {k: v for k, v in policies[key].items() if k != "method"} !=
            {k: v for k, v in policies["static_K" + key[len("fixed_K"):]].items()
             if k != "method"}
            for key in shared):
        raise ValueError("expected historical fixed/static alias not verified")
    order = old["static_order"].get("order")
    if not isinstance(order, list) or order == list(range(len(order))):
        raise ValueError("expected fitted static order must be noncanonical")
    return {"purpose": "distinct_canonical_fixed_validation_replay",
            "historical_fixed_static_alias_verified": True,
            "shared_aliased_budgets": sorted(int(key[len("fixed_K"):]) for key in shared)}


def verify_plan_files_unchanged(binding, planned_dir):
    """Check the exact plan artifacts without depending on package-version API."""
    plan = binding.get("plan_binding")
    if not isinstance(plan, dict):
        raise ValueError("source binding is missing outer plan hashes")
    for name, key in (("plan.json", "plan_sha256"),
                      ("fold_manifest.jsonl", "fold_manifest_sha256")):
        path = Path(planned_dir) / name
        if not path.is_file() or file_hash(path) != plan.get(key):
            raise ValueError("outer plan changed during budget-grid evaluation: " + str(path))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--planned", required=True)
    parser.add_argument("--merged", required=True)
    parser.add_argument("--responder", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", default=["stop", "fixed", "random", "static"])
    parser.add_argument("--static-order-dir")
    parser.add_argument("--cost-weight", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--responder-source-dir")
    parser.add_argument("--expected-grid", help="Source grid whose fitted system must exactly match this canonical fixed-only replay")
    return parser.parse_args()


def main():
    args = parse_args()
    prepared = Path(args.prepared)
    planned = Path(args.planned)
    merged = Path(args.merged)
    responder = Path(args.responder)
    cache = Path(args.cache)
    static_order_dir = Path(args.static_order_dir) if args.static_order_dir else None
    out = Path(args.out)
    if out.exists():
        raise ValueError("output must be new: " + str(out))
    methods = list(args.methods)
    allowed = {"stop", "all", "fixed", "random", "static",
               "value", "value_singleton", "static_value"}
    if not methods or len(set(methods)) != len(methods) or set(methods) - allowed:
        raise ValueError("unsupported, empty or duplicate methods")
    if set(methods) & {"static", "static_value"} and static_order_dir is None:
        raise ValueError("static requires --static-order-dir")
    if args.expected_grid and (methods != ["fixed"] or static_order_dir is not None):
        raise ValueError("--expected-grid requires canonical fixed-only replay without a static order")
    source_before = code_fingerprint()
    evaluator_before = file_hash(__file__)
    print(f"[budget-grid] loading sources once for budgets={args.budgets} methods={methods}", flush=True)
    schema, rows, config, head, controller, binding = _load_sources(
        prepared, planned, merged, responder, cache, args.device,
        static_order_dir, args.responder_source_dir)
    expected_source = None
    if args.expected_grid:
        expected_path = Path(args.expected_grid)
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        expected_source = verify_expected_grid_binding(expected, binding, config["seed"])
        expected_source.update(path=str(expected_path), sha256=file_hash(expected_path))
    print(f"[budget-grid] loaded rows={len(rows)} groups={schema.num_groups}", flush=True)
    if set(methods) & {"value", "value_singleton", "static_value"} and controller.objective != "value":
        raise ValueError("value policies require a value-objective merged controller")
    config["device"] = args.device
    config = resolve_config(config)
    if args.cost_weight is not None:
        config["policy"]["cost_weight"] = args.cost_weight
    cost = DeclaredCost(**config["cost"])
    include_pairs = config["learning"]["include_pairs"]
    include_all = config["learning"]["include_all"]
    static_order = (binding["static_order"]["order"]
                    if set(methods) & {"static", "static_value"} else None)
    traces, reports, settings = [], {}, {}
    for budget in args.budgets:
        if not 0 <= budget <= schema.num_groups:
            raise ValueError("budget outside valid group range")
        for method in methods:
            if method == "all" and budget != schema.num_groups:
                continue
            key = f"{method}_K{budget}"
            spec = {"method": method, "budget": budget, "config": config,
                    "mode": "offline_replay", "split": "validation",
                    "source_binding": binding}
            spec["system_hash"] = stable_hash(spec)
            settings[key] = spec
            print(f"[budget-grid] replay start {key}", flush=True)
            current = []
            for row_index, row in enumerate(rows, 1):
                trace = run_episode(ReplayEnvironment(row["z"], schema), schema, head,
                    method=method, controller=controller, cost=cost,
                    cost_weight=config["policy"]["cost_weight"], max_groups=budget,
                    max_cost=config["policy"]["max_cost"], include_pairs=include_pairs,
                    include_all=include_all, pairs=controller.pairs,
                    order=order_for_method(method, static_order),
                    seed=config["seed"], device=args.device)
                trace.update(sample_id=row["sample_id"], group_id=row["group_id"],
                             y=row["y"], split="validation", policy_id=key,
                             system_hash=spec["system_hash"], seed=config["seed"],
                             num_query_groups=schema.num_groups,
                             num_atoms=schema.num_atoms, budget_groups=budget)
                current.append(trace)
                if row_index % 100 == 0 or row_index == len(rows):
                    print(f"[budget-grid] replay progress {key} "
                          f"{row_index}/{len(rows)}", flush=True)
            traces.extend(current)
            reports[key] = summarize_traces(current, num_classes=schema.num_classes)
            reports[key]["budget_groups"] = budget
            print(f"[budget-grid] replay done {key} "
                  f"acc={reports[key]['accuracy']:.6f} "
                  f"macro_f1={reports[key]['macro_f1']:.6f} "
                  f"mean_groups={reports[key]['mean_queried_groups']:.3f}",
                  flush=True)
    out = fresh_dir(out)
    write_jsonl(out / "traces.jsonl", traces)
    write_json(out / "settings.json", settings)
    write_json(out / "metrics.json", {"split": "validation", "mode": "offline_replay",
        "seed": config["seed"], "budgets": args.budgets, "methods": methods,
        "policies": reports, "paper_evidence": False,
        "evidence_status": "OFFLINE_VALIDATION_DEVELOPMENT_NOT_PAPER_OR_LATENCY_EVIDENCE",
        "source_binding": binding, "source_code_hash": source_before,
        "evaluator_script_sha256": evaluator_before,
        "expected_grid_source": expected_source})
    # The deployed server package can lag the local package API. Verify the plan
    # hashes here, then use the stable helper signature shared by both versions.
    _verify_binding_files_unchanged(binding, merged, responder, cache, static_order_dir)
    verify_plan_files_unchanged(binding, planned)
    if code_fingerprint() != source_before or file_hash(__file__) != evaluator_before:
        raise ValueError("source or evaluator code changed during budget-grid evaluation")
    receipt = {"format": "cbmjev-crossfit-budget-grid-v1", "status": "COMPLETE",
               "paper_evidence": False,
               "evidence_status": "OFFLINE_VALIDATION_DEVELOPMENT_NOT_PAPER_OR_LATENCY_EVIDENCE",
               "source_binding": binding, "source_code_hash": source_before,
               "evaluator_script_sha256": evaluator_before,
               "samples_per_policy": len(rows), "num_policies": len(reports),
               "budgets": args.budgets, "methods": methods,
               "expected_grid_source": expected_source}
    receipt["receipt_hash"] = stable_hash(receipt)
    write_json(out / "receipt.json", receipt)
    print(f"[budget-grid] complete out={out} policies={len(reports)}", flush=True)


if __name__ == "__main__":
    main()

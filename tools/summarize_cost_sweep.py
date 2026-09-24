#!/usr/bin/env python3
"""Summarize validation cost sweeps without mixing sample and seed uncertainty."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def load_point(directory):
    directory = Path(directory)
    metrics_bytes = (directory / "metrics.json").read_bytes()
    settings_bytes = (directory / "settings.json").read_bytes()
    metrics = json.loads(metrics_bytes)
    settings = json.loads(settings_bytes)
    if metrics.get("split") != "validation":
        raise ValueError(f"cost sweep accepts validation only: {directory}")
    if set(metrics["policies"]) != set(settings):
        raise ValueError(f"metrics/settings policy mismatch: {directory}")
    if type(metrics["seed"]) is not int or metrics["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    expected_status = {"offline_replay": "OFFLINE_REPLAY_NOT_LATENCY",
                       "live": "LIVE_SINGLE_RUN_REQUIRES_REPLICATION"}
    if metrics.get("evidence_status") != expected_status.get(metrics.get("mode")) or metrics.get("mode") not in expected_status:
        raise ValueError("invalid mode/evidence status")
    rows = []
    for policy_id, report in metrics["policies"].items():
        config = settings[policy_id]["config"]
        if settings[policy_id].get("mode") != metrics["mode"]:
            raise ValueError("settings/metrics mode mismatch")
        if "seed" in config and config["seed"] != metrics["seed"]:
            raise ValueError("settings/metrics seed mismatch")
        rows.append({
            "run": str(directory),
            "seed": int(metrics["seed"]),
            "mode": metrics["mode"],
            "policy_id": policy_id,
            "method": settings[policy_id]["method"],
            "cost_weight": _finite(config["policy"]["cost_weight"], "cost_weight"),
            "max_groups": config["policy"]["max_groups"],
            "accuracy": _finite(report["accuracy"], "accuracy"),
            "macro_f1": _finite(report["macro_f1"], "macro_f1"),
            "group_mean_risk": _finite(report["group_mean_risk"], "group_mean_risk"),
            "mean_queried_groups": _finite(report["mean_queried_groups"], "mean_queried_groups"),
            "mean_calls": _finite(report["mean_calls"], "mean_calls"),
            "mean_declared_cost": _finite(report["mean_declared_cost"], "mean_declared_cost"),
            "num_samples": int(report["num_samples"]),
            "num_groups": int(report["num_groups"]),
            "evidence_status": metrics["evidence_status"],
            "source_metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
            "source_settings_sha256": hashlib.sha256(settings_bytes).hexdigest(),
            "schema_hash": metrics.get("schema_hash"),
            "source_revision": metrics.get("source_revision"),
            "training_source_code_hash": metrics.get("training_source_code_hash"),
            "comparison_config": {key: value for key, value in config.items()
                                  if key not in ("seed", "device")},
        })
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["cost_weight"], row["max_groups"])].append(row)
    summaries = []
    for (method, weight, max_groups), items in sorted(grouped.items()):
        for field in ("mode", "policy_id", "num_samples", "num_groups", "evidence_status",
                      "schema_hash", "source_revision", "training_source_code_hash",
                      "comparison_config"):
            if any(item.get(field) != items[0].get(field) for item in items[1:]):
                raise ValueError(f"incomparable sweep inputs: {field}")
        seeds = [item["seed"] for item in items]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seed for {method} at cost weight {weight}, budget {max_groups}")
        result = {"method": method, "cost_weight": weight, "max_groups": max_groups,
                  "num_seeds": len(items), "seeds": seeds}
        for metric in ("accuracy", "macro_f1", "group_mean_risk", "mean_queried_groups",
                       "mean_calls", "mean_declared_cost"):
            values = [item[metric] for item in items]
            result[metric + "_mean"] = statistics.mean(values)
            result[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        summaries.append(result)
    return summaries


def write_summary(run_dirs, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    rows = [row for directory in run_dirs for row in load_point(directory)]
    if not rows:
        raise ValueError("at least one sweep run is required")
    aggregate = summarize(rows)
    out.mkdir(parents=True)
    fields = list(rows[0])
    with (out / "points.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    with (out / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump({
            "format": "cbmjev-validation-cost-sweep-v1",
            "split": "validation",
            "provenance_scope": "Hashes bind source metric/config bytes; missing historical metadata remains unknown. Equal counts do not prove identical sample identities.",
            "points": rows,
            "aggregate": aggregate,
            "uncertainty": "sample SD across distinct training seeds; validation development evidence only",
            "paper_claim": False,
        }, stream, indent=2)
        stream.write("\n")
    return rows, aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows, aggregate = write_summary(args.runs, args.out)
    print(json.dumps({"out": str(Path(args.out).resolve()), "points": len(rows),
                      "aggregate_rows": len(aggregate)}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Quantify CUB gold-attribute observability before oracle diagnostics.

This is a data-coverage audit, not a model evaluation. It does not inspect
test labels or train/tune a policy. Missing gold values are never imputed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from cbmjev.io import file_hash


def summary(values):
    if not values:
        raise ValueError("empty split")
    ordered = sorted(values)
    n = len(ordered)
    return {"mean": sum(ordered) / n, "min": ordered[0],
            "p10": ordered[int(.10 * (n - 1))],
            "median": ordered[int(.50 * (n - 1))],
            "p90": ordered[int(.90 * (n - 1))], "max": ordered[-1]}


def audit(prepared, splits):
    prepared = Path(prepared)
    schema = json.loads((prepared / "schema.json").read_text(encoding="utf-8"))
    concepts, groups = schema["concepts"], schema["groups"]
    atoms = [tuple(group["atoms"]) for group in groups]
    if (not concepts or not groups or
            sorted(a for group in atoms for a in group) != list(range(len(concepts)))):
        raise ValueError("groups must partition all concept atoms")
    state = {split: {"cases": 0, "target_status": Counter(),
                     "annotation_status": Counter(),
                     "observed_atoms": [], "complete_groups": [],
                     "group_complete_cases": [0] * len(groups),
                     "all_groups_complete_cases": 0,
                     "target_counts": Counter(),
                     "complete_target_counts": Counter(),
                     "group_ids": set(), "sample_ids": set()}
             for split in splits}
    with (prepared / "samples.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            split = row["split"]
            if split not in state:
                continue
            item = state[split]
            if row["sample_id"] in item["sample_ids"]:
                raise ValueError("duplicate sample ID")
            item["sample_ids"].add(row["sample_id"])
            item["group_ids"].add(row["group_id"])
            labels = row["concepts"]
            if len(labels) != len(concepts):
                raise ValueError("concept width mismatch")
            observed = []
            for index, (label, concept) in enumerate(zip(labels, concepts)):
                if label["concept_id"] != concept["id"]:
                    raise ValueError(f"concept order mismatch at {index}")
                status = label["annotation_status"]
                item["annotation_status"][status] += 1
                good = status == "OBSERVED"
                if good != (type(label["value"]) is int):
                    raise ValueError("gold value/status mismatch")
                observed.append(good)
            complete = [all(observed[a] for a in atom_ids) for atom_ids in atoms]
            item["cases"] += 1
            item["target_status"][row["target"]["status"]] += 1
            if row["target"]["status"] == "OBSERVED":
                value = row["target"]["value"]
                if type(value) is not int or not 0 <= value < schema["num_classes"]:
                    raise ValueError("invalid target class")
                item["target_counts"][value] += 1
                if all(complete):
                    item["complete_target_counts"][value] += 1
            item["observed_atoms"].append(sum(observed))
            item["complete_groups"].append(sum(complete))
            item["all_groups_complete_cases"] += int(all(complete))
            for index, covered in enumerate(complete):
                item["group_complete_cases"][index] += int(covered)
    report = {"format": "cbmjev-cub-gold-coverage-v1",
        "purpose": "oracle-concept-feasibility-data-coverage-not-model-evaluation",
        "prepared": str(prepared),
        "analysis_script_sha256": file_hash(__file__),
        "schema_sha256": file_hash(prepared / "schema.json"),
        "prepared_audit_sha256": file_hash(prepared / "audit.json"),
        "num_atoms": len(concepts), "num_groups": len(groups), "splits": {}}
    for split, item in state.items():
        n = item["cases"]
        if not n:
            raise ValueError(f"no {split} rows")
        total_targets = sum(item["target_counts"].values())
        complete_targets = sum(item["complete_target_counts"].values())
        tv = None
        if total_targets and complete_targets:
            tv = .5 * sum(abs(item["target_counts"][y] / total_targets
                                 - item["complete_target_counts"][y] / complete_targets)
                            for y in range(schema["num_classes"]))
        report["splits"][split] = {
            "cases": n, "distinct_sample_ids": len(item["sample_ids"]),
            "distinct_group_ids": len(item["group_ids"]),
            "target_status": dict(item["target_status"]),
            "annotation_status": dict(item["annotation_status"]),
            "observed_atoms_per_case": summary(item["observed_atoms"]),
            "complete_groups_per_case": summary(item["complete_groups"]),
            "all_groups_complete_cases": item["all_groups_complete_cases"],
            "all_groups_complete_fraction": item["all_groups_complete_cases"] / n,
            "observed_target_classes": len(item["target_counts"]),
            "complete_case_target_classes": len(item["complete_target_counts"]),
            "target_class_total_variation_all_vs_complete": tv,
            "group_complete_fraction": {
                group["id"]: count / n for group, count in zip(groups,
                                                   item["group_complete_cases"])}}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("train", "validation"))
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refuse to overwrite existing report")
    result = audit(args.prepared, tuple(args.splits))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps({"out": str(args.out), "num_atoms": result["num_atoms"],
        "num_groups": result["num_groups"],
        "split_summary": {k: {"cases": v["cases"],
            "mean_observed_atoms": v["observed_atoms_per_case"]["mean"],
            "mean_complete_groups": v["complete_groups_per_case"]["mean"],
            "all_groups_complete_fraction": v["all_groups_complete_fraction"],
            "complete_case_target_classes": v["complete_case_target_classes"],
            "target_class_tv": v["target_class_total_variation_all_vs_complete"]}
            for k, v in result["splits"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()

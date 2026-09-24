#!/usr/bin/env python3
"""Compare validation semantic audits on exactly the same gold population."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbmjev.io import file_hash, write_json


def compare(before, after):
    for report in (before, after):
        if (report.get("format") != "cbmjev-semantic-response-audit-v1"
                or report.get("status") != "COMPLETED"
                or report.get("split") != "validation" or report.get("test_evaluated") is not False):
            raise ValueError("completed validation audits only")
    for key in ("dataset", "schema_hash"):
        if not before.get(key) or before[key] != after.get(key):
            raise ValueError("different or missing " + key)
    for key in ("prepared_samples_sha256", "membership_sha256", "prepared_schema_file_sha256"):
        if not before["source_hashes"].get(key) or before["source_hashes"][key] != after["source_hashes"].get(key):
            raise ValueError("different or missing population source: " + key)
    for key in ("audited_sample_ids_hash", "audited_cached_split_cases"):
        if not before["selection"].get(key) or before["selection"][key] != after["selection"].get(key):
            raise ValueError("different or missing audit population: " + key)
    a = {c["concept_id"]: c for c in before["concepts"]}
    b = {c["concept_id"]: c for c in after["concepts"]}
    if not a or set(a) != set(b) or len(a) != len(before["concepts"]) or len(b) != len(after["concepts"]):
        raise ValueError("concept IDs differ or are duplicated")
    rows = []
    fields = ("accuracy_runtime_nonanswers_count_as_errors", "macro_f1_fixed_semantic_classes",
              "coverage_semantic_answers_on_observed_gold", "balanced_accuracy_supported_semantic_classes")
    for cid in a:
        x, y = a[cid], b[cid]
        if (x["per_class_gold_support"] != y["per_class_gold_support"]
                or x["response_column_labels"] != y["response_column_labels"]):
            raise ValueError("gold support or semantic order differs: " + cid)
        metrics = {}
        for name in fields:
            old, new = x[name], y[name]
            if (old is None) != (new is None):
                raise ValueError("metric support differs: " + cid)
            metrics[name] = {"before": old, "after": new,
                             "delta": None if old is None else new - old}
        recalls = []
        for index, support in enumerate(x["per_class_gold_support"]):
            old = x["per_class_recall_missing_support_null"][index]
            new = y["per_class_recall_missing_support_null"][index]
            if (support == 0) != (old is None) or (support == 0) != (new is None):
                raise ValueError("recall support inconsistent: " + cid)
            recalls.append({"class": x["response_column_labels"][index], "support": support,
                            "before": old, "after": new, "delta": None if not support else new - old})
        rows.append({"concept_id": cid, "all_classes_supported": all(x["per_class_gold_support"]),
                     "metrics": metrics, "recalls": recalls})
    summary = {}
    for subset, selected in (("observed_support", rows),
                             ("all_classes_supported", [r for r in rows if r["all_classes_supported"]])):
        summary[subset] = {}
        for field in fields:
            supported = [r["metrics"][field] for r in selected if r["metrics"][field]["before"] is not None]
            summary[subset][field] = {"concept_count": len(supported), **{
                k: statistics.mean(r[k] for r in supported) if supported else None
                for k in ("before", "after", "delta")}}
    class_recalls = {}
    for label in sorted({c["class"] for row in rows for c in row["recalls"]}):
        supported = [c for row in rows for c in row["recalls"] if c["class"] == label and c["support"]]
        class_recalls[label] = {"supported_concept_count": len(supported),
                               **{key: statistics.mean(c[key] for c in supported) if supported else None
                                  for key in ("before", "after", "delta")},
                               "zero_recall_before": sum(c["before"] == 0 for c in supported),
                               "zero_recall_after": sum(c["after"] == 0 for c in supported)}
    return {"scope": "VALIDATION_DESCRIPTIVE_COMPARISON_NOT_TEST_OR_SIGNIFICANCE",
            "limitation": "Aggregate confusion matrices do not identify paired errors; no paired confidence intervals.",
            "dataset": before["dataset"], "selection": before["selection"],
            "summary": summary, "class_recall_summary_equal_concept_weight": class_recalls, "concepts": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    report = compare(json.loads(Path(args.before).read_text()), json.loads(Path(args.after).read_text()))
    report["audit_sha256"] = {"before": file_hash(args.before), "after": file_hash(args.after)}
    write_json(args.out, report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

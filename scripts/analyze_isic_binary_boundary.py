#!/usr/bin/env python3
"""Report concept and task confusion for an ISIC binary validation pilot."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def confusion_summary(pairs):
    counts = Counter(pairs)
    tp, fp, tn, fn = (counts[1, 1], counts[0, 1],
                      counts[0, 0], counts[1, 0])
    n = tp + fp + tn + fn
    if not n:
        raise ValueError("empty binary confusion")
    if n != sum(counts.values()):
        raise ValueError("non-binary confusion value")
    return {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "gold_positive": tp + fn, "predicted_positive": tp + fp,
            "accuracy": (tp + tn) / n,
            "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2
            if (tp + fn) and (tn + fp) else None}


def analyze(prepared, cache, traces):
    prepared, cache, traces = map(Path, (prepared, cache, traces))
    schema = json.loads((prepared / "schema.json").read_text(encoding="utf-8"))
    if schema["num_classes"] != 2:
        raise ValueError("binary task required")
    samples = {}
    for row in read_jsonl(prepared / "samples.jsonl"):
        sample_id = row["sample_id"]
        if sample_id in samples:
            raise ValueError("duplicate prepared sample")
        samples[sample_id] = row
    validation = {}
    for row in read_jsonl(cache / "responses.jsonl"):
        if row["split"] != "validation":
            continue
        sample_id = row["sample_id"]
        if sample_id in validation or sample_id not in samples:
            raise ValueError("duplicate or unknown validation response")
        if len(row["z"]) != len(schema["concepts"]):
            raise ValueError("response width mismatch")
        validation[sample_id] = row
    if not validation:
        raise ValueError("no validation responses")
    concepts = {}
    for index, concept in enumerate(schema["concepts"]):
        pairs = []
        for sample_id, row in validation.items():
            annotation = samples[sample_id]["concepts"][index]
            if annotation["concept_id"] != concept["id"]:
                raise ValueError("concept order mismatch")
            pairs.append((annotation["value"], row["z"][index]))
        concepts[concept["id"]] = confusion_summary(pairs)
    policies = defaultdict(list)
    for row in read_jsonl(traces):
        sample_id = row["sample_id"]
        if sample_id not in validation or row["split"] != "validation":
            raise ValueError("task trace does not match validation cache")
        if row["y"] != validation[sample_id]["y"]:
            raise ValueError("task target mismatch")
        policies[row["method"]].append(row)
    policy_results = {}
    for name, rows in sorted(policies.items()):
        if len(rows) != len(validation) or len({r["sample_id"] for r in rows}) != len(validation):
            raise ValueError("policy validation sample coverage mismatch")
        result = confusion_summary((r["y"], r["prediction"]) for r in rows)
        result["mean_queried_groups"] = sum(len(r["queried_groups"]) for r in rows) / len(rows)
        policy_results[name] = result
    return {"format": "cbmjev-isic-binary-boundary-diagnostic-v1",
            "evidence_status": "OFFLINE_VALIDATION_DIAGNOSTIC_NOT_PAPER_EVIDENCE",
            "prepared": str(prepared), "cache": str(cache), "traces": str(traces),
            "validation_samples": len(validation), "concepts": concepts,
            "policies": policy_results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise ValueError("output already exists")
    report = analyze(args.prepared, args.cache, args.traces)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, values in report["concepts"].items():
        print(name, "gold+", values["gold_positive"], "pred+",
              values["predicted_positive"], "balanced_acc",
              round(values["balanced_accuracy"], 3))
    for name, values in report["policies"].items():
        print(name, "accuracy", round(values["accuracy"], 3),
              "balanced_acc", round(values["balanced_accuracy"], 3),
              "queried", round(values["mean_queried_groups"], 3))


if __name__ == "__main__":
    main()

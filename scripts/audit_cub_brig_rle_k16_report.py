#!/usr/bin/env python3
"""Independent read-only trace check for the frozen BRiG-RLE K16 report.

Run from the server project root after the four seed evaluations complete.
This does not re-evaluate or alter any policy.
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path("runs")
REPORT = Path("results/diagnostics/cub_brig_rle_k16_gate_60_63_v1.json")
SEEDS = (60, 61, 62, 63)
N_CLASSES = 200
N_IMAGES = 594


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def selected_rows(path, policy):
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["policy_id"] != policy:
                continue
            sample_id = row["sample_id"]
            if sample_id in rows:
                raise ValueError(f"duplicate {policy} sample {sample_id}")
            groups = row["queried_groups"]
            if (row["split"] != "validation" or row["mode"] != "offline_replay"
                    or len(groups) != 16 or len(set(groups)) != 16):
                raise ValueError(f"invalid K16 trace for {policy} sample {sample_id}")
            rows[sample_id] = (row["group_id"], row["y"], row["prediction"])
    if len(rows) != N_IMAGES:
        raise ValueError(f"{policy}: expected {N_IMAGES} rows, got {len(rows)}")
    return rows


def metrics(rows):
    confusion = Counter((y, prediction) for _, y, prediction in rows.values())
    accuracy = sum(confusion[(label, label)] for label in range(N_CLASSES)) / N_IMAGES
    f1 = []
    for label in range(N_CLASSES):
        tp = confusion[(label, label)]
        fp = sum(confusion[(other, label)] for other in range(N_CLASSES) if other != label)
        fn = sum(confusion[(label, other)] for other in range(N_CLASSES) if other != label)
        denominator = 2 * tp + fp + fn
        f1.append(2 * tp / denominator if denominator else 0.0)
    return accuracy, statistics.mean(f1)


def close(actual, expected):
    return math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10)


def main():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    unsigned = dict(report)
    recorded_hash = unsigned.pop("report_hash")
    canonical = json.dumps(unsigned, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), allow_nan=False).encode("utf-8")
    if (hashlib.sha256(canonical).hexdigest() != recorded_hash
            or report["seeds"] != list(SEEDS)
            or report["samples_per_seed"] != N_IMAGES
            or report["budget_groups"] != 16):
        raise ValueError("invalid frozen gate report")
    reference_images = None
    checked = []
    for row in report["per_seed"]:
        seed = row["seed"]
        if seed not in SEEDS:
            raise ValueError(f"unexpected seed {seed}")
        brig_path = ROOT / f"cub_brig_rle_seed{seed}_eval_K16_v1/traces.jsonl"
        static_path = ROOT / f"cub_eval_value_budget_grid_seed{seed}_v1/traces.jsonl"
        if (digest(brig_path) != row["brig_traces_sha256"]
                or digest(static_path) != row["static_traces_sha256"]):
            raise ValueError(f"seed {seed}: trace hash changed")
        brig = selected_rows(brig_path, "brig_mean_q_K16")
        static = selected_rows(static_path, "static_K16")
        if brig.keys() != static.keys():
            raise ValueError(f"seed {seed}: policy sample IDs differ")
        image_labels = {key: (value[0], value[1]) for key, value in brig.items()}
        if reference_images is None:
            reference_images = image_labels
        elif reference_images != image_labels:
            raise ValueError(f"seed {seed}: validation identity/labels differ")
        for key in brig:
            if brig[key][:2] != static[key][:2]:
                raise ValueError(f"seed {seed}: paired identity/label mismatch")
        brig_accuracy, brig_f1 = metrics(brig)
        static_accuracy, static_f1 = metrics(static)
        if not all((close(brig_accuracy, row["brig_accuracy"]),
                    close(static_accuracy, row["static_accuracy"]),
                    close(brig_f1, row["brig_macro_f1"]),
                    close(static_f1, row["static_macro_f1"]))):
            raise ValueError(f"seed {seed}: reported metrics disagree with traces")
        checked.append({"seed": seed, "accuracy_delta": brig_accuracy - static_accuracy,
                        "macro_f1_delta": brig_f1 - static_f1, "images": N_IMAGES})
    if [row["seed"] for row in checked] != list(SEEDS):
        raise ValueError("seed order or coverage mismatch")
    if (not close(statistics.mean(row["accuracy_delta"] for row in checked),
                  report["mean_accuracy_delta"])
            or not close(statistics.mean(row["macro_f1_delta"] for row in checked),
                         report["mean_macro_f1_delta"])):
        raise ValueError("summary differs from independent trace metrics")
    print(json.dumps({"status": "PASS", "report_sha256": digest(REPORT),
                      "same_validation_images_across_seeds": True,
                      "seeds": checked}, indent=2))


if __name__ == "__main__":
    main()

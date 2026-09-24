#!/usr/bin/env python3
"""Prepare local Task2 masks + Archive JPEGs/metadata without changing raw data.

This is a derived classification benchmark, NOT the official challenge test.
Only IDs and exact decoded pixels determine identity components. Metadata is
audit/target material and is never a model input. Requires Pillow, no network.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbmjev.data import _concept, _file_hash, _hash, _sample, _schema

ATTRIBUTES = ("globules", "milia_like_cyst", "negative_network", "pigment_network", "streaks")
TARGETS = ("Benign", "Indeterminate", "Malignant")
DATASET = "isic2018_task2"
PROTOCOL = "isic-task2-known-identity-exact-pixels-70-10-10-10-v1"
BINARY_PROTOCOL = "isic-task2-binary-known-identity-exact-pixels-70-10-10-10-v1"


def assign(samples, seed, protocol=PROTOCOL):
    """Hash-ordered group stratification; floor 10% each test/val/cert.

    Stratification uses the full sorted set of target classes within a component.
    Tiny strata remain in training; no class-aware post-hoc reshuffling.
    """
    grouped = defaultdict(list)
    for row in samples:
        grouped[row["group_id"]].append(row)
    strata = defaultdict(list)
    for group, rows in grouped.items():
        strata[tuple(sorted({r["target"]["value"] for r in rows}))].append(group)
    def ordered(groups, purpose):
        return sorted(groups, key=lambda g: (hashlib.sha256(
            (protocol + "|" + str(seed) + "|" + purpose + "|" + g).encode()).hexdigest(), g))
    membership = []
    for stratum, groups in sorted(strata.items()):
        groups = ordered(groups, "outer")
        n = len(groups) // 10
        outer_groups = {"test": groups[:n], "validation": groups[n:2*n],
                        "calibration": groups[2*n:3*n], "train": groups[3*n:]}
        for outer, ids in outer_groups.items():
            ids = ordered(ids, "train-roles")
            a, b = int(len(ids) * .6), int(len(ids) * .8)
            for rank, group in enumerate(ids):
                role = outer if outer != "train" else (
                    "responder_fit" if rank < a else "head_fit" if rank < b else "policy_fit")
                for row in grouped[group]:
                    row["split"] = outer
                    row["fold_id"] = rank % 3 if outer == "train" else None
                    membership.append({"sample_id": row["sample_id"], "group_id": group,
                                       "split": role, "outer_split": outer})
    return sorted(membership, key=lambda r: r["sample_id"])


def prepare(source, metadata_path, out, seed=17, expected_ids=2594, reuse_prepared=None,
            target_mode="three_class", require_complete=False):
    from PIL import Image
    if target_mode not in ("three_class", "binary"):
        raise ValueError("target_mode must be three_class or binary")
    targets = ("Benign", "Malignant") if target_mode == "binary" else TARGETS
    dataset = DATASET + "_binary" if target_mode == "binary" else DATASET
    protocol = BINARY_PROTOCOL if target_mode == "binary" else PROTOCOL
    source, metadata_path, out = (Path(p).resolve() for p in (source, metadata_path, out))
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("refusing nonempty output: " + str(out))
    if source == out or source in out.parents:
        raise ValueError("prepared output must be outside raw source")
    records = json.loads(metadata_path.read_text())["records"]
    verified_cache, reuse_evidence = {}, None
    if reuse_prepared is not None:
        reuse = Path(reuse_prepared).resolve()
        old_audit = json.loads((reuse / "audit.json").read_text())
        old_manifest = [json.loads(line) for line in (reuse / "source_manifest.jsonl").read_text().splitlines()]
        if (old_audit.get("raw_manifest_hash") != _hash(old_manifest)
                or old_audit.get("source_root") != str(source)
                or old_audit.get("metadata_snapshot", {}).get("sha256") != _file_hash(metadata_path)
                or old_audit.get("checks", {}).get("image_and_mask_decode") != "CHECKED"
                or old_audit.get("image_mask_dimension_mismatch_count") != 0):
            raise ValueError("reuse audit/manifest not a compatible verified snapshot")
        verified_cache = {r["isic_id"]: r for r in old_manifest}
        reuse_evidence = {"prepared": str(reuse), "audit_sha256": _file_hash(reuse / "audit.json"),
                          "manifest_sha256": _file_hash(reuse / "source_manifest.jsonl"), "reused_ids": 0}
    mask_root = source / "groundtruth_extracted_v1/ISIC2018_Task2_Training_GroundTruth_v3"
    masks = defaultdict(dict)
    for path in sorted(mask_root.glob("*.png")):
        match = re.fullmatch(r"(ISIC_\d{7})_attribute_(.+)\.png", path.name)
        if not match or match[2] not in ATTRIBUTES:
            raise ValueError("unknown mask filename: " + path.name)
        masks[match[1]][match[2]] = path
    if len(masks) != expected_ids:
        raise ValueError("mask ID count {}, expected {}".format(len(masks), expected_ids))
    if any(set(v) != set(ATTRIBUTES) for v in masks.values()):
        raise ValueError("incomplete mask set (missing is not absence)")
    parent = {sid: sid for sid in masks}
    # Availability is snapshotted once; a concurrent downloader cannot change
    # inclusion halfway through decoding. Newly completed files need a new out.
    available_images = {p.stem for p in (source / "images").glob("*.jpg") if p.is_file()}
    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    seen_identity, seen_pixels = {}, {}
    coverage, licenses = Counter(), Counter()
    # Include identity bridges even when that case's image/target is unavailable.
    for sid in sorted(masks):
        record = records.get(sid)
        if not record:
            continue
        if record.get("isic_id") != sid:
            raise ValueError("metadata ID join mismatch: " + sid)
        clinical = record.get("metadata", {}).get("clinical", {})
        for field in ("patient_id", "lesion_id"):
            value = clinical.get(field)
            if value is not None and str(value).strip():
                coverage[field] += 1
                key = (field, str(value))
                if key in seen_identity:
                    union(sid, seen_identity[key])
                seen_identity[key] = sid
    samples, exclusions, manifest = [], [], []
    duplicate_pairs, dimensions_differ = [], []
    for sid in sorted(masks):
        record = records.get(sid)
        image_path = source / "images" / (sid + ".jpg")
        if not record or sid not in available_images:
            exclusions.append({"source_id": sid, "reason": "missing_metadata" if not record else "missing_image"})
            continue
        if record.get("isic_id") != sid:
            raise ValueError("metadata ID join mismatch: " + sid)
        clinical = record.get("metadata", {}).get("clinical", {})
        target = clinical.get("diagnosis_1")
        if target is None or target == "":
            exclusions.append({"source_id": sid, "reason": "missing_target"})
            continue
        if target not in TARGETS:
            raise ValueError("unknown diagnosis_1: " + repr(target))
        licenses[str(record.get("copyright_license"))] += 1
        cached = verified_cache.get(sid)
        if cached:
            if (cached["image_sha256"] != _file_hash(image_path)
                    or cached["metadata_record_sha256"] != _hash(record)
                    or any(cached["masks"][attr]["sha256"] != _file_hash(masks[sid][attr]) for attr in ATTRIBUTES)):
                raise ValueError("raw files changed since verified decode: " + sid)
            dimensions, pixel_hash = cached["image_size"], cached["decoded_rgb_sha256"]
            reuse_evidence["reused_ids"] += 1
        else:
            with Image.open(image_path) as image:
                image.load()
                rgb = image.convert("RGB")
                dimensions = list(rgb.size)
                pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        if pixel_hash in seen_pixels:
            other = seen_pixels[pixel_hash]
            duplicate_pairs.append([other, sid])
            union(sid, other)
        seen_pixels[pixel_hash] = sid
        concepts, raw_masks = [], {}
        for attr in ATTRIBUTES:
            path = masks[sid][attr]
            if cached:
                value = cached["masks"][attr]["value"]
            else:
                with Image.open(path) as mask:
                    mask.load()
                    if list(mask.size) != dimensions:
                        dimensions_differ.append({"source_id": sid, "attribute": attr,
                                                  "image_size": dimensions, "mask_size": list(mask.size)})
                    value = int(mask.convert("L").getextrema()[1] > 0)
            raw_masks[attr] = {"path": str(path.relative_to(source)), "sha256": _file_hash(path), "value": value}
            concepts.append(_concept("isic2018_" + attr, value, "OBSERVED"))
        raw = {"isic_id": sid, "image_sha256": _file_hash(image_path),
               "metadata_record_sha256": _hash(record), "masks": raw_masks}
        manifest.append(dict(raw, image_bytes=image_path.stat().st_size,
                             image_size=dimensions, decoded_rgb_sha256=pixel_hash))
        # Audit/decode excluded cases too: their identity or exact-pixel links
        # may connect two retained cases and must never be silently discarded.
        if target_mode == "binary" and target == "Indeterminate":
            exclusions.append({"source_id": sid, "reason": "indeterminate_excluded_from_binary_task",
                               "target_class": target, "metadata_record_sha256": raw["metadata_record_sha256"]})
            continue
        samples.append(_sample(dataset, sid, sid, "official_task2_training", raw,
            (targets.index(target), "OBSERVED"), concepts,
            images=[str(image_path.relative_to(source))],
            audit={"isic_id": sid, "target_level": "diagnosis_1", "target_class": target,
                   "metadata_record_sha256": raw["metadata_record_sha256"],
                   "copyright_license": record.get("copyright_license")}))
    if not samples:
        raise ValueError("no usable samples")
    if require_complete and any(x["reason"] != "indeterminate_excluded_from_binary_task" for x in exclusions):
        raise ValueError("complete preparation required; missing source image/metadata/target detected")
    for row in samples:
        row["group_id"] = "isic2018:identity:" + find(row["provenance"]["source_id"])
    membership = assign(samples, seed, protocol)
    # Check actual assignments, not just the intended algorithm.
    for field in ("split", "outer_split"):
        assignments = defaultdict(set)
        for row in membership:
            assignments[row["group_id"]].add(row[field])
        if any(len(v) != 1 for v in assignments.values()):
            raise AssertionError("group crosses " + field)
    schema = _schema(dataset, len(targets), [
        ("isic2018_" + attr, attr.replace("_", " "), ("absent", "present")) for attr in ATTRIBUTES])
    counts = {split: sum(s["split"] == split for s in samples)
              for split in ("train", "validation", "calibration", "test")}
    role_by_id = {m["sample_id"]: m["split"] for m in membership}
    audit = {"schema_version": "cbmjev-data-audit-v1", "status": "PREPARED_NOT_PAPER_ACCEPTANCE",
        "dataset": dataset, "source_root": str(source), "seed": seed, "fold_count": 3,
        "source_revision": protocol, "split_protocol": protocol, "target_mode": target_mode,
        "require_complete": require_complete, "source_audited_images": len(manifest),
        "metadata_snapshot": {"path": str(metadata_path), "sha256": _file_hash(metadata_path)},
        "preparer_sha256": _file_hash(Path(__file__)), "target_classes": list(targets),
        "verified_decode_reuse": reuse_evidence,
        "source_mask_ids": len(masks), "metadata_records": len(records),
        "observed_counts": counts, "effective_task_counts": counts,
        "role_counts": dict(Counter(m["split"] for m in membership)),
        "role_target_counts": {role: dict(Counter(s["audit_metadata"]["target_class"] for s in samples
            if role_by_id[s["sample_id"]] == role))
            for role in ("responder_fit", "head_fit", "policy_fit", "validation", "calibration", "test")},
        "target_counts": {split: dict(Counter(s["audit_metadata"]["target_class"] for s in samples if s["split"] == split)) for split in counts},
        "concept_positive_counts": {attr: sum(s["concepts"][i]["value"] for s in samples) for i, attr in enumerate(ATTRIBUTES)},
        "groups": len({s["group_id"] for s in samples}), "known_identity_coverage": dict(coverage),
        "licenses_from_metadata": dict(licenses), "exact_pixel_duplicate_pairs": duplicate_pairs,
        "image_mask_dimension_mismatch_count": len(dimensions_differ),
        "exclusions_by_reason": dict(Counter(x["reason"] for x in exclusions)),
        "split_hash": _hash(membership), "schema_hash": _hash(schema), "raw_manifest_hash": _hash(manifest),
        "checks": {"image_and_mask_decode": "CHECKED", "ids_and_joins": "CHECKED",
                   "known_identity_cross_split_overlap": 0, "known_identity_cross_role_overlap": 0,
                   "exact_decoded_pixel_cross_split_overlap": 0, "near_duplicate_phash": "NOT_CHECKED",
                   "patient_independence_missing_ids": "NOT_CHECKED", "license_legal_review": "NOT_CHECKED",
                   "model_training_ancestor_isolation": "NOT_CHECKED"},
        "calibration_role": "terminal certification only; all candidate procedures frozen before labels",
        "limitations": ["Internally held-out official Task2 TRAINING cases, not challenge test.",
          "diagnosis_1 comes from a frozen current Archive snapshot, not official 2018 task labels.",
          ("Indeterminate excluded from binary task before splits; all source identity/pixel bridges retained."
           if target_mode == "binary" else "Indeterminate is a distinct class, not benign/malignant; tiny strata have no held-out coverage."),
          "Known patient/lesion and exact-pixel links grouped; unknown identity and near duplicates remain unaudited.",
          "Concepts mean any positive grayscale-mask pixel; no segmentation alignment/performance claim."]}
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("samples", samples), ("membership", membership), ("exclusions", exclusions),
                       ("source_manifest", manifest), ("dimension_mismatches", dimensions_differ)):
        with (out / (name + ".jsonl")).open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    for name, value in (("schema", schema), ("audit", audit)):
        with (out / (name + ".json")).open("x") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target-mode", choices=("three_class", "binary"), default="three_class")
    parser.add_argument("--require-complete", action="store_true", help="Reject missing images/metadata/targets; binary Indeterminate exclusion remains allowed")
    parser.add_argument("--reuse-prepared", type=Path, help="Reuse decoded-pixel audit only after rehashing every image/mask and checking metadata")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.metadata, args.out, args.seed, reuse_prepared=args.reuse_prepared,
                             target_mode=args.target_mode, require_complete=args.require_complete), sort_keys=True))


if __name__ == "__main__":
    main()

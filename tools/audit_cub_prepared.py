#!/usr/bin/env python3
"""Independent streaming semantic audit of CUB prepared records; no training.

The CLI requires official dimensions. Compact byte/double arrays hold raw
annotations; prepared JSONL is read one case at a time. Output contains only
hashes and aggregate counts, never sample labels, descriptions or images.
"""
from array import array
from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path

KNOWN_ANNOTATION_SHA256 = "5ebb9782d589f41a9c046bc7c5b1365e01839308e7cf88cbe83c0fb6d5362d98"
REFERENCE = "https://github.com/yewsiang/ConceptBottleneck/blob/d6353f270702b92feb5b084a6fd065f891d583f8/CUB/data_processing.py#L31-L39"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def indexed(path):
    result = {}
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            key, value = line.split(maxsplit=1)
            key = int(key)
            require(key > 0 and key not in result, "duplicate/nonpositive source dictionary key")
            result[key] = value.strip()
    return result


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                require(isinstance(row, dict), "JSONL record must be an object")
                yield row


def audit_cub(source, prepared, out, *, expected_images=11788, expected_attributes=312,
              expected_classes=200, expected_groups=28, expected_train=5994,
              expected_test=5794, expected_anomalies=606):
    """Fixture dimensions are injectable in Python only, not exposed by the CLI."""
    source, prepared, out = Path(source).resolve(), Path(prepared).resolve(), Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError("refusing existing audit output")
    paths = {"images": source / "images.txt", "classes": source / "classes.txt",
             "labels": source / "image_class_labels.txt", "splits": source / "train_test_split.txt",
             "certainties": source / "attributes/certainties.txt",
             "annotations": source / "attributes/image_attribute_labels.txt"}
    attributes = [p for p in (source / "attributes/attributes.txt", source.parent / "attributes.txt") if p.is_file()]
    require(bool(attributes), "missing source attribute vocabulary")
    require(len({file_hash(p) for p in attributes}) == 1, "conflicting attribute vocabulary files")
    paths["attributes"] = attributes[0]
    tables = {name: indexed(path) for name, path in paths.items() if name != "annotations"}
    ids = set(range(1, expected_images + 1))
    require(all(set(tables[name]) == ids for name in ("images", "labels", "splits")), "source image/label/split joins incomplete")
    require(set(tables["classes"]) == set(range(1, expected_classes + 1)), "source classes incomplete")
    require(set(tables["attributes"]) == set(range(1, expected_attributes + 1)), "source attributes incomplete")
    require(len(set(tables["images"].values())) == expected_images, "duplicate source image paths")
    raw_splits = Counter(tables["splits"].values())
    require(dict(raw_splits) == {"1": expected_train, "0": expected_test}, "official train/test counts mismatch")
    require(all(int(value) in tables["classes"] for value in tables["labels"].values()), "unknown source class")
    certainty_status = {}
    for key, value in tables["certainties"].items():
        normalized = value.lower().replace("_", " ").replace("-", " ")
        require(normalized in ("not visible", "guessing", "probably", "definitely") and key < 255,
                "unknown or out-of-range certainty dictionary entry")
        certainty_status[key] = {"not visible": "NOT_VISIBLE", "guessing": "UNCERTAIN_ANNOTATION",
                                 "probably": "OBSERVED", "definitely": "OBSERVED"}[normalized]
    n = expected_images * expected_attributes
    present, certainty = bytearray([255]) * n, bytearray(n)
    duration = array("d", [float("nan")]) * n
    anomalies, line_count = {}, 0
    annotation_hash = file_hash(paths["annotations"])
    with paths["annotations"].open(encoding="utf-8-sig") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            tokens = line.split()
            require(len(tokens) in (5, 6), "unexpected source annotation field count")
            iid, aid, value, cert = map(int, tokens[:4])
            require(iid in ids and 1 <= aid <= expected_attributes and value in (0, 1)
                    and cert in certainty_status, "invalid source annotation semantic key/value")
            offset = (iid - 1) * expected_attributes + aid - 1
            require(present[offset] == 255, "duplicate source image/attribute pair")
            present[offset], certainty[offset] = value, cert
            if len(tokens) == 6:
                require(annotation_hash == KNOWN_ANNOTATION_SHA256 and iid in (2275, 9364)
                        and 10 <= aid <= 312 and tokens[4] == "0", "unrecognized six-column source exception")
                suffix = float(tokens[5])
                require(math.isfinite(suffix) and suffix >= 0, "invalid ambiguous suffix")
                anomalies[(iid, aid)] = (lineno, line.rstrip("\r\n"), tokens[4:])
            else:
                value_time = float(tokens[4])
                require(math.isfinite(value_time) and value_time >= 0, "invalid source annotation duration")
                duration[offset] = value_time
            line_count += 1
    require(line_count == n and 255 not in present, "source annotation Cartesian join incomplete")
    require(len(anomalies) == expected_anomalies, "source anomaly count mismatch")
    groups = {}
    expected_concepts = []
    for aid in range(1, expected_attributes + 1):
        description = tables["attributes"][aid]
        groups.setdefault(description.split("::", 1)[0], []).append(aid - 1)
        expected_concepts.append({"id": "cub_attr_{:03d}".format(aid), "description": description,
                                  "values": ["absent", "present"]})
    require(len(groups) == expected_groups, "attribute group count mismatch")
    schema_path = prepared / "schema.json"
    schema = json.loads(schema_path.read_text())
    require(schema["dataset"] == "cub" and schema["num_classes"] == expected_classes
            and schema["concepts"] == expected_concepts
            and schema["groups"] == [{"id": k, "atoms": v} for k, v in groups.items()], "prepared schema differs from source")
    seen, sample_info, gold_status, split_counts = set(), {}, Counter(), Counter()
    null_duration = 0
    for row in rows(prepared / "samples.jsonl"):
        iid = int(row["provenance"]["source_id"])
        require(iid in ids and iid not in seen and row["sample_id"] == "cub:" + str(iid)
                and row["provenance"]["source_id"] == str(iid), "invalid/duplicate prepared sample ID")
        seen.add(iid)
        raw_split = "train" if tables["splits"][iid] == "1" else "test"
        require(row["dataset"] == "cub" and row["provenance"]["source_split"] == raw_split, "prepared provenance split differs")
        require(type(row["target"]["value"]) is int
                and row["target"] == {"value": int(tables["labels"][iid]) - 1, "status": "OBSERVED"}, "prepared class target differs")
        require(row["input"] == {"modality": "image", "text": None, "image_paths": ["images/" + tables["images"][iid]]}, "prepared input mapping differs")
        metadata = row["audit_metadata"]
        require(metadata["native_group_id"] == "cub:image:" + str(iid), "native image identity differs")
        anomaly_lines = [details[0] for (image_id, _), details in anomalies.items() if image_id == iid]
        require(metadata.get("annotation_suffix_anomaly_lines", []) == anomaly_lines,
                "prepared anomaly line references differ")
        if anomaly_lines:
            require(metadata.get("annotation_suffix_anomaly_sidecar") == "annotation_anomalies.jsonl",
                    "prepared anomaly sidecar reference differs")
        annotations = metadata["certainty_annotations"]
        require(set(annotations) == {str(a) for a in range(1, expected_attributes + 1)}
                and len(row["concepts"]) == expected_attributes, "prepared annotation join incomplete")
        for aid, concept in enumerate(row["concepts"], 1):
            offset = (iid - 1) * expected_attributes + aid - 1
            values = annotations[str(aid)]
            require(isinstance(values, list) and len(values) == 3 and type(values[0]) is int and type(values[1]) is int
                    and values[:2] == [present[offset], certainty[offset]], "prepared present/certainty differs")
            if math.isnan(duration[offset]):
                require(values[2] is None, "ambiguous duration must remain null")
                null_duration += 1
            else:
                require(type(values[2]) in (float, int) and values[2] == duration[offset], "prepared duration differs")
            status = certainty_status[certainty[offset]]
            expected_value = present[offset] if status == "OBSERVED" else None
            require(concept["concept_id"] == expected_concepts[aid - 1]["id"]
                    and concept["annotation_status"] == status and concept["value"] == expected_value
                    and (concept["value"] is None or type(concept["value"]) is int)
                    and concept.get("distribution") is None, "prepared concept value/status differs")
            gold_status[status] += 1
        outer = row["split"]
        require(isinstance(row["group_id"], str) and bool(row["group_id"]), "invalid prepared group ID")
        require(outer in ("train", "validation", "calibration", "test")
                and (outer == "test") == (raw_split == "test"), "prepared outer split violates official holdout")
        sample_info[row["sample_id"]] = (row["group_id"], outer, raw_split)
        split_counts[outer] += 1
    excluded, reasons = set(), Counter()
    kept_test_groups = {g for g, _, raw_split in sample_info.values() if raw_split == "test"}
    for row in rows(prepared / "exclusions.jsonl"):
        sid = row["sample_id"]
        require(sid.startswith("cub:"), "invalid excluded sample ID")
        iid = int(sid[4:])
        require(iid in ids and iid not in seen and iid not in excluded and sid == "cub:" + str(iid), "invalid/duplicate exclusion")
        require(row["reason"] == "lower_priority_group_overlap" and row["source_split"] == "train"
                and tables["splits"][iid] == "1" and row["group_id"] in kept_test_groups, "unsupported exclusion or missing retained test component")
        excluded.add(iid)
        reasons[row["reason"]] += 1
    require(seen | excluded == ids, "prepared plus excluded IDs do not cover raw data")
    members, group_role, group_outer, roles = set(), {}, {}, Counter()
    for row in rows(prepared / "membership.jsonl"):
        sid, group, role, outer = row["sample_id"], row["group_id"], row["split"], row["outer_split"]
        require(sid in sample_info and sid not in members, "membership incomplete/duplicate/unknown sample")
        require((group, outer) == sample_info[sid][:2], "membership sample group/outer split differs")
        require((outer == "train" and role in ("responder_fit", "head_fit", "policy_fit"))
                or (outer in ("validation", "calibration", "test") and role == outer), "invalid role-to-outer mapping")
        require(group_role.setdefault(group, role) == role, "group crosses fitting/evaluation roles")
        require(group_outer.setdefault(group, outer) == outer, "group crosses outer splits")
        members.add(sid)
        roles[role] += 1
    require(members == set(sample_info), "membership does not cover every prepared sample")
    sidecar = prepared / "annotation_anomalies.jsonl"
    audited_anomalies = set()
    if sidecar.exists():
        for row in rows(sidecar):
            key = (row["image_id"], row["attribute_id"])
            require(key in anomalies and key not in audited_anomalies, "unexpected/duplicate anomaly sidecar row")
            lineno, raw, suffix = anomalies[key]
            offset = (key[0] - 1) * expected_attributes + key[1] - 1
            require(row["line_number"] == lineno and row["raw_line"] == raw and row["suffix_tokens"] == suffix
                    and row["source_sha256"] == annotation_hash and row["duration"] is None
                    and row["present"] == present[offset] and row["certainty"] == certainty[offset]
                    and row["duration_status"] == "AMBIGUOUS_TRAILING_FIELDS", "anomaly sidecar does not preserve source")
            audited_anomalies.add(key)
    require(audited_anomalies == set(anomalies), "anomaly sidecar missing records")
    prepared_files = [prepared / name for name in ("schema.json", "samples.jsonl", "membership.jsonl", "exclusions.jsonl")]
    if sidecar.exists():
        prepared_files.append(sidecar)
    summary = {
        "format": "cbmjev-independent-cub-semantic-audit-v1",
        "audit_tool_sha256": file_hash(Path(__file__)),
        "status": "PREPARED_MECHANICALLY_VERIFIED_NOT_PAPER_ACCEPTANCE",
        "reference_parser": REFERENCE, "raw_images": expected_images, "classes": expected_classes,
        "attributes": expected_attributes, "attribute_groups": len(groups), "raw_annotation_pairs": line_count,
        "original_split_counts": {"train": raw_splits["1"], "test": raw_splits["0"]},
        "prepared_samples": len(seen), "excluded_samples": len(excluded), "exclusion_counts": dict(reasons),
        "prepared_outer_counts": dict(split_counts), "membership_role_counts": dict(roles),
        "prepared_concept_status_counts": dict(gold_status), "raw_six_column_anomalies": len(anomalies),
        "prepared_null_duration_count": null_duration, "anomaly_sidecar_records": len(audited_anomalies),
        "membership_records": len(members), "group_count": len(group_role),
        "group_cross_role_overlap": 0, "group_cross_outer_overlap": 0,
        "compact_annotation_storage_bytes": len(present) + len(certainty) + len(duration) * duration.itemsize,
        "source_file_sha256": {name: file_hash(path) for name, path in paths.items()},
        "prepared_file_sha256": {path.name: file_hash(path) for path in prepared_files},
        "limitations": ["Does not independently re-decode/hash image pixels or establish duplicate-group membership.",
                        "Does not establish model contamination absence, model performance or paper acceptance."],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Raw CUB_200_2011 directory")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out", required=True, help="New aggregate JSON report")
    args = parser.parse_args()
    print(json.dumps(audit_cub(args.source, args.prepared, args.out)))


if __name__ == "__main__":
    main()

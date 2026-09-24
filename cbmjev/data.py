"""Local-only public-data adapters. Gold records stay on the orchestration side.

No network, model imports, or synthetic fallback. See docs/DATA_ADAPTERS.md.
"""
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata


ASPECTS = ("food", "noise", "ambiance", "service")
# Exact official-file exception, not a general six-column CUB parser. The extra
# suffix has no verified semantics; only the established first four fields are used.
CUB_KNOWN_SIX_COLUMN_ANNOTATIONS_SHA256 = "5ebb9782d589f41a9c046bc7c5b1365e01839308e7cf88cbe83c0fb6d5362d98"
CUB_KNOWN_SIX_COLUMN_KEYS = frozenset((image_id, attr) for image_id in (2275, 9364)
                                    for attr in range(10, 313))
DERM_VALUES = {
    "pigment_network": ("absent", "typical", "atypical"),
    "blue_whitish_veil": ("absent", "present"),
    "vascular_structures": ("absent", "regular", "dotted/irregular"),
    "pigmentation": ("absent", "regular", "irregular"),
    "streaks": ("absent", "regular", "irregular"),
    "dots_and_globules": ("absent", "regular", "irregular"),
    "regression_structures": ("absent", "present"),
}
DERM_ALIASES = {
    "vascular_structures": {"arborizing": 1, "comma": 1, "hairpin": 1,
                            "within regression": 1, "wreath": 1,
                            "dotted": 2, "linear irregular": 2},
    "pigmentation": {"diffuse regular": 1, "localized regular": 1,
                     "diffuse irregular": 2, "localized irregular": 2},
    "regression_structures": {"blue areas": 1, "white areas": 1, "combinations": 1},
}
DERM_DIAGNOSES = (
    ("basal cell carcinoma",),
    ("nevus", "blue nevus", "clark nevus", "combined nevus", "congenital nevus",
     "dermal nevus", "recurrent nevus", "reed or spitz nevus"),
    ("melanoma", "melanoma (in situ)", "melanoma (less than 0.76 mm)",
     "melanoma (0.76 to 1.5 mm)", "melanoma (more than 1.5 mm)", "melanoma metastasis"),
    ("DF/LT/MLS/MISC", "dermatofibroma", "lentigo", "melanosis", "miscellaneous", "vascular lesion"),
    ("seborrheic keratosis",),
)
SKINCON_EXCLUDED_COLUMNS = {"", "ImageID", "Do not consider this image"}
ISIC2018_TASK2_ATTRIBUTES = ("globules", "milia_like_cyst", "negative_network",
                             "pigment_network", "streaks")


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sort_key(seed, purpose, dataset, group):
    value = "acam-split-v1|{}|{}|{}|{}".format(seed, purpose, dataset, group)
    return hashlib.sha256(value.encode("utf-8")).hexdigest(), group


def _identifier(value, field):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("{} must be a nonempty string/integer ID".format(field))
    return str(value)


def _label(value, mapping, field):
    if value is None or value == "":
        return None, "MISSING_ANNOTATION"
    if value == "no majority":
        return None, "NO_MAJORITY"
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("invalid {} label: {!r}".format(field, value))
    key = str(value)
    if key not in mapping:
        raise ValueError("unknown {} label: {!r}".format(field, value))
    return mapping[key], "OBSERVED"


def _distribution(value, labels, field):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                raise ValueError("invalid annotation distribution: " + field) from exc
    if not isinstance(value, dict) or not value:
        raise ValueError("annotation distribution must be a nonempty count dictionary: " + field)
    counts = [0.0] * len(labels)
    for key, count in value.items():
        if str(key) not in labels or isinstance(count, bool) or not isinstance(count, (int, float)):
            raise ValueError("unknown key/non-numeric count in " + field)
        if not math.isfinite(count) or count < 0:
            raise ValueError("negative/nonfinite annotation count in " + field)
        counts[labels.index(str(key))] += count
    total = sum(counts)
    if total <= 0:
        raise ValueError("zero annotation count total in " + field)
    return [count / total for count in counts]


def _concept(cid, value=None, status="MISSING_ANNOTATION", distribution=None):
    return {"concept_id": cid, "value": value, "annotation_status": status,
            "distribution": distribution}


def _sample(dataset, sid, group, source_split, raw, target, concepts, text=None, images=None, audit=None):
    return {"schema_version": "acam-data-v1", "dataset": dataset,
            "sample_id": dataset + ":" + sid, "group_id": group,
            "split": source_split, "fold_id": None,
            "input": {"modality": "text" if text is not None else "image", "text": text,
                      "image_paths": [] if images is None else images},
            "target": {"value": target[0], "status": target[1]}, "concepts": concepts,
            "provenance": {"source_id": sid, "source_split": source_split, "raw_sha256": _hash(raw)},
            "audit_metadata": audit or {}}


def _schema(dataset, classes, concepts, groups=None):
    return {"schema_version": "cbmjev-schema-v1", "dataset": dataset, "num_classes": classes,
            "concepts": [{"id": cid, "description": desc, "values": list(values)}
                         for cid, desc, values in concepts],
            "groups": groups if groups is not None else
            [{"id": cid, "atoms": [i]} for i, (cid, _, _) in enumerate(concepts)]}


def _rows(value):
    """Records, columns-of-lists, pandas columns-of-indexed-dicts, or ID->record."""
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict) and not value:
        rows = []
    elif isinstance(value, dict) and "id" in value:
        if isinstance(value["id"], list):
            n = len(value["id"])
            if any(not isinstance(col, list) or len(col) != n for col in value.values()):
                raise ValueError("column-oriented JSON has inconsistent column lengths")
            rows = [{key: col[i] for key, col in value.items()} for i in range(n)]
        elif isinstance(value["id"], dict):
            keys = list(value["id"])
            if any(not isinstance(col, dict) or set(col) != set(keys) for col in value.values()):
                raise ValueError("indexed column-oriented JSON has inconsistent row keys")
            rows = [{key: col[i] for key, col in value.items()} for i in keys]
        else:
            rows = [value]
    elif isinstance(value, dict) and all(isinstance(row, dict) for row in value.values()):
        rows = list(value.values())
    else:
        raise ValueError("unsupported JSON table layout")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("each source row must be an object")
    return rows


def _json_rows(path):
    if path.suffix.lower() == ".jsonl":
        result = []
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise ValueError("JSONL lines must be objects")
                    result.append(obj)
        return result
    return _rows(json.loads(path.read_text(encoding="utf-8-sig")))


def _cebab(source, options):
    train_variant = options.pop("train_variant", "train_exclusive")
    if train_variant not in ("train_exclusive", "train_inclusive"):
        raise ValueError("CEBaB train_variant must select exactly one official training split")
    if options:
        raise ValueError("unknown CEBaB options: " + ", ".join(sorted(options)))
    files = []
    tables = {}
    if source.is_file():
        obj = json.loads(source.read_text(encoding="utf-8-sig"))
        if not isinstance(obj, dict) or train_variant not in obj or "test" not in obj:
            raise ValueError("single CEBaB JSON must be a split->table object")
        dev = [key for key in ("dev", "validation") if key in obj]
        if len(dev) != 1:
            raise ValueError("provide exactly one official dev/validation split")
        tables = {key: _rows(obj[key]) for key in (train_variant, dev[0], "test")}
        files.append(source)
    else:
        for names in ((train_variant,), ("dev", "validation"), ("test",)):
            candidates = [source / (name + ext) for name in names for ext in (".json", ".jsonl")
                          if (source / (name + ext)).is_file()]
            if len(candidates) != 1:
                raise ValueError("need exactly one local JSON/JSONL for " + "/".join(names))
            path = candidates[0]
            tables[path.stem] = _json_rows(path)
            files.append(path)
    samples = []
    aspect_labels = ("Negative", "Positive", "unknown")
    aspect_map = dict(zip(aspect_labels, range(3)))
    for split, rows in tables.items():
        for row in rows:
            sid = _identifier(row.get("id"), "id")
            family = _identifier(row.get("original_id"), "original_id")
            if family in ("None", "null", "-1"):
                raise ValueError("unverified CEBaB original_id sentinel; require documented version mapping")
            # Zero and zero-padded zero are genuine family IDs, not missing values.
            original = row.get("is_original")
            if type(original) is not bool:
                raise ValueError("CEBaB is_original must be JSON boolean")
            text = row.get("description")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("CEBaB description must be nonempty text")
            concepts = []
            for aspect in ASPECTS:
                value, status = _label(row.get(aspect + "_aspect_majority"), aspect_map, aspect)
                distribution = _distribution(row.get(aspect + "_aspect_label_distribution"),
                                             aspect_labels, aspect)
                concepts.append(_concept(aspect, value, status, distribution))
            target = _label(row.get("review_majority"), {str(i): i - 1 for i in range(1, 6)}, "review")
            review_distribution = _distribution(row.get("review_label_distribution"),
                                                tuple(str(i) for i in range(1, 6)), "review")
            audit = {"native_group_id": "cebab:family:" + family, "original_id": family,
                     "is_original": original, "edit_id": row.get("edit_id"),
                     "edit_type": row.get("edit_type"), "review_distribution": review_distribution}
            samples.append(_sample("cebab", sid, audit["native_group_id"], split, row, target,
                                   concepts, text=text, audit=audit))
    schema = _schema("cebab", 5, [(a, a + " aspect sentiment", aspect_labels) for a in ASPECTS])
    report = {"identity_rule": "exact original_id; zero is a valid family", "train_variant": train_variant,
              "original_source_counts": {split: len(rows) for split, rows in tables.items()}}
    if train_variant == "train_inclusive":
        samples, exclusions = _inclusive_text_components(samples)
        report.update(text_identity_rule="NFKC + casefold + whitespace collapse; transitive family/text components",
                      _preparation_exclusions=exclusions)
    return samples, schema, files, report


def _inclusive_text_components(samples):
    """Drop entire train components touching held-out; group safe train edits.

    Held-out groups are not rewritten by this stronger training-only filter.
    The original NFC held-out rule and optional frozen membership remain intact.
    """
    parent = {s["group_id"]: s["group_id"] for s in samples}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    seen = {}
    for sample in samples:
        key = " ".join(unicodedata.normalize("NFKC", sample["input"]["text"]).casefold().split())
        group = sample["group_id"]
        if key in seen:
            a, b = find(group), find(seen[key])
            parent[max(a, b)] = min(a, b)
        else:
            seen[key] = group
    tainted = {find(s["group_id"]) for s in samples if s["split"] != "train_inclusive"}
    kept, excluded = [], []
    for sample in samples:
        if sample["split"] == "train_inclusive":
            component = find(sample["group_id"])
            if component in tainted:
                excluded.append({"sample_id": sample["sample_id"], "group_id": component,
                                 "source_split": "train_inclusive",
                                 "reason": "inclusive_family_text_component_touches_heldout"})
                continue
            sample["group_id"] = component
        kept.append(sample)
    return kept, excluded


def _indexed(path):
    result = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError("invalid indexed line in " + str(path))
            key = int(parts[0])
            if key < 1 or key in result:
                raise ValueError("duplicate/nonpositive ID in " + str(path))
            result[key] = parts[1]
    if not result:
        raise ValueError("empty required dictionary: " + str(path))
    return result


def _safe_image(root, relative):
    if not isinstance(relative, str) or not relative or ":" in relative or "\\" in relative:
        raise ValueError("invalid relative image path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("image path escapes source root")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("image symlink escapes source root") from exc
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError("missing/empty image: " + relative)
    return path.as_posix()


def _image_under_images(source, name):
    # Validate before prefixing: '/foo' must not become an apparently safe
    # 'images//foo'. The official image directory is the allowed boundary.
    return "images/" + _safe_image(source / "images", name)


def _cub(source, options):
    attributes_path = options.pop("attributes_file", None)
    certainties_path = options.pop("certainties_file", None)
    if options:
        raise ValueError("unknown CUB options: " + ", ".join(sorted(options)))
    paths = {key: source / name for key, name in (
        ("images", "images.txt"), ("labels", "image_class_labels.txt"),
        ("splits", "train_test_split.txt"), ("classes", "classes.txt"),
        ("annotations", "attributes/image_attribute_labels.txt"))}
    paths["attributes"] = Path(attributes_path) if attributes_path else source / "attributes/attributes.txt"
    paths["certainties"] = Path(certainties_path) if certainties_path else source / "attributes/certainties.txt"
    maps = {key: _indexed(path) for key, path in paths.items() if key != "annotations"}
    image_ids = set(maps["images"])
    if any(set(maps[key]) != image_ids for key in ("labels", "splits")):
        raise ValueError("CUB image/label/split ID joins are not one-to-one")
    for key in ("classes", "attributes"):
        if sorted(maps[key]) != list(range(1, len(maps[key]) + 1)):
            raise ValueError("CUB {} IDs must be contiguous from one".format(key))
    if len(maps["classes"]) < 2:
        raise ValueError("CUB requires at least two declared classes")
    certainty_map = {}
    for cid, name in maps["certainties"].items():
        normalized = name.lower().replace("_", " ").replace("-", " ").strip()
        if normalized == "not visible":
            certainty_map[cid] = "NOT_VISIBLE"
        elif normalized == "guessing":
            certainty_map[cid] = "UNCERTAIN_ANNOTATION"
        elif normalized in ("probably", "definitely"):
            certainty_map[cid] = "OBSERVED"
        else:
            raise ValueError("unknown CUB certainty label: " + name)
    annotations = defaultdict(dict)
    suffix_anomalies, annotation_sha256 = [], None
    anomaly_lines = defaultdict(list)
    with paths["annotations"].open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) not in (5, 6):
                raise ValueError("CUB annotation must contain image, attribute, present, certainty, time")
            image_id, attr, present, certainty = map(int, parts[:4])
            if len(parts) == 6:
                if annotation_sha256 is None:
                    annotation_sha256 = _file_hash(paths["annotations"])
                if annotation_sha256 != CUB_KNOWN_SIX_COLUMN_ANNOTATIONS_SHA256:
                    raise ValueError("unrecognized six-column CUB annotation file SHA256: " + annotation_sha256)
                try:
                    suffix_number = float(parts[5])
                except ValueError as exc:
                    raise ValueError("invalid known CUB annotation suffix") from exc
                if ((image_id, attr) not in CUB_KNOWN_SIX_COLUMN_KEYS or parts[4] != "0"
                        or not math.isfinite(suffix_number) or suffix_number < 0):
                    raise ValueError("invalid known CUB annotation suffix/key pattern")
                duration = None  # Neither suffix token is assigned an unverified time meaning.
                suffix_anomalies.append({"source_relative_path": "attributes/image_attribute_labels.txt",
                    "source_sha256": annotation_sha256, "line_number": line_number,
                    "raw_line": line.rstrip("\r\n"), "image_id": image_id, "attribute_id": attr,
                    "present": present, "certainty": certainty, "suffix_tokens": parts[4:],
                    "duration": None, "duration_status": "AMBIGUOUS_TRAILING_FIELDS",
                    "suffix_semantics": "UNINTERPRETED"})
                anomaly_lines[image_id].append(line_number)
            else:
                duration = float(parts[4])
            if (image_id not in image_ids or attr not in maps["attributes"] or present not in (0, 1)
                    or certainty not in certainty_map
                    or (duration is not None and (not math.isfinite(duration) or duration < 0))):
                raise ValueError("invalid CUB annotation key/value")
            if attr in annotations[image_id]:
                raise ValueError("duplicate CUB image/attribute annotation")
            annotations[image_id][attr] = (present, certainty, duration)
    if suffix_anomalies and {(row["image_id"], row["attribute_id"]) for row in suffix_anomalies} != CUB_KNOWN_SIX_COLUMN_KEYS:
        raise ValueError("known CUB annotation suffix exception requires the complete 606-key pattern")
    concepts, groups = [], {}
    for index, (aid, desc) in enumerate(sorted(maps["attributes"].items())):
        concepts.append(("cub_attr_{:03d}".format(aid), desc, ("absent", "present")))
        group = desc.split("::", 1)[0]
        groups.setdefault(group, []).append(index)
    samples = []
    for iid in sorted(image_ids):
        label = int(maps["labels"][iid])
        if label not in maps["classes"] or maps["splits"][iid] not in ("0", "1"):
            raise ValueError("invalid CUB class/split value")
        relative = _image_under_images(source, maps["images"][iid])
        gold = []
        for aid in sorted(maps["attributes"]):
            cid = "cub_attr_{:03d}".format(aid)
            if aid not in annotations[iid]:
                gold.append(_concept(cid))
            else:
                present, certainty, _ = annotations[iid][aid]
                status = certainty_map[certainty]
                gold.append(_concept(cid, present if status == "OBSERVED" else None, status))
        split = "train" if maps["splits"][iid] == "1" else "test"
        raw = {"image_id": iid, "image": maps["images"][iid], "class_id": label,
               "official_train": maps["splits"][iid], "annotations": annotations[iid]}
        group = "cub:image:" + str(iid)
        audit = {"native_group_id": group,
                 "certainty_annotations": {str(k): list(v) for k, v in annotations[iid].items()}}
        if anomaly_lines[iid]:
            audit["annotation_suffix_anomaly_lines"] = anomaly_lines[iid]
            audit["annotation_suffix_anomaly_sidecar"] = "annotation_anomalies.jsonl"
        samples.append(_sample("cub", str(iid), group, split, raw, (label - 1, "OBSERVED"),
                               gold, images=[relative], audit=audit))
    schema = _schema("cub", len(maps["classes"]), concepts,
                     [{"id": key, "atoms": value} for key, value in groups.items()])
    report = {"certainty_mapping": certainty_map, "class_names": maps["classes"],
              "identity_rule": "image ID, augmented by exact duplicates"}
    if suffix_anomalies:
        report["annotation_suffix_anomalies"] = {"count": len(suffix_anomalies),
                "source_sha256": annotation_sha256,
                "sidecar": "annotation_anomalies.jsonl", "suffix_semantics": "UNINTERPRETED",
                "handling": "first_four_fields_preserved_duration_null"}
        report["_annotation_anomaly_rows"] = suffix_anomalies
    return samples, schema, list(paths.values()), report


def _csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("missing/duplicate CSV header in " + str(path))
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError("CSV row width mismatch in " + str(path))
        return rows, reader.fieldnames


def _derm(source, options):
    patient_column = options.pop("patient_column", None)
    if options:
        raise ValueError("unknown Derm7pt options: " + ", ".join(sorted(options)))
    meta = source / "meta/meta.csv"
    rows, fields = _csv_rows(meta)
    required = {"diagnosis", "derm", "clinic"} | set(DERM_VALUES)
    if not required.issubset(fields):
        raise ValueError("Derm7pt metadata missing columns: " + ", ".join(sorted(required - set(fields))))
    if patient_column is not None and patient_column not in fields:
        raise ValueError("explicit trusted patient_column does not exist")
    files, assignments = [meta], {}
    for split in ("train", "valid", "test"):
        path = source / "meta" / (split + "_indexes.csv")
        indexes, columns = _csv_rows(path)
        if "indexes" not in columns:
            raise ValueError("Derm7pt index CSV requires official 'indexes' header")
        for row in indexes:
            index = int(row["indexes"])
            if index < 0 or index >= len(rows) or index in assignments:
                raise ValueError("Derm7pt indexes must be disjoint, valid 0-based metadata positions")
            assignments[index] = split
        files.append(path)
    if set(assignments) != set(range(len(rows))):
        raise ValueError("Derm7pt split indexes do not cover metadata exactly once")
    diagnosis_map = {name: i for i, names in enumerate(DERM_DIAGNOSES) for name in names}
    samples = []
    for index, row in enumerate(rows):
        concepts = []
        for cid, values in DERM_VALUES.items():
            mapping = dict(zip(values, range(len(values))))
            mapping.update(DERM_ALIASES.get(cid, {}))
            value, status = _label(row[cid], mapping, cid)
            # no-majority is meaningful only in the CEBaB annotator vote data.
            if status == "NO_MAJORITY":
                raise ValueError("unknown Derm7pt label: no majority")
            concepts.append(_concept(cid, value, status))
        target = _label(row["diagnosis"], diagnosis_map, "diagnosis")
        if target[1] == "NO_MAJORITY":
            raise ValueError("unknown Derm7pt diagnosis: no majority")
        derm = _image_under_images(source, row["derm"])
        clinic = _image_under_images(source, row["clinic"]) if row["clinic"] else None
        if patient_column:
            group = "derm7pt:patient:" + _identifier(row[patient_column], patient_column)
        else:
            group = "derm7pt:case:" + str(index)
        audit = {"native_group_id": group, "source_row": index, "clinic_image_path": clinic,
                 "case_num": row.get("case_num")}
        samples.append(_sample("derm7pt", str(index), group, assignments[index], row, target,
                               concepts, images=[derm], audit=audit))
    schema = _schema("derm7pt", 5, [(cid, cid.replace("_", " "), values)
                                   for cid, values in DERM_VALUES.items()])
    return samples, schema, files, {"identity_rule": "trusted patient column " + patient_column
            if patient_column else "metadata row/case only; patient independence NOT_CHECKED",
            "diagnosis_mapping": diagnosis_map, "index_rule": "indexes header; 0-based original row position"}


def _skincon_id(value):
    if not isinstance(value, str) or not value.endswith(".jpg") or "/" in value or "\\" in value:
        raise ValueError("SkinCon ImageID must be a local md5 jpg filename")
    stem = value[:-4]
    if len(stem) != 32 or any(ch not in "0123456789abcdef" for ch in stem.lower()):
        raise ValueError("SkinCon ImageID must be md5hash.jpg")
    return value, stem.lower()


def _skincon(source, options):
    target_column = options.pop("target_column", "three_partition_label")
    if options:
        raise ValueError("unknown SkinCon options: " + ", ".join(sorted(options)))
    if target_column not in ("three_partition_label", "nine_partition_label", "label"):
        raise ValueError("unsupported SkinCon target_column")
    ann_path = source / "annotations_fitzpatrick17k.csv"
    fitz_path = source / "fitzpatrick17k.csv"
    ann_rows, ann_fields = _csv_rows(ann_path)
    fitz_rows, fitz_fields = _csv_rows(fitz_path)
    required_ann = {"ImageID", "Do not consider this image"}
    required_fitz = {"md5hash", "fitzpatrick_scale", "fitzpatrick_centaur", "label",
                     "nine_partition_label", "three_partition_label", "url"}
    if not required_ann.issubset(ann_fields):
        raise ValueError("SkinCon annotations missing columns: "
                         + ", ".join(sorted(required_ann - set(ann_fields))))
    if not required_fitz.issubset(fitz_fields):
        raise ValueError("Fitzpatrick17k metadata missing columns: "
                         + ", ".join(sorted(required_fitz - set(fitz_fields))))
    concept_columns = [field for field in ann_fields if field not in SKINCON_EXCLUDED_COLUMNS]
    if not concept_columns:
        raise ValueError("SkinCon requires at least one concept column")
    fitz_by_hash = {}
    for row in fitz_rows:
        key = str(row["md5hash"]).lower()
        if len(key) != 32 or key in fitz_by_hash:
            raise ValueError("Fitzpatrick17k md5hash values must be unique 32-char hashes")
        fitz_by_hash[key] = row
    target_names = sorted({row[target_column] for row in fitz_rows if row.get(target_column)})
    if len(target_names) < 2:
        raise ValueError("SkinCon target requires at least two classes")
    target_map = {name: i for i, name in enumerate(target_names)}
    samples, ignored = [], 0
    for index, row in enumerate(ann_rows):
        image_id, md5hash = _skincon_id(row["ImageID"])
        if row["Do not consider this image"] not in ("0", "1"):
            raise ValueError("SkinCon quality flag must be binary")
        if row["Do not consider this image"] == "1":
            ignored += 1
            continue
        if md5hash not in fitz_by_hash:
            raise ValueError("SkinCon ImageID missing from Fitzpatrick17k metadata: " + image_id)
        fitz = fitz_by_hash[md5hash]
        concepts = []
        for column in concept_columns:
            value = row[column]
            if value not in ("0", "1"):
                raise ValueError("SkinCon concept must be binary: " + column)
            cid = "skincon_" + "".join(ch.lower() if ch.isalnum() else "_" for ch in column).strip("_")
            while "__" in cid:
                cid = cid.replace("__", "_")
            concepts.append(_concept(cid, int(value), "OBSERVED"))
        target_value, target_status = _label(fitz[target_column], target_map, target_column)
        relative = _image_under_images(source, image_id)
        raw = {"annotation": row, "fitzpatrick17k": fitz}
        audit = {"native_group_id": "skincon:image:" + md5hash, "source_row": index,
                 "md5hash": md5hash, "original_label": fitz["label"],
                 "nine_partition_label": fitz["nine_partition_label"],
                 "three_partition_label": fitz["three_partition_label"],
                 "fitzpatrick_scale": fitz["fitzpatrick_scale"],
                 "fitzpatrick_centaur": fitz["fitzpatrick_centaur"],
                 "source_url": fitz["url"]}
        samples.append(_sample("skincon", md5hash, audit["native_group_id"], "all", raw,
                               (target_value, target_status), concepts, images=[relative],
                               audit=audit))
    schema = _schema("skincon", len(target_map),
                     [(("skincon_" + "".join(ch.lower() if ch.isalnum() else "_" for ch in column).strip("_")).replace("__", "_"),
                       column, ("absent", "present")) for column in concept_columns])
    return samples, schema, [ann_path, fitz_path], {
        "identity_rule": "Fitzpatrick17k md5hash / SkinCon ImageID",
        "target_column": target_column, "target_mapping": target_map,
        "ignored_do_not_consider_count": ignored,
        "index_rule": "SkinCon ImageID equals md5hash.jpg; images stored under source/images"}


def _mask_has_positive_pixel(path):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("Pillow is required to parse ISIC2018 attribute masks") from exc
    with Image.open(path) as image:
        image.load()
        extrema = image.convert("L").getextrema()
    return extrema[1] > 0


def _isic2018_task2(source, options):
    target_level = options.pop("target_level", "diagnosis_1")
    if options:
        raise ValueError("unknown ISIC2018 Task2 options: " + ", ".join(sorted(options)))
    if target_level not in ("diagnosis_1", "diagnosis_2", "diagnosis_3"):
        raise ValueError("unsupported ISIC2018 target_level")
    masks_root = source / "groundtruth_extracted_v1/ISIC2018_Task2_Training_GroundTruth_v3"
    images_root = source / "images"
    metadata_path = source / "isic_archive_metadata_task2.json"
    if not masks_root.is_dir() or not images_root.is_dir() or not metadata_path.is_file():
        raise ValueError("ISIC2018 source requires extracted Task2 masks, images/, and API metadata JSON")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = metadata.get("records")
    if not isinstance(records, dict):
        raise ValueError("ISIC metadata JSON requires records object")
    mask_ids = defaultdict(set)
    pattern = re.compile(r"^(ISIC_\d{7})_attribute_(.+)\.png$")
    for path in masks_root.glob("ISIC_*_attribute_*.png"):
        match = pattern.match(path.name)
        if match and match.group(2) in ISIC2018_TASK2_ATTRIBUTES:
            mask_ids[match.group(1)].add(match.group(2))
    complete_mask_ids = {image_id for image_id, attrs in mask_ids.items()
                         if attrs == set(ISIC2018_TASK2_ATTRIBUTES)}
    target_names = sorted({record.get("metadata", {}).get("clinical", {}).get(target_level)
                           for image_id, record in records.items()
                           if image_id in complete_mask_ids
                           and record.get("metadata", {}).get("clinical", {}).get(target_level)})
    if len(target_names) < 2:
        raise ValueError("ISIC metadata target level has fewer than two observed classes")
    target_map = {name: i for i, name in enumerate(target_names)}
    targets = {}
    for image_id in complete_mask_ids:
        record = records.get(image_id)
        if isinstance(record, dict):
            value = record.get("metadata", {}).get("clinical", {}).get(target_level)
            if value in target_map:
                targets[image_id] = target_map[value]
    missing_target = sorted(complete_mask_ids - set(targets))
    usable_ids, skipped_missing_image = [], []
    for image_id in sorted(complete_mask_ids & set(targets)):
        image_path = images_root / (image_id + ".jpg")
        if image_path.is_file() and image_path.stat().st_size > 0:
            usable_ids.append(image_id)
        else:
            skipped_missing_image.append(image_id)
    samples = []
    for image_id in usable_ids:
        concepts = []
        raw_masks = {}
        for attr in ISIC2018_TASK2_ATTRIBUTES:
            mask_relative = "groundtruth_extracted_v1/ISIC2018_Task2_Training_GroundTruth_v3/{}_attribute_{}.png".format(image_id, attr)
            mask_path = source / mask_relative
            if not mask_path.is_file() or mask_path.stat().st_size == 0:
                raise ValueError("missing ISIC2018 mask: " + mask_relative)
            present = 1 if _mask_has_positive_pixel(mask_path) else 0
            concepts.append(_concept("isic2018_" + attr, present, "OBSERVED"))
            raw_masks[attr] = {"relative_path": mask_relative, "sha256": _file_hash(mask_path),
                               "present": present}
        image_relative = _image_under_images(source, image_id + ".jpg")
        record = records.get(image_id, {})
        clinical = record.get("metadata", {}).get("clinical", {}) if isinstance(record, dict) else {}
        raw = {"isic_id": image_id, "target": targets[image_id], "masks": raw_masks,
               "clinical": clinical}
        audit = {"native_group_id": "isic2018:image:" + image_id, "isic_id": image_id,
                 "mask_paths": {key: value["relative_path"] for key, value in raw_masks.items()},
                 "target_level": target_level, "target_class": target_names[targets[image_id]],
                 "copyright_license": record.get("copyright_license"),
                 "attribution": record.get("attribution")}
        samples.append(_sample("isic2018_task2", image_id, audit["native_group_id"], "all",
                               raw, (targets[image_id], "OBSERVED"), concepts,
                               images=[image_relative], audit=audit))
    schema = _schema("isic2018_task2", len(target_names),
                     [("isic2018_" + attr, attr.replace("_", " "), ("absent", "present"))
                      for attr in ISIC2018_TASK2_ATTRIBUTES])
    return samples, schema, [metadata_path], {
        "identity_rule": "ISIC image ID; one dermoscopic JPEG per Task2 mask set",
        "target_level": target_level, "target_classes": target_names,
        "task2_attributes": list(ISIC2018_TASK2_ATTRIBUTES),
        "mask_rule": "present iff decoded grayscale mask has any nonzero pixel",
        "complete_mask_ids": len(complete_mask_ids),
        "usable_ids_with_image_and_target": len(usable_ids),
        "skipped_missing_image_count": len(skipped_missing_image),
        "skipped_missing_target_count": len(missing_target),
        "skipped_missing_image_ids_first20": skipped_missing_image[:20],
        "skipped_missing_target_ids_first20": missing_target[:20]}


def _group_and_deduplicate(samples, root, decode_images):
    parent = {sample["group_id"]: sample["group_id"] for sample in samples}

    def find(key):
        while key != parent[key]:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    image_module = None
    if decode_images:
        try:
            from PIL import Image
            image_module = Image
        except ImportError:
            pass
    seen, images, duplicate_edges = {}, {}, []
    for sample in samples:
        group = sample["group_id"]
        keys = []
        if sample["input"]["modality"] == "text":
            text = sample["input"]["text"].replace("\r\n", "\n").replace("\r", "\n").strip()
            keys.append("text:" + _hash(unicodedata.normalize("NFC", text)))
        else:
            paths = list(sample["input"]["image_paths"])
            clinic = sample["audit_metadata"].get("clinic_image_path")
            if clinic:
                paths.append(clinic)
            for path in paths:
                if path not in images:
                    full = root / path
                    info = {"bytes": full.stat().st_size, "sha256": _file_hash(full)}
                    if image_module is not None:
                        try:
                            with image_module.open(full) as im:
                                im.load()
                                rgb = im.convert("RGB")
                                digest = hashlib.sha256(_encoded(list(rgb.size)) + rgb.tobytes()).hexdigest()
                                info["pixel_sha256"] = digest
                        except Exception as exc:
                            raise ValueError("image decode failed: " + path) from exc
                    images[path] = info
                keys.append("bytes:" + images[path]["sha256"])
                if "pixel_sha256" in images[path]:
                    keys.append("pixels:" + images[path]["pixel_sha256"])
        for key in keys:
            if key in seen and seen[key] != group:
                duplicate_edges.append([group, seen[key]])
                union(group, seen[key])
            else:
                seen[key] = group
    mapping = {key: find(key) for key in parent}
    for sample in samples:
        sample["group_id"] = mapping[sample["group_id"]]
    priority = {"all": 0, "train": 0, "train_exclusive": 0, "train_inclusive": 0,
                "dev": 1, "validation": 1, "valid": 1, "test": 2}
    highest = {}
    for sample in samples:
        group = sample["group_id"]
        highest[group] = max(highest.get(group, -1), priority[sample["split"]])
    kept, exclusions = [], []
    for sample in samples:
        if priority[sample["split"]] < highest[sample["group_id"]]:
            exclusions.append({"sample_id": sample["sample_id"], "group_id": sample["group_id"],
                               "source_split": sample["split"], "reason": "lower_priority_group_overlap"})
        else:
            kept.append(sample)
    merged = Counter(mapping.values())
    return kept, exclusions, {"native_to_component": mapping, "exact_duplicate_edges": duplicate_edges,
            "exact_duplicate_components": sum(n > 1 for n in merged.values()),
            "image_manifest": images, "image_decode_and_pixel_hash": "CHECKED" if image_module else "NOT_CHECKED"}


def _stratum(rows, dataset):
    if dataset == "cebab":
        originals = [s for s in rows if s["audit_metadata"]["is_original"]]
        if not originals:
            return "missing-original"
        rows = originals
    targets = {s["target"]["value"] for s in rows}
    if len(targets) > 1:
        return "mixed"
    value = next(iter(targets))
    return "missing-target" if value is None else "class:" + str(value)


def _assign(samples, dataset, seed, fold_count):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["group_id"]].append(sample)
    strata = defaultdict(list)
    outer = {}
    for group, rows in grouped.items():
        split = rows[0]["provenance"]["source_split"]
        if split == "test":
            outer[group] = "test"
        elif dataset in ("cub", "skincon", "isic2018_task2"):
            strata[_stratum(rows, dataset)].append(group)
        elif split in ("train", "train_exclusive", "train_inclusive"):
            outer[group] = "train"
        else:
            strata["all" if dataset == "cebab" else _stratum(rows, dataset)].append(group)
    for groups in strata.values():
        purpose = "cub-holdout" if dataset == "cub" else (
            "skincon-holdout" if dataset == "skincon" else (
                "isic2018-holdout" if dataset == "isic2018_task2" else "dev-half"))
        ordered = sorted(groups, key=lambda g: _sort_key(seed, purpose, dataset, g))
        n = len(ordered) // 10 if dataset in ("cub", "skincon", "isic2018_task2") else len(ordered) // 2
        for i, group in enumerate(ordered):
            outer[group] = "calibration" if i < n else (
                "validation" if dataset not in ("cub", "skincon", "isic2018_task2") or i < 2 * n else "train")
    train_strata = defaultdict(list)
    for group, rows in grouped.items():
        if outer[group] == "train":
            train_strata[_stratum(rows, dataset)].append(group)
    roles, folds, role_strata = {}, {}, {}
    for stratum, groups in train_strata.items():
        ordered = sorted(groups, key=lambda g: _sort_key(seed, "pilot-role", dataset, g))
        n_r, n_h = 3 * len(ordered) // 5, len(ordered) // 5
        for i, group in enumerate(ordered):
            roles[group] = "responder_fit" if i < n_r else ("head_fit" if i < n_r + n_h else "policy_fit")
        role_strata[stratum] = {role: sum(roles[g] == role for g in groups)
                               for role in ("responder_fit", "head_fit", "policy_fit")}
        ordered = sorted(groups, key=lambda g: _sort_key(seed, "outer-fold", dataset, g))
        folds.update({group: i % fold_count for i, group in enumerate(ordered)})
    membership = []
    for sample in samples:
        group = sample["group_id"]
        sample["split"], sample["fold_id"] = outer[group], folds.get(group)
        membership.append({"sample_id": sample["sample_id"], "group_id": group,
                           "split": roles.get(group, outer[group]), "outer_split": outer[group]})
    return membership, role_strata


def prepare_dataset(dataset, source, out, seed=17, source_revision="unspecified", **options):
    """Convert local official-format data; refuse any nonempty output directory.

    Returns path entries samples/schema/membership/audit/exclusions, source_root,
    plus report (the audit object). Optional common keys: fold_count=3,
    decode_images=True. Dataset-specific keys are documented, unknown keys fail.
    """
    aliases = {"cub": "cub", "cub200": "cub", "cub-200-2011": "cub",
               "cebab": "cebab", "derm7pt": "derm7pt",
               "skincon": "skincon", "skincon-fitzpatrick17k": "skincon",
               "isic2018_task2": "isic2018_task2", "isic2018-task2": "isic2018_task2"}
    if not isinstance(dataset, str) or dataset.lower() not in aliases:
        raise ValueError("dataset must be cub, cebab, derm7pt, or skincon")
    dataset = aliases[dataset.lower()]
    source, out = Path(source).resolve(), Path(out).resolve()
    if not source.exists():
        raise ValueError("local source does not exist")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("refusing to overwrite nonempty output: " + str(out))
    if source == out:
        raise ValueError("source and output must differ")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("source_revision must be a nonempty string")
    fold_count = options.pop("fold_count", 3)
    decode_images = options.pop("decode_images", True)
    heldout_reference = options.pop("heldout_reference", None)
    if heldout_reference is not None and dataset != "cebab":
        raise ValueError("heldout_reference is currently CEBaB-only")
    if type(fold_count) is not int or fold_count < 2 or type(decode_images) is not bool:
        raise ValueError("fold_count >= 2 integer and decode_images boolean required")
    if dataset != "cebab" and not source.is_dir():
        raise ValueError("image datasets require an extracted source directory")
    loader = {"cebab": _cebab, "cub": _cub, "derm7pt": _derm, "skincon": _skincon,
              "isic2018_task2": _isic2018_task2}[dataset]
    samples, schema, source_files, adapter_report = loader(source, dict(options))
    annotation_anomalies = adapter_report.pop("_annotation_anomaly_rows", [])
    preparation_exclusions = adapter_report.pop("_preparation_exclusions", [])
    if not samples:
        raise ValueError("dataset contains no samples")
    ids = [sample["sample_id"] for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample IDs (including across official splits)")
    source_counts = dict(Counter(s["provenance"]["source_split"] for s in samples))
    source_counts = adapter_report.get("original_source_counts", source_counts)
    root = source.parent if source.is_file() else source
    samples, exclusions, duplicates = _group_and_deduplicate(samples, root, decode_images)
    exclusions = preparation_exclusions + exclusions
    frozen_roles = {}
    if heldout_reference is not None:
        reference = Path(heldout_reference).resolve()
        reference_paths = [reference / name for name in ("samples.jsonl", "membership.jsonl", "schema.json")]
        reference_rows = {s["sample_id"]: s for s in _json_rows(reference_paths[0]) if s["split"] != "train"}
        frozen_roles = {m["sample_id"]: m for m in _json_rows(reference_paths[1]) if m["outer_split"] != "train"}
        if set(reference_rows) != set(frozen_roles) or json.loads(reference_paths[2].read_text()) != schema:
            raise ValueError("frozen heldout reference has inconsistent IDs/schema")
        filtered = []
        for sample in samples:
            sid = sample["sample_id"]
            if sample["provenance"]["source_split"] in ("dev", "validation", "test"):
                if sid not in frozen_roles:
                    exclusions.append({"sample_id": sid, "group_id": sample["group_id"],
                        "source_split": sample["provenance"]["source_split"], "reason": "not_in_frozen_heldout_reference"})
                    continue
                previous = reference_rows[sid]
                if any(sample[field] != previous[field] for field in ("input", "target", "concepts")):
                    raise ValueError("frozen heldout contents/labels changed: " + sid)
                sample["group_id"] = frozen_roles[sid]["group_id"]
            filtered.append(sample)
        if not set(frozen_roles).issubset({s["sample_id"] for s in filtered}):
            raise ValueError("new source/filter removed frozen heldout samples")
        samples = filtered
        adapter_report["frozen_heldout_reference"] = [{"path": str(p), "sha256": _file_hash(p)} for p in reference_paths]
    samples.sort(key=lambda sample: sample["sample_id"])
    membership, role_strata = _assign(samples, dataset, seed, fold_count)
    for sample, member in zip(samples, membership):
        if sample["sample_id"] in frozen_roles:
            frozen = frozen_roles[sample["sample_id"]]
            if frozen["split"] not in ("validation", "calibration", "test") or frozen["outer_split"] != frozen["split"]:
                raise ValueError("invalid frozen heldout role")
            member.update(frozen)
            sample["split"], sample["fold_id"] = frozen["outer_split"], None
    counts, effective, target_status, concept_status, group_sets = {}, {}, {}, {}, {}
    for split in ("train", "validation", "calibration", "test"):
        subset = [sample for sample in samples if sample["split"] == split]
        counts[split] = len(subset)
        effective[split] = sum(s["target"]["status"] == "OBSERVED" for s in subset)
        target_status[split] = dict(Counter(s["target"]["status"] for s in subset))
        concept_status[split] = {c["id"]: dict(Counter(s["concepts"][i]["annotation_status"] for s in subset))
                                 for i, c in enumerate(schema["concepts"])}
        group_sets[split] = {s["group_id"] for s in subset}
    manifest = [{"path": str(path), "sha256": _file_hash(path), "bytes": path.stat().st_size}
                for path in sorted(set(source_files))]
    audit = {"schema_version": "cbmjev-data-audit-v1", "status": "PREPARED_NOT_ACCEPTED",
             "dataset": dataset, "source_revision": source_revision, "source_root": str(root),
             "seed": seed, "fold_count": fold_count, "source_files": manifest,
             "raw_manifest_hash": _hash(manifest), "schema_hash": _hash(schema),
             "split_hash": _hash(membership), "adapter": adapter_report,
             "source_counts": source_counts, "observed_counts": counts,
             "effective_task_counts": effective, "target_status_counts": target_status,
             "concept_status_counts": concept_status,
             "groups_per_split": {key: len(value) for key, value in group_sets.items()},
             "role_counts": {role: sum(m["split"] == role for m in membership) for role in
                             ("responder_fit", "head_fit", "policy_fit", "validation", "calibration", "test")},
             "role_groups_per_stratum": role_strata,
             "exclusions_by_reason": dict(Counter(row["reason"] for row in exclusions)),
             "checks": {"ids_and_joins": "CHECKED", "known_label_values": "CHECKED",
                        "input_files_exist": "CHECKED", "group_cross_split_overlap": 0,
                        "group_cross_role_overlap": 0, "license_and_access": "NOT_CHECKED",
                        "source_authenticity": "NOT_CHECKED", "near_duplicate_phash": "NOT_CHECKED",
                        "patient_independence": "NOT_CHECKED",
                        "model_training_ancestor_isolation": "NOT_CHECKED"},
             "calibration_role": "terminal certification only; no fitting or adaptive threshold creation",
             "duplicates": duplicates}
    # Assert the actual manifests, rather than trusting the intended split algorithm.
    for field in ("split", "outer_split"):
        seen = {}
        for row in membership:
            group = row["group_id"]
            if group in seen and seen[group] != row[field]:
                raise AssertionError("group crosses " + field)
            seen[group] = row[field]
    # No source modification or dataset copying. Output is created only after all validation.
    out.mkdir(parents=True, exist_ok=True)
    result = {key: out / filename for key, filename in (
        ("samples", "samples.jsonl"), ("schema", "schema.json"), ("membership", "membership.jsonl"),
        ("audit", "audit.json"), ("exclusions", "exclusions.jsonl"))}
    output_rows = [("samples", samples), ("membership", membership), ("exclusions", exclusions)]
    if annotation_anomalies:
        result["annotation_anomalies"] = out / "annotation_anomalies.jsonl"
        output_rows.append(("annotation_anomalies", annotation_anomalies))
    for key, rows in output_rows:
        with result[key].open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(_encoded(row).decode("utf-8") + "\n")
    for key, value in (("schema", schema), ("audit", audit)):
        with result[key].open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    result.update(report=audit, source_root=root)
    return result

"""Strict reader for the official ASAP CSV release.

ASAP's ``-2`` means *not mentioned*. It is a valid observable answer, not a
missing annotation. Text-identical reviews share a duplicate component even
when the source assigns distinct IDs.
"""

from collections import Counter
import csv
import hashlib
from pathlib import Path
import unicodedata


ASPECTS = (
    "Location#Transportation", "Location#Downtown", "Location#Easy_to_find",
    "Service#Queue", "Service#Hospitality", "Service#Parking", "Service#Timely",
    "Price#Level", "Price#Cost_effective", "Price#Discount",
    "Ambience#Decoration", "Ambience#Noise", "Ambience#Space", "Ambience#Sanitary",
    "Food#Portion", "Food#Taste", "Food#Appearance", "Food#Recommend",
)
HEADER = ("id", "review", "star") + ASPECTS
ASPECT_VALUES = (-2, -1, 0, 1)
SOURCE_SPLITS = ("train", "dev", "test")
EXPECTED_COUNTS = {"train": 36850, "dev": 4940, "test": 4940}


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_asap_split(path, split, *, include_labels=True):
    """Return validated rows; ``include_labels=False`` reads only test identity/text.

    This lets the development audit detect duplicate components touching test
    without exposing test labels or scores to experiment selection.
    """
    if split not in SOURCE_SPLITS:
        raise ValueError("unknown ASAP source split")
    rows = []
    seen_ids = set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != HEADER:
            raise ValueError("ASAP CSV schema differs from the official 21-column release")
        for line_number, raw in enumerate(reader, 2):
            sid = raw["id"]
            if not sid or sid in seen_ids:
                raise ValueError("empty or repeated ID in {} at line {}".format(split, line_number))
            seen_ids.add(sid)
            review = raw["review"]
            if not review or not review.strip():
                raise ValueError("empty review in {} at line {}".format(split, line_number))
            text_hash = hashlib.sha256(normalized_text(review).encode("utf-8")).hexdigest()
            row = {"id": sid, "group_id": text_hash, "split": split}
            if include_labels:
                star = float(raw["star"])
                if star not in (1., 2., 3., 4., 5.):
                    raise ValueError("invalid native ASAP star at line {}".format(line_number))
                aspects = tuple(int(raw[key]) for key in ASPECTS)
                if any(value not in ASPECT_VALUES for value in aspects):
                    raise ValueError("invalid ASAP aspect response at line {}".format(line_number))
                row.update(y=int(star) - 1, z=aspects)
            rows.append(row)
    if len(rows) != EXPECTED_COUNTS[split]:
        raise ValueError("unexpected {} row count: {}".format(split, len(rows)))
    return rows


def load_development(root):
    """Load official train/dev, audit test overlap, and purge tainted components.

    The official dev remains a development set. Test labels are never loaded.
    Exact text duplicates spanning splits are excluded from earlier roles.
    """
    root = Path(root)
    paths = {split: root / (split + ".csv") for split in SOURCE_SPLITS}
    rows = {split: read_asap_split(paths[split], split,
                                   include_labels=split != "test")
            for split in SOURCE_SPLITS}
    group_sets = {split: {row["group_id"] for row in rows[split]}
                  for split in SOURCE_SPLITS}
    dev_overlap = group_sets["dev"] & group_sets["test"]
    train_overlap = group_sets["train"] & (group_sets["dev"] | group_sets["test"])
    usable_train = [r for r in rows["train"] if r["group_id"] not in train_overlap]
    usable_dev = [r for r in rows["dev"] if r["group_id"] not in dev_overlap]
    ids = [r["id"] for split in SOURCE_SPLITS for r in rows[split]]
    if len(ids) != len(set(ids)):
        raise ValueError("ASAP IDs overlap across source splits")
    report = {
        "source": "https://github.com/Meituan-Dianping/asap",
        "source_file_sha256": {split: file_sha256(path) for split, path in paths.items()},
        "source_rows": {split: len(rows[split]) for split in SOURCE_SPLITS},
        "source_duplicate_groups": {
            split: len(rows[split]) - len(group_sets[split]) for split in SOURCE_SPLITS},
        "cross_split_group_overlap": {
            "train_dev": len(group_sets["train"] & group_sets["dev"]),
            "train_test": len(group_sets["train"] & group_sets["test"]),
            "dev_test": len(group_sets["dev"] & group_sets["test"]),
        },
        "usable_rows": {"train": len(usable_train), "dev": len(usable_dev)},
        "excluded_duplicate_rows": {
            "train": len(rows["train"]) - len(usable_train),
            "dev": len(rows["dev"]) - len(usable_dev)},
        "aspect_values": list(ASPECT_VALUES),
        "aspect_counts_train": {aspect: dict(Counter(row["z"][j] for row in usable_train))
                                for j, aspect in enumerate(ASPECTS)},
        "star_counts_train": dict(Counter(row["y"] + 1 for row in usable_train)),
        "test_labels_loaded": False,
    }
    return usable_train, usable_dev, report

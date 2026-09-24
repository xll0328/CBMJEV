#!/usr/bin/env python3
"""Read-only audit: inclusive parquet stays in memory; outputs aggregate JSON.

Run on the data server. Does not convert, replace or extend prepared data.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import unicodedata
from urllib.request import urlopen

REVISION = "9d2a5a096f1b94344309406886ccc402d9048de8"
URL = "https://huggingface.co/datasets/CEBaB/CEBaB/resolve/" + REVISION + "/data/train_inclusive-00000-of-00001-8c2c561df4b78063.parquet"
SHA256 = "ec902716ca2ccad314bb3afad371f09d3b43e8b26a1d5b937c21c56b5a4c7683"


def text_key(row):
    text = " ".join(unicodedata.normalize("NFKC", row["description"]).casefold().split())
    return hashlib.sha256(text.encode()).hexdigest()


def audit(raw, data=None):
    import pyarrow.parquet as pq
    if data is None:
        with urlopen(URL, timeout=45) as response:
            data = response.read()
    if len(data) != 1336809 or hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("inclusive parquet does not match pinned revision LFS metadata")
    inclusive = pq.read_table(io.BytesIO(data)).to_pylist()
    tables = {}
    for split in ("train_exclusive", "validation", "test"):
        paths = list(Path(raw).glob(split + "-*.parquet"))
        if len(paths) != 1:
            raise ValueError("need exactly one pinned parquet for " + split)
        tables[split] = pq.read_table(paths[0]).to_pylist()
    families = lambda rows: {str(r["original_id"]) for r in rows}
    ids = lambda rows: {str(r["id"]) for r in rows}
    overlaps = {}
    for split, rows in tables.items():
        overlap = families(inclusive) & families(rows)
        overlaps[split] = {"family_count": len(overlap),
            "inclusive_rows_in_shared_families": sum(str(r["original_id"]) in overlap for r in inclusive),
            "id_count": len(ids(inclusive) & ids(rows)),
            "normalized_text_count": len({text_key(r) for r in inclusive} & {text_key(r) for r in rows})}
    forbidden_families = families(tables["validation"]) | families(tables["test"])
    forbidden_texts = {text_key(r) for r in tables["validation"] + tables["test"]}
    # Transitive family/text components: shared held-out text taints the entire
    # training family, including indirect links through another training family.
    tainted = set(forbidden_families)
    text_families = {}
    for row in inclusive:
        text_families.setdefault(text_key(row), set()).add(str(row["original_id"]))
    for key in forbidden_texts:
        tainted.update(text_families.get(key, set()))
    while True:
        expanded = set(tainted)
        for groups in text_families.values():
            if groups & tainted:
                expanded.update(groups)
        if expanded == tainted:
            break
        tainted = expanded
    safe = [r for r in inclusive if str(r["original_id"]) not in tainted]
    label_fields = [a + "_aspect_majority" for a in ("food", "noise", "ambiance", "service")] + ["review_majority"]
    def coverage(rows):
        return {field: dict(Counter(str(r.get(field)) for r in rows)) for field in label_fields}
    return {"status": "READ_ONLY_FEASIBILITY_NOT_PREPARED", "revision": REVISION,
        "inclusive_url": URL, "inclusive_sha256": SHA256, "inclusive_bytes": len(data),
        "inclusive_rows": len(inclusive), "inclusive_families": len(families(inclusive)),
        "original_vs_edit": dict(Counter(str(r["is_original"]) for r in inclusive)),
        "official_rows": {s: len(r) for s, r in tables.items()}, "overlaps": overlaps,
        "exclusive_ids_missing_from_inclusive": len(ids(tables["train_exclusive"]) - ids(inclusive)),
        "inclusive_label_coverage": coverage(inclusive),
        "safe_after_family_and_transitive_text_filter": {"rows": len(safe), "families": len(families(safe)),
            "removed_rows": len(inclusive) - len(safe), "label_coverage": coverage(safe)},
        "recommendation": "Potential separate prepared revision only; replace train source with filtered inclusive, never concatenate inclusive+exclusive. Rebuild all role memberships and retrain descendants. Hold validation/test fixed, do not inspect performance to choose inclusion.",
        "checks": {"source_sha256": "CHECKED", "family_and_exact_normalized_text_overlap": "CHECKED",
                   "semantic_near_duplicate": "NOT_CHECKED", "license_legal_review": "NOT_CHECKED"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stdin", action="store_true", help="Read pinned parquet bytes from stdin; never persist raw data")
    args = parser.parse_args()
    if args.out and args.out.exists():
        raise FileExistsError(args.out)
    report = audit(args.raw, sys.stdin.buffer.read() if args.stdin else None)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True))

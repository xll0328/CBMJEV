#!/usr/bin/env python3
"""Fetch exactly three content-locked public CEBaB splits, without credentials.

Requires pyarrow only for reading Parquet. No filtering, relabeling, or split
construction is performed. Failed runs intentionally retain partial artifacts;
use a new empty output directory after diagnosing a failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

REPOSITORY = "CEBaB/CEBaB"
REVISION = "9d2a5a096f1b94344309406886ccc402d9048de8"
BASE_URL = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/data/"
MANIFEST = (
    dict(split="train_exclusive", filename="train_exclusive-00000-of-00001-b84042a0c7a61311.parquet",
         bytes=436068, rows=1755, sha256="30ac4d294246f68e63acbd43cd698ff10fe7525075cff4f9eb0fac47cdf3c95f"),
    dict(split="validation", filename="validation-00000-of-00001-bc93208f2e115655.parquet",
         bytes=204497, rows=1673, sha256="25b47ac35bcef16ed3f177fa88477ba7ac6f43d1055b624d8da7eea0d9559e85"),
    dict(split="test", filename="test-00000-of-00001-853640e4ea036da4.parquet",
         bytes=205892, rows=1689, sha256="414c674f1219ba02146ee0e9bd20b60b6644fa3ceaa886e5bee796da07296743"),
)
INCLUSIVE = dict(split="train_inclusive", filename="train_inclusive-00000-of-00001-8c2c561df4b78063.parquet",
                 bytes=1336809, rows=11728, sha256="ec902716ca2ccad314bb3afad371f09d3b43e8b26a1d5b937c21c56b5a4c7683")


def parquet_reader():
    try:
        from pyarrow.parquet import ParquetFile
    except ImportError as exc:
        raise RuntimeError("CEBaB conversion requires pyarrow; install it in the isolated experiment environment (python -m pip install pyarrow). No download was started.") from exc
    return ParquetFile


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(spec, raw_dir, opener=None):
    """One anonymous request per artifact; redirects are handled by urllib."""
    opener = urlopen if opener is None else opener
    target = Path(raw_dir) / spec["filename"]
    partial = target.with_name(target.name + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError(f"Refusing existing raw artifact: {target}")
    url = BASE_URL + spec["filename"]
    request = Request(url, headers={"User-Agent": "CBMJev-locked-CEBaB-fetch/1"})
    digest, size = hashlib.sha256(), 0
    with partial.open("xb") as output, opener(request, timeout=60) as response:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > spec["bytes"]:
                raise ValueError(f"Byte count exceeded pinned size for {spec['split']}")
            digest.update(chunk)
            output.write(chunk)
    if size != spec["bytes"]:
        raise ValueError(f"Byte count mismatch for {spec['split']}: {size} != {spec['bytes']}")
    if digest.hexdigest() != spec["sha256"]:
        raise ValueError(f"SHA256 mismatch for {spec['split']}")
    # Output directory is owned by this single invocation; no overwrite allowed.
    if target.exists():
        raise FileExistsError(f"Refusing existing raw artifact: {target}")
    partial.rename(target)
    return target


def convert_parquet(source, target, expected_rows, reader=None):
    """Preserve native records and types as UTF-8 JSONL, in original order."""
    reader = parquet_reader() if reader is None else reader
    parquet = reader(str(source))
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(f"Parquet metadata row count mismatch: {parquet.metadata.num_rows} != {expected_rows}")
    rows, size, digest = 0, 0, hashlib.sha256()
    with Path(target).open("xb") as output:
        for batch in parquet.iter_batches(batch_size=512):
            for record in batch.to_pylist():
                line = (json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
                output.write(line)
                digest.update(line)
                size += len(line)
                rows += 1
                if rows > expected_rows:
                    raise ValueError("Decoded row count exceeded pinned count")
    if rows != expected_rows:
        raise ValueError(f"Decoded row count mismatch: {rows} != {expected_rows}")
    return dict(rows=rows, bytes=size, sha256=digest.hexdigest())


def fetch_cebab(out, *, opener=None, reader=None, train_variant="train_exclusive", local_raw=None, inclusive_bytes=None):
    if train_variant not in ("train_exclusive", "train_inclusive"):
        raise ValueError("select exactly one training split")
    if inclusive_bytes is not None and train_variant != "train_inclusive":
        raise ValueError("inclusive byte stream requires train_inclusive")
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Refusing nonempty or non-directory output: {out}")
    reader = parquet_reader() if reader is None else reader
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir()
    artifacts = []
    manifest = MANIFEST if train_variant == "train_exclusive" else (INCLUSIVE,) + MANIFEST[1:]
    for spec in manifest:
        reused = Path(local_raw) / spec["filename"] if local_raw is not None else None
        acquisition = "anonymous_official_download"
        if spec["split"] == "train_inclusive" and inclusive_bytes is not None:
            if len(inclusive_bytes) != spec["bytes"] or hashlib.sha256(inclusive_bytes).hexdigest() != spec["sha256"]:
                raise ValueError("inclusive input stream does not match pinned SHA256/bytes")
            path = raw / spec["filename"]
            with path.open("xb") as stream:
                stream.write(inclusive_bytes)
            acquisition = "verified_stream_relay"
        elif reused is not None and reused.is_file():
            if reused.stat().st_size != spec["bytes"] or file_hash(reused) != spec["sha256"]:
                raise ValueError("local parquet does not match pinned SHA256/bytes")
            path = raw / spec["filename"]
            with reused.open("rb") as stream, path.open("xb") as target:
                shutil.copyfileobj(stream, target)
            acquisition = "verified_local_raw_reuse"
        else:
            path = download_verified(spec, raw, opener=opener)
        converted = out / (spec["split"] + ".jsonl")
        stats = convert_parquet(path, converted, spec["rows"], reader=reader)
        artifacts.append(dict(
            split=spec["split"], canonical_url=BASE_URL + spec["filename"], acquisition=acquisition,
            raw=dict(path=str(path.relative_to(out)), bytes=spec["bytes"], sha256=spec["sha256"], verified=True),
            jsonl=dict(path=converted.name, **stats), expected_rows=spec["rows"]))
    receipt = dict(
        format="cbmjev-cebab-source-v1", repository=REPOSITORY, revision=REVISION, train_variant=train_variant,
        fetched_at_utc=datetime.now(timezone.utc).isoformat(),
        metadata_url=f"https://huggingface.co/datasets/{REPOSITORY}/raw/{REVISION}/dataset_infos.json",
        source_website="https://cebabing.github.io/CEBaB/",
        license=dict(stated="CC BY 4.0", url="https://creativecommons.org/licenses/by/4.0/",
                     evidence=["https://cebabing.github.io/CEBaB/", "https://cebabing.github.io/CEBaB/cebab_datasheet.html"],
                     caveat="Official website body and datasheet state CC BY 4.0, but the website's legacy badge says CC BY-NC; HF dataset_infos has an empty license field. This receipt records that inconsistency, not a legal resolution."),
        hf_snapshot_equivalence_to_zip_v1_1="NOT_VERIFIED; no official ZIP version claim is made",
        conversion=dict(format="UTF-8 JSONL", row_order="preserved", filtering=False, relabeling=False,
                        native_types_preserved=True, distribution_strings="kept as strings; not parsed",
                        ids="kept as native strings; leading zeros preserved",
                        reader="pyarrow.parquet.ParquetFile.iter_batches", batch_size=512,
                        raw_counts_are_not_post_adapter_counts=True),
        downloader_sha256=file_hash(__file__), total_rows=sum(x["jsonl"]["rows"] for x in artifacts),
        artifacts=artifacts,
    )
    with (out / "source_receipt.json").open("x", encoding="utf-8") as output:
        json.dump(receipt, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New or empty directory; existing artifacts are never overwritten")
    parser.add_argument("--train-variant", choices=("train_exclusive", "train_inclusive"), default="train_exclusive")
    parser.add_argument("--local-raw", type=Path, help="Reuse only content-verified pinned Parquets")
    parser.add_argument("--inclusive-stdin", action="store_true", help="Stream inclusive Parquet over stdin without relay-side disk storage")
    args = parser.parse_args()
    receipt = fetch_cebab(args.out, train_variant=args.train_variant, local_raw=args.local_raw,
                          inclusive_bytes=sys.stdin.buffer.read() if args.inclusive_stdin else None)
    print(json.dumps(dict(out=str(Path(args.out).resolve()), revision=REVISION, total_rows=receipt["total_rows"])))


if __name__ == "__main__":
    main()

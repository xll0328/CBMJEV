#!/usr/bin/env python3
"""Fetch public Fitzpatrick17k images used by SkinCon.

The script reads SkinCon ImageID values, joins them to Fitzpatrick17k md5hash
metadata, and stores images as images/<ImageID>. It records every attempted URL;
missing or broken upstream URLs are expected and must be reported, not hidden.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def skincon_jobs(root):
    ann_rows, ann_fields = read_csv(root / "annotations_fitzpatrick17k.csv")
    fitz_rows, fitz_fields = read_csv(root / "fitzpatrick17k.csv")
    if "ImageID" not in ann_fields:
        raise SystemExit("annotations_fitzpatrick17k.csv missing ImageID")
    if not {"md5hash", "url"}.issubset(fitz_fields):
        raise SystemExit("fitzpatrick17k.csv missing md5hash/url")
    fitz = {row["md5hash"].lower(): row for row in fitz_rows}
    jobs = []
    seen = set()
    for row in ann_rows:
        image_id = row["ImageID"]
        if not image_id.endswith(".jpg"):
            raise SystemExit("unexpected SkinCon ImageID: " + image_id)
        md5 = image_id[:-4].lower()
        if md5 in seen:
            raise SystemExit("duplicate SkinCon ImageID: " + image_id)
        seen.add(md5)
        if md5 not in fitz:
            raise SystemExit("SkinCon ImageID not found in Fitzpatrick17k: " + image_id)
        jobs.append({"image_id": image_id, "md5hash": md5, "url": fitz[md5]["url"]})
    return jobs


def fetch_one(root, job, timeout, user_agent):
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    out = image_dir / job["image_id"]
    if out.exists() and out.stat().st_size > 0:
        data = out.read_bytes()
        return {**job, "status": "existing", "bytes": len(data), "sha256": sha256(data)}
    partial = image_dir / (job["image_id"] + ".partial")
    try:
        request = Request(job["url"], headers={"User-Agent": user_agent})
        with urlopen(request, timeout=timeout) as response:
            data = response.read()
    except HTTPError as exc:
        return {**job, "status": "http_error", "error": str(exc.code)}
    except URLError as exc:
        return {**job, "status": "url_error", "error": str(exc.reason)}
    except Exception as exc:
        return {**job, "status": "error", "error": repr(exc)}
    if not data:
        return {**job, "status": "empty_response"}
    partial.write_bytes(data)
    partial.replace(out)
    return {**job, "status": "downloaded", "bytes": len(data), "sha256": sha256(data)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    root = args.root.resolve()
    jobs = skincon_jobs(root)
    if args.limit is not None:
        jobs = jobs[:args.limit]
    out = args.out or root / "image_fetch_report.jsonl"
    counts = {}
    started = time.time()
    with out.open("w", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_one, root, job, args.timeout,
                                   "CBMJEV research data preparation") for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                row = future.result()
                counts[row["status"]] = counts.get(row["status"], 0) + 1
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                if i % 100 == 0 or i == len(jobs):
                    print(json.dumps({"done": i, "total": len(jobs), "counts": counts,
                                      "elapsed_sec": round(time.time() - started, 3)},
                                     sort_keys=True), flush=True)
    print(json.dumps({"status": "complete", "total": len(jobs), "counts": counts,
                      "out": str(out), "elapsed_sec": round(time.time() - started, 3)},
                     sort_keys=True))


if __name__ == "__main__":
    main()

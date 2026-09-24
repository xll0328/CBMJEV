#!/usr/bin/env python3
"""Fetch ISIC 2018 Task 2 training images from public ISIC S3 image URLs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile


ID_RE = re.compile(r"(ISIC_\d{7})_attribute_[a-z_]+\.png$")


def ids_from_groundtruth(zip_path):
    ids = set()
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            match = ID_RE.search(name)
            if match:
                ids.add(match.group(1))
    if not ids:
        raise SystemExit("no ISIC attribute-mask IDs found")
    return sorted(ids)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch_one(image_id, image_dir, timeout):
    url = "https://isic-archive.s3.amazonaws.com/images/{}.jpg".format(image_id)
    out = image_dir / (image_id + ".jpg")
    if out.exists() and out.stat().st_size > 0:
        data = out.read_bytes()
        return {"isic_id": image_id, "url": url, "status": "existing",
                "bytes": len(data), "sha256": sha256(data)}
    partial = image_dir / (image_id + ".jpg.partial")
    try:
        req = Request(url, headers={"User-Agent": "CBMJEV research data preparation"})
        with urlopen(req, timeout=timeout) as response:
            data = response.read()
    except HTTPError as exc:
        return {"isic_id": image_id, "url": url, "status": "http_error", "error": str(exc.code)}
    except URLError as exc:
        return {"isic_id": image_id, "url": url, "status": "url_error", "error": str(exc.reason)}
    except Exception as exc:
        return {"isic_id": image_id, "url": url, "status": "error", "error": repr(exc)}
    if not data:
        return {"isic_id": image_id, "url": url, "status": "empty_response"}
    partial.write_bytes(data)
    partial.replace(out)
    return {"isic_id": image_id, "url": url, "status": "downloaded",
            "bytes": len(data), "sha256": sha256(data)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--groundtruth-zip", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    zip_path = args.groundtruth_zip or root / "ISIC2018_Task2_Training_GroundTruth_v3.zip"
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    report = args.out or root / "image_fetch_report.jsonl"
    ids = ids_from_groundtruth(zip_path)
    counts = {}
    started = time.time()
    with report.open("w", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_one, image_id, image_dir, args.timeout) for image_id in ids]
            for i, future in enumerate(as_completed(futures), 1):
                row = future.result()
                counts[row["status"]] = counts.get(row["status"], 0) + 1
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                if i % 100 == 0 or i == len(ids):
                    print(json.dumps({"done": i, "total": len(ids), "counts": counts,
                                      "elapsed_sec": round(time.time() - started, 3)},
                                     sort_keys=True), flush=True)
    print(json.dumps({"status": "complete", "total": len(ids), "counts": counts,
                      "out": str(report), "elapsed_sec": round(time.time() - started, 3)},
                     sort_keys=True))


if __name__ == "__main__":
    main()

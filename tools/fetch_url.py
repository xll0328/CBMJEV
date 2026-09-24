#!/usr/bin/env python3
"""Small resumable URL fetcher for server-side dataset downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = args.out.stat().st_size if args.out.exists() else 0
    headers = {"User-Agent": "CBMJEV research data preparation"}
    if existing:
        headers["Range"] = "bytes={}-".format(existing)
    request = Request(args.url, headers=headers)
    started = time.time()
    with urlopen(request, timeout=60) as response:
        status = getattr(response, "status", None)
        if existing and status != 206:
            raise SystemExit("server did not honor resume range")
        total_header = response.headers.get("Content-Range") or response.headers.get("Content-Length")
        mode = "ab" if existing else "wb"
        written = existing
        last_report = time.time()
        with args.out.open(mode) as stream:
            while True:
                chunk = response.read(args.chunk_mib * 1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                written += len(chunk)
                now = time.time()
                if now - last_report >= 30:
                    print(json.dumps({"bytes": written, "elapsed_sec": round(now - started, 3),
                                      "source_size_hint": total_header}, sort_keys=True), flush=True)
                    last_report = now
    receipt = {"url": args.url, "path": str(args.out), "bytes": args.out.stat().st_size,
               "sha256": file_sha256(args.out),
               "retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()

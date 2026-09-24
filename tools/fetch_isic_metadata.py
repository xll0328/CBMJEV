#!/usr/bin/env python3
"""Fetch public ISIC Archive metadata for IDs appearing in Task 2 masks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
import zipfile


ID_RE = re.compile(r"(ISIC_\d{7})_attribute_[a-z_]+\.png$")


def ids_from_masks(zip_or_dir):
    ids = set()
    path = Path(zip_or_dir)
    if path.is_file():
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    else:
        names = [p.name for p in path.rglob("ISIC_*_attribute_*.png")]
    for name in names:
        match = ID_RE.search(name)
        if match:
            ids.add(match.group(1))
    if not ids:
        raise SystemExit("no Task2 mask IDs found")
    return set(sorted(ids))


def get_json(url, timeout):
    req = Request(url, headers={"User-Agent": "CBMJEV research data preparation"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def write_result(path, wanted, found, missing, pages, next_url):
    result = {"retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "source_api": "https://api.isic-archive.com/api/v2/images/",
              "wanted": len(wanted), "found": len(found), "missing": sorted(missing),
              "pages": pages, "next_url": next_url, "records": found}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--scan-pages", action="store_true")
    args = parser.parse_args()
    wanted = ids_from_masks(args.mask_source)
    found, url, pages = {}, "https://api.isic-archive.com/api/v2/images/?limit={}".format(args.limit), 0
    if args.out.exists():
        checkpoint = json.loads(args.out.read_text(encoding="utf-8"))
        if checkpoint.get("wanted") == len(wanted) and isinstance(checkpoint.get("records"), dict):
            found = dict(checkpoint["records"])
            url = checkpoint.get("next_url") or url
            pages = int(checkpoint.get("pages") or 0)
    started = time.time()
    if not args.scan_pages:
        missing_ids = sorted(wanted - set(found))

        def fetch_id(image_id):
            last_error = None
            for attempt in range(args.retries):
                try:
                    return image_id, get_json("https://api.isic-archive.com/api/v2/images/{}/".format(image_id),
                                              args.timeout), None
                except Exception as exc:
                    last_error = repr(exc)
                    time.sleep(min(2 ** attempt, 30))
            return image_id, None, last_error

        completed, errors = 0, {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_id, image_id) for image_id in missing_ids]
            for future in as_completed(futures):
                image_id, row, error = future.result()
                completed += 1
                if row is not None and row.get("isic_id") == image_id:
                    found[image_id] = row
                else:
                    errors[image_id] = error or "bad_isic_id"
                if completed % 100 == 0 or completed == len(missing_ids):
                    write_result(args.out, wanted, found, wanted - set(found), pages, None)
                    print(json.dumps({"completed_new": completed, "found": len(found),
                                      "remaining": len(wanted - set(found)),
                                      "errors": len(errors),
                                      "elapsed_sec": round(time.time() - started, 3)},
                                     sort_keys=True), flush=True)
        write_result(args.out, wanted, found, wanted - set(found), pages, None)
        if errors:
            (args.out.parent / (args.out.name + ".errors.json")).write_text(
                json.dumps(errors, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "complete", "out": str(args.out), "wanted": len(wanted),
                          "found": len(found), "missing": len(wanted - set(found)),
                          "errors": len(errors)}, sort_keys=True))
        return

    while url and wanted - set(found):
        last_error = None
        for attempt in range(args.retries):
            try:
                payload = get_json(url, args.timeout)
                break
            except Exception as exc:
                last_error = repr(exc)
                time.sleep(min(2 ** attempt, 30))
        else:
            write_result(args.out, wanted, found, wanted - set(found), pages, url)
            raise SystemExit("failed after retries: " + str(last_error))
        pages += 1
        for row in payload.get("results", []):
            image_id = row.get("isic_id")
            if image_id in wanted:
                found[image_id] = row
        remaining = len(wanted) - len(found)
        url = payload.get("next")
        write_result(args.out, wanted, found, wanted - set(found), pages, url)
        print(json.dumps({"pages": pages, "found": len(found), "remaining": remaining,
                          "elapsed_sec": round(time.time() - started, 3)},
                         sort_keys=True), flush=True)
        if pages > 100:
            raise SystemExit("too many pages without full coverage")
    write_result(args.out, wanted, found, wanted - set(found), pages, url)
    print(json.dumps({"status": "complete", "out": str(args.out), "wanted": len(wanted),
                      "found": len(found), "missing": len(result["missing"])},
                     sort_keys=True))


if __name__ == "__main__":
    main()

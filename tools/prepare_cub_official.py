#!/usr/bin/env python3
"""Verify a local official CUB archive, safely extract, then call CBMJev prepare.

No download, training, or scientific acceptance is performed. Both output paths
must be new and disjoint. On failure, partial output is retained for diagnosis;
it is never reused or silently overwritten by a later invocation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile

OFFICIAL_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
OFFICIAL_BYTES = 1150585339
OFFICIAL_MD5 = "97eceeb196236b17998738112f37df78"
MAX_MEMBERS = 50000
MAX_EXTRACTED_BYTES = 5 * 1024 ** 3


def verify_stream(stream, expected_bytes, expected_md5):
    """Hash the same open descriptor subsequently used to read the archive."""
    size, md5, sha256 = 0, hashlib.md5(), hashlib.sha256()
    stream.seek(0)
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        size += len(chunk)
        if size > expected_bytes:
            raise ValueError("archive byte count exceeds official size")
        md5.update(chunk)
        sha256.update(chunk)
    if size != expected_bytes:
        raise ValueError("archive byte count does not match official size")
    if md5.hexdigest() != expected_md5:
        raise ValueError("archive MD5 does not match official checksum")
    stream.seek(0)
    return {"bytes": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def validated_members(archive):
    """Validate every member before writing any extracted bytes."""
    members, names, total = [], {}, 0
    for member in archive:
        if len(members) >= MAX_MEMBERS:
            raise ValueError("archive exceeds member limit")
        path = PurePosixPath(member.name)
        if (not member.name or path.is_absolute() or ".." in path.parts
                or "\\" in member.name or ":" in member.name or str(path) == "."):
            raise ValueError("unsafe archive path: " + member.name)
        if not (member.isdir() or member.isreg()) or member.issparse():
            raise ValueError("links, sparse files and special archive members are forbidden: " + member.name)
        normalized = path.as_posix()
        if normalized in names:
            raise ValueError("duplicate archive path: " + normalized)
        if member.size < 0 or (member.isdir() and member.size != 0):
            raise ValueError("invalid archive member size")
        total += member.size
        if total > MAX_EXTRACTED_BYTES:
            raise ValueError("archive exceeds extracted byte limit")
        names[normalized] = member.isdir()
        members.append((member, path))
    if not members:
        raise ValueError("empty archive")
    for _, path in members:
        for parent in path.parents:
            if parent.as_posix() in names and not names[parent.as_posix()]:
                raise ValueError("archive file is also a parent directory: " + parent.as_posix())
    return members, total


def _new_path(path):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError("output must be a new path: " + str(path))
    return path.resolve()


def safe_extract_verified(stream, out):
    """Extract a previously verified stream without tar.extract/extractall.

    This helper does not assert official provenance on its own. Its caller must
    verify the archive first. Modes/ownership are deliberately not imported.
    """
    out = _new_path(out)
    stream.seek(0)
    with tarfile.open(fileobj=stream, mode="r:gz") as archive:
        members, total = validated_members(archive)
        out.mkdir(parents=True, exist_ok=False)
        for member, relative in members:
            target = out.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                if target.stat().st_size != member.size:
                    raise ValueError("extracted file size mismatch: " + member.name)
    return {"member_count": len(members), "regular_files": sum(m.isreg() for m, _ in members),
            "uncompressed_file_bytes": total, "policy": "regular files/directories only; no inherited ownership or modes"}


def write_receipt(path, receipt):
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(receipt, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def prepare_cub(archive, out, prepared_out, seed=17):
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    archive = Path(archive).resolve(strict=True)
    if not archive.is_file():
        raise ValueError("archive must be a regular local file")
    out, prepared_out = _new_path(out), _new_path(prepared_out)
    if out == prepared_out or out in prepared_out.parents or prepared_out in out.parents:
        raise ValueError("extracted and prepared outputs must be disjoint, non-nested paths")
    # Direct script execution also works before an editable package install.
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from cbmjev.data import prepare_dataset

    with archive.open("rb") as stream:
        hashes = verify_stream(stream, OFFICIAL_BYTES, OFFICIAL_MD5)
        extraction = safe_extract_verified(stream, out)
    source, attributes = out / "CUB_200_2011", out / "attributes.txt"
    required = ("images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt",
                "attributes/image_attribute_labels.txt", "attributes/certainties.txt")
    for path in [attributes] + [source / relative for relative in required]:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("missing/empty official-format file: " + str(path))
    receipt = {
        "format": "cbmjev-cub-official-source-v1", "status": "ARCHIVE_VERIFIED_NOT_SCIENTIFICALLY_ACCEPTED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "archive": str(archive),
        "official_url": OFFICIAL_URL, "official_record": "https://data.caltech.edu/records/65de6-vp158",
        "official_md5": OFFICIAL_MD5, "official_bytes": OFFICIAL_BYTES,
        "verified_archive": hashes, "extraction": extraction, "seed": seed,
        "source_root": str(source), "attributes_file": str(attributes),
        "license_notice": "Author site restricts use to non-commercial research and educational purposes; authors do not own image copyrights.",
        "license_source": "https://www.vision.caltech.edu/datasets/cub_200_2011/",
        "pretraining_overlap_notice": "Author site warns of CUB/ImageNet image overlap; archive integrity does not resolve model contamination.",
        "scientific_acceptance": "NOT_EVALUATED", "prepare_output": str(prepared_out),
    }
    source_receipt = out / "source_receipt.json"
    write_receipt(source_receipt, receipt)
    revision = "CaltechDATA:65de6-vp158;md5=" + hashes["md5"] + ";sha256=" + hashes["sha256"]
    result = prepare_dataset("cub", source, prepared_out, seed=seed, source_revision=revision,
                             attributes_file=str(attributes))
    completion = {"format": "cbmjev-cub-official-preparation-v1", "status": "PREPARED_NOT_ACCEPTED",
                  "source_receipt": str(source_receipt), "source_archive_sha256": hashes["sha256"],
                  "seed": seed, "prepared_out": str(prepared_out), "source_counts": result["report"]["source_counts"],
                  "adapter_audit": str(result["audit"]), "scientific_acceptance": "NOT_EVALUATED"}
    write_receipt(prepared_out / "official_source_receipt.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Already downloaded CUB_200_2011.tgz")
    parser.add_argument("--out", required=True, help="New safe-extraction directory")
    parser.add_argument("--prepared-out", required=True, help="New, disjoint adapter-output directory")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(prepare_cub(args.archive, args.out, args.prepared_out, args.seed)))


if __name__ == "__main__":
    main()

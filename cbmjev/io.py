"""Explicit, no-overwrite artifact I/O and content-addressed receipts."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def fresh_dir(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("output must be new or empty; existing artifacts are never overwritten: " + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    info = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "evidence_status": "ENVIRONMENT_ONLY_NOT_BENCHMARK"}
    try:
        import torch
        info.update(torch=torch.__version__, cuda_version=torch.version.cuda,
                    cuda_available=torch.cuda.is_available(),
                    devices=[{"index": i, "name": torch.cuda.get_device_name(i),
                              "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                             for i in range(torch.cuda.device_count())])
    except ImportError:
        info["torch"] = "NOT_INSTALLED"
    return info

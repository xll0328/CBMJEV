"""Lossless epoch JSONL storage for logical v1 action-target packages."""
import hashlib
import json
from pathlib import Path

from .io import file_hash, read_json, write_json


STORAGE_FORMAT = "cbmjev-action-target-epoch-jsonl-v1"


def _encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _array_chunks(records):
    yield b"["
    for index, record in enumerate(records):
        if index:
            yield b","
        yield _encode(record)
    yield b"]"


def records_hash(records):
    digest = hashlib.sha256()
    for chunk in _array_chunks(records):
        digest.update(chunk)
    return digest.hexdigest()


def package_hash(package):
    """Exactly stable_hash(unsigned logical package), without materialization."""
    digest = hashlib.sha256()
    digest.update(b"{")
    for index, key in enumerate(sorted(k for k in package if k != "package_sha256")):
        if index:
            digest.update(b",")
        digest.update(_encode(key) + b":")
        if key == "records":
            for chunk in _array_chunks(package[key]):
                digest.update(chunk)
        else:
            digest.update(_encode(package[key]))
    digest.update(b"}")
    return digest.hexdigest()


class DiskRecords:
    def __init__(self, root, shards):
        self.root = Path(root).resolve()
        self.shards = shards
        seen = set()
        for epoch, shard in enumerate(shards):
            path = Path(shard["path"])
            if (path.is_absolute() or ".." in path.parts or str(path) in seen
                    or shard.get("epoch") != epoch
                    or type(shard.get("record_count")) is not int
                    or shard["record_count"] < 1):
                raise ValueError("invalid or duplicate target shard")
            seen.add(str(path))
            self._path(shard)

    def _path(self, shard):
        path = self.root / shard["path"]
        if not path.resolve().is_relative_to(self.root) or path.is_symlink():
            raise ValueError("target shard escapes package directory")
        return path

    def __len__(self):
        return sum(s["record_count"] for s in self.shards)

    def __iter__(self):
        return self.iter_epoch()

    def iter_epoch(self, epoch=None):
        for shard in self.shards:
            if epoch is not None and shard["epoch"] != epoch:
                continue
            path = self._path(shard)
            digest, count, size = hashlib.sha256(), 0, 0
            with path.open("rb") as handle:
                for line in handle:
                    digest.update(line)
                    size += len(line)
                    record = json.loads(line)
                    if record.get("epoch") != shard["epoch"]:
                        raise ValueError("target shard epoch mismatch")
                    count += 1
                    yield record
            if (count != shard["record_count"] or size != shard["size_bytes"]
                    or digest.hexdigest() != shard["sha256"]):
                raise ValueError("target shard content hash/count mismatch")


def iter_epoch(package, epoch):
    records = package["records"]
    if isinstance(records, DiskRecords):
        return records.iter_epoch(epoch)
    return (record for record in records if record["epoch"] == epoch)


class TargetWriter:
    """Append records in epoch order; publish manifest only after final hashing."""
    def __init__(self, path):
        self.path = Path(path)
        self.directory = self.path.parent / (self.path.stem + "_records")
        if self.path.exists() or self.directory.exists():
            raise ValueError("target output exists; refusing overwrite")
        self.directory.mkdir(parents=True)
        self.shards, self.handle = [], None
        self.count = 0

    def _close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
            shard = self.shards[-1]
            path = self.path.parent / shard["path"]
            shard.update(sha256=file_hash(path), size_bytes=path.stat().st_size)

    def append(self, record):
        epoch = record["epoch"]
        if not self.shards or self.shards[-1]["epoch"] != epoch:
            self._close()
            if epoch != len(self.shards):
                raise ValueError("target records must have contiguous ordered epochs")
            path = self.directory / ("epoch_%03d.jsonl" % epoch)
            self.handle = path.open("xb")
            self.shards.append({"path": str(path.relative_to(self.path.parent)),
                                "epoch": epoch, "record_count": 0})
        self.handle.write(_encode(record) + b"\n")
        self.shards[-1]["record_count"] += 1
        self.count += 1

    def records(self):
        self._close()
        return DiskRecords(self.path.parent, self.shards)

    def finish(self, package, *, validate):
        self._close()
        # The producer supplies the complete schema/config/ancestry validator.
        # A hash-only storage check is not a substitute for logical validation.
        validate(package)
        metadata = {key: value for key, value in package.items() if key != "records"}
        write_json(self.path, {"storage_format": STORAGE_FORMAT,
                              "metadata": metadata, "shards": self.shards})

    def close(self):
        self._close()


def open_target_package(path):
    package = read_json(path)
    if "storage_format" not in package:
        return package
    if (package["storage_format"] != STORAGE_FORMAT
            or set(package) != {"storage_format", "metadata", "shards"}
            or "records" in package["metadata"]):
        raise ValueError("invalid target storage manifest")
    return {**package["metadata"],
            "records": DiskRecords(Path(path).parent, package["shards"])}

"""Verified read-only source bindings for historical responder artifacts."""
from pathlib import Path

from .contracts import stable_hash
from .io import file_hash


SEMANTIC_RESPONDER_FILES = ("contracts.py", "responders.py", "nanojev.py", "runtime.py")


def _package_dir(source_dir):
    root = Path(source_dir).resolve(strict=True)
    package = root / "cbmjev"
    if package.is_dir():
        root = package.resolve(strict=True)
    if not all((root / name).exists() for name in SEMANTIC_RESPONDER_FILES):
        raise ValueError("responder_source_dir must be a release root or cbmjev package directory")
    return root


def _top_level_python_hashes(package):
    package = Path(package).resolve(strict=True)
    hashes = {}
    for path in sorted(package.glob("*.py")):
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(package)
        except ValueError as exc:
            raise ValueError("responder source file escapes package directory") from exc
        if not path.is_file():
            raise ValueError("responder source entry is not a regular file")
        hashes[path.name] = file_hash(path)
    return hashes


def source_code_fingerprint(source_dir):
    """Hash top-level cbmjev Python files without importing or executing them."""
    hashes = _top_level_python_hashes(_package_dir(source_dir))
    if not hashes:
        raise ValueError("responder source directory has no Python files")
    return stable_hash(hashes)


def responder_semantic_fingerprint(source_dir):
    package = _package_dir(source_dir)
    hashes = _top_level_python_hashes(package)
    semantic = {}
    for name in SEMANTIC_RESPONDER_FILES:
        if name not in hashes:
            raise ValueError("responder source is missing semantic file: " + name)
        semantic[name] = hashes[name]
    return stable_hash(semantic), semantic


def current_contracts_hash():
    return file_hash(Path(__file__).resolve().parent / "contracts.py")


def current_reader_fingerprint():
    return stable_hash(_top_level_python_hashes(Path(__file__).resolve().parent))


def verify_responder_source(receipt, *, responder_source_dir=None):
    """Bind a responder receipt to an on-disk source tree.

    ``None`` is strict current-source mode and returns ``None``. A directory
    argument is evidence, not an override: its full and semantic hashes must
    reproduce the hashes already recorded in the responder receipt.
    """
    if responder_source_dir is None:
        return None
    if not isinstance(receipt, dict):
        raise ValueError("responder receipt must be a dictionary")
    recorded_full = receipt.get("source_code_hash")
    recorded_semantic = receipt.get("semantic_code_hash")
    if not isinstance(recorded_full, str) or not isinstance(recorded_semantic, str):
        raise ValueError("responder receipt is missing source hashes")
    package = _package_dir(responder_source_dir)
    full_hash = source_code_fingerprint(package)
    semantic_hash, semantic_files = responder_semantic_fingerprint(package)
    if full_hash != recorded_full:
        raise ValueError("responder source directory does not match receipt source_code_hash")
    if semantic_hash != recorded_semantic:
        raise ValueError("responder source directory does not match receipt semantic_code_hash")
    if semantic_files["contracts.py"] != current_contracts_hash():
        raise ValueError("historical responder contracts.py is not compatible with current contracts.py")
    return {"format": "cbmjev-responder-source-binding-v1",
            "binding_mode": "verified_historical_source_directory",
            "source_package_dir": str(package),
            "source_code_hash": full_hash,
            "semantic_code_hash": semantic_hash,
            "semantic_files_sha256": semantic_files,
            "compatibility_rule": "contracts.py_sha256_must_equal_current",
            "current_contracts_sha256": current_contracts_hash(),
            "current_reader_source_code_hash": current_reader_fingerprint()}

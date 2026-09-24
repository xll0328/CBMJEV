"""Read-only bundle integrity checks; not a scientific claim verifier."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET


EXCLUDED_DIRECTORIES = frozenset((".venv", "build", "__pycache__", ".git"))


def verify(root, original_root=None):
    root = Path(root).resolve()
    if original_root is not None:
        original_root = Path(original_root)
        if not original_root.is_absolute():
            raise ValueError("original_root must be an explicit absolute path")
        # Normalize lexical parents without resolving historical-machine symlinks.
        original_root = Path(os.path.normpath(str(original_root)))
    failures, relocated = [], []
    counts = {"json": 0, "jsonl_rows": 0, "svg": 0, "local_links": 0, "relocated_links": 0}
    files = []
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        # Prune before descent: a server's environment can contain millions of files.
        subdirectories[:] = sorted(name for name in subdirectories if name not in EXCLUDED_DIRECTORIES)
        files.extend(Path(directory) / name for name in filenames if (Path(directory) / name).is_file())
    files.sort()
    for path in files:
        try:
            if path.suffix == ".json":
                json.loads(path.read_text())
                counts["json"] += 1
            elif path.suffix == ".jsonl":
                for number, line in enumerate(path.read_text().splitlines(), 1):
                    if line.strip():
                        json.loads(line)
                        counts["jsonl_rows"] += 1
            elif path.suffix == ".svg":
                if not ET.parse(path).getroot().tag.endswith("svg"):
                    raise ValueError("not an SVG root")
                counts["svg"] += 1
            elif path.suffix == ".md":
                # Historical reviewer traces may cite former source line numbers.
                # Existence still checked; anchors/line suffixes are not resolved.
                content = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
                for destination in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
                    destination = destination.strip().strip("<>")
                    if destination.startswith("#") or urlsplit(destination).scheme:
                        continue
                    target = unquote(destination.split("#", 1)[0])
                    target = re.sub(r":\d+$", "", target)
                    checked_path = path.parent / target
                    if original_root is not None and Path(target).is_absolute():
                        absolute = Path(os.path.normpath(target))
                        try:
                            relative = absolute.relative_to(original_root)
                        except ValueError:
                            pass  # Unrelated absolute references are never remapped.
                        else:
                            checked_path = root / relative
                            relocated.append({"file": str(path.relative_to(root)),
                                              "original_target": target,
                                              "relocated_target": str(checked_path)})
                            counts["relocated_links"] += 1
                    if target and not checked_path.exists():
                        failures.append({"file": str(path.relative_to(root)), "missing_link": target})
                    counts["local_links"] += 1
        except Exception as error:
            failures.append({"file": str(path.relative_to(root)), "error": str(error)})
    return {"kind": "bundle_integrity_not_research_evidence", "status": "PASS" if not failures else "FAIL",
            "counts": counts, "failures": failures, "relocated_references": relocated,
            "original_root": str(original_root) if original_root is not None else None,
            "notice": "Historical absolute references under original_root were checked at the current root; no files were rewritten."
                      if original_root is not None else "No historical-path relocation requested."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path,
                        help="explicit former absolute project root; remap only its absolute link descendants")
    args = parser.parse_args()
    if args.original_root is not None and not args.original_root.is_absolute():
        parser.error("--original-root must be an absolute path")
    report = verify(Path(__file__).resolve().parents[1], original_root=args.original_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(bool(report["failures"]))

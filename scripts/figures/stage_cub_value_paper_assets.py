#!/usr/bin/env python3
"""Stage completed CUB value-budget aggregate artifacts for the CVPR paper.

This script is deliberately conservative: it only consumes an already completed
aggregate directory produced by ``monitor_cub_value_aggregate.sh`` /
``summarize_cub_budget_grids.py``.  It does not train models, evaluate
policies, or invent missing rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REQUIRED = [
    "frontier_long.csv",
    "frontier_summary.json",
    "frontier_table.tex",
    "frontier_accuracy.pdf",
    "frontier_macro_f1.pdf",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_summary(path: Path) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if "policies" not in summary or "seeds" not in summary:
        raise ValueError(f"not a CUB value frontier summary: {path}")
    if not summary["policies"]:
        raise ValueError(f"summary has no policies: {path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", required=True,
                        help="Directory such as results/main/cub_value_budget_grid_60_61_62")
    parser.add_argument("--paper-dir", default="paper/cvpr2027")
    parser.add_argument("--tag", default="cub_value_policy",
                        help="Output stem under paper figures/generated")
    args = parser.parse_args()

    aggregate_dir = Path(args.aggregate_dir)
    paper_dir = Path(args.paper_dir)
    missing = [name for name in REQUIRED if not (aggregate_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"aggregate directory is incomplete: {aggregate_dir}; missing {missing}")

    summary = load_summary(aggregate_dir / "frontier_summary.json")
    figures_dir = paper_dir / "figures"
    generated_dir = paper_dir / "generated"
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "accuracy_pdf": figures_dir / f"{args.tag}_accuracy.pdf",
        "macro_f1_pdf": figures_dir / f"{args.tag}_macro_f1.pdf",
        "table_tex": generated_dir / f"{args.tag}_table.tex",
        "summary_json": generated_dir / f"{args.tag}_summary.json",
        "long_csv": generated_dir / f"{args.tag}_frontier_long.csv",
        "manifest_json": generated_dir / f"{args.tag}_manifest.json",
    }
    shutil.copy2(aggregate_dir / "frontier_accuracy.pdf", outputs["accuracy_pdf"])
    shutil.copy2(aggregate_dir / "frontier_macro_f1.pdf", outputs["macro_f1_pdf"])
    shutil.copy2(aggregate_dir / "frontier_table.tex", outputs["table_tex"])
    shutil.copy2(aggregate_dir / "frontier_summary.json", outputs["summary_json"])
    shutil.copy2(aggregate_dir / "frontier_long.csv", outputs["long_csv"])

    manifest = {
        "schema_version": "cbmjev-cub-value-paper-assets-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "aggregate_dir": str(aggregate_dir),
        "paper_dir": str(paper_dir),
        "tag": args.tag,
        "seeds": summary["seeds"],
        "num_seeds": summary.get("num_seeds"),
        "num_rows": summary.get("num_rows"),
        "source_files": {
            name: {
                "path": str(aggregate_dir / name),
                "sha256": sha256(aggregate_dir / name),
            }
            for name in REQUIRED
        },
        "outputs": {
            key: {
                "path": str(path),
                "sha256": sha256(path),
            }
            for key, path in outputs.items()
            if key != "manifest_json"
        },
        "claim_scope": (
            "Validation aggregate only; paper claims require the corresponding "
            "evidence-state and claim-matrix update before promotion."
        ),
    }
    outputs["manifest_json"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "staged": True,
        "tag": args.tag,
        "seeds": summary["seeds"],
        "outputs": {key: str(value) for key, value in outputs.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

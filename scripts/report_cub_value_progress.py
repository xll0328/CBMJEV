#!/usr/bin/env python3
"""Lightweight progress/ETA report for active CUB value outer folds.

This is a read-only convenience dashboard.  It estimates progress from
``action_targets_records/epoch_*.jsonl`` timestamps and each run's configured
``learning.policy_epochs``.  The latest shard can still be actively writing, so
this is an observed-shard progress indicator rather than a completion receipt.
The ETA is approximate and should be used for operations only, not as a paper
measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def expected_epochs(run_dir: Path) -> int | None:
    config = load_json(run_dir / "config.json")
    if not config:
        return None
    try:
        return int(config["learning"]["policy_epochs"])
    except Exception:
        return None


def epoch_files(run_dir: Path) -> list[tuple[int, float]]:
    target_dir = run_dir / "action_targets_records"
    rows: list[tuple[int, float]] = []
    if not target_dir.is_dir():
        return rows
    for path in target_dir.glob("epoch_*.jsonl"):
        match = re.fullmatch(r"epoch_(\d+)\.jsonl", path.name)
        if not match:
            continue
        try:
            rows.append((int(match.group(1)), path.stat().st_mtime))
        except OSError:
            continue
    return sorted(rows)


def read_pid(pid_path: Path) -> int | None:
    if not pid_path.is_file():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def pid_live(pid_path: Path) -> bool | None:
    pid = read_pid(pid_path)
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def open_target_epoch_hint(run_dir: Path, pid_path: Path) -> str | None:
    pid = read_pid(pid_path)
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return None
    fd_dir = Path("/proc") / str(pid) / "fd"
    if not fd_dir.is_dir():
        return None
    action_targets = run_dir / "action_targets.json"
    target_dir = str((run_dir / "action_targets_records").resolve())
    epochs: list[int] = []
    try:
        fds = list(fd_dir.iterdir())
    except OSError:
        return None
    for fd in fds:
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if not target.startswith(target_dir):
            continue
        match = re.search(r"epoch_(\d+)\.jsonl$", target)
        if match:
            epochs.append(int(match.group(1)))
    if not epochs:
        return None
    prefix = "validating_targets_fd" if action_targets.is_file() else "target_reader_fd"
    return f"{prefix}_e{max(epochs):03d}"


def summarize_fold(root: Path, seed: int, fold: int) -> dict[str, Any]:
    run_dir = root / f"runs/cub_outer{fold}_value_seed{seed}_v1"
    pid_path = root / f"logs/cub_outer{fold}_value_seed{seed}_v1.pid"
    receipt = run_dir / "receipt.json"
    expected = expected_epochs(run_dir)
    epochs = epoch_files(run_dir)
    latest_epoch = max((epoch for epoch, _ in epochs), default=None)
    now = time.time()
    completed_count = (latest_epoch + 1) if latest_epoch is not None else 0
    percent = None
    remaining_epochs = None
    if expected and expected > 0:
        percent = min(100.0, 100.0 * completed_count / expected)
        remaining_epochs = max(0, expected - completed_count)
    eta_seconds = None
    seconds_per_epoch = None
    if len(epochs) >= 2 and remaining_epochs is not None:
        first_epoch, first_time = epochs[0]
        last_epoch, last_time = epochs[-1]
        denom = max(1, last_epoch - first_epoch)
        seconds_per_epoch = max(0.0, (last_time - first_time) / denom)
        eta_seconds = seconds_per_epoch * remaining_epochs
    newest_age_seconds = None
    if epochs:
        newest_age_seconds = max(0.0, now - max(mtime for _, mtime in epochs))
    return {
        "seed": seed,
        "fold": fold,
        "run_dir": str(run_dir),
        "complete": receipt.is_file(),
        "pid_live": pid_live(pid_path),
        "expected_policy_epochs": expected,
        "latest_epoch": latest_epoch,
        "completed_epoch_files": completed_count,
        "progress_percent": percent,
        "remaining_epochs": remaining_epochs,
        "seconds_per_epoch_estimate": seconds_per_epoch,
        "eta_seconds_after_target_phase": eta_seconds,
        "newest_epoch_age_seconds": newest_age_seconds,
        "open_target_epoch_hint": open_target_epoch_hint(run_dir, pid_path),
    }


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--"
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def pct(value: float | None) -> str:
    return "--" if value is None else f"{value:.1f}%"


def build_report(root: Path, seeds: list[int]) -> dict[str, Any]:
    folds = [summarize_fold(root, seed, fold) for seed in seeds for fold in range(3)]
    active = [row for row in folds if row["pid_live"] is True and not row["complete"]]
    seed_summary = {}
    for seed in seeds:
        rows = [row for row in folds if row["seed"] == seed]
        seed_summary[str(seed)] = {
            "complete_folds": sum(1 for row in rows if row["complete"]),
            "target_phase_folds": sum(1 for row in rows if row["latest_epoch"] is not None),
            "min_progress_percent": min(
                (row["progress_percent"] for row in rows if row["progress_percent"] is not None),
                default=None,
            ),
            "max_eta_seconds_after_target_phase": max(
                (row["eta_seconds_after_target_phase"] for row in rows if row["eta_seconds_after_target_phase"] is not None),
                default=None,
            ),
        }
    return {
        "format": "cbmjev-cub-value-progress-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "seeds": seeds,
        "folds": folds,
        "seed_summary": seed_summary,
        "active_fold_count": len(active),
        "note": "ETA covers only the observed action-target epoch cadence; later merge/eval/stage time is not included.",
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# CUB value sprint progress",
        "",
        f"Generated UTC: `{report['generated_at_utc']}`",
        "",
        "| Seed | Fold | Complete | Live | Epoch shards observed | Progress | ETA target phase | Last shard age | Live target hint |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["folds"]:
        expected = row["expected_policy_epochs"]
        latest = row["latest_epoch"]
        epoch = "--" if latest is None else f"{latest + 1}/{expected or '?'}"
        lines.append(
            f"| {row['seed']} | {row['fold']} | {row['complete']} | {row['pid_live']} | "
            f"{epoch} | {pct(row['progress_percent'])} | "
            f"{fmt_duration(row['eta_seconds_after_target_phase'])} | "
            f"{fmt_duration(row['newest_epoch_age_seconds'])} | "
            f"{row['open_target_epoch_hint'] or '--'} |"
        )
    lines.extend([
        "",
        "ETA is operational only and excludes merge/evaluation/paper-staging time.",
        "The newest epoch shard may still be actively writing; top-level `receipt.json` is the completion signal.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--seeds", nargs="+", type=int, default=[60, 61, 62, 63])
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    report = build_report(args.root.resolve(), args.seeds)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, args.out_md)
    if not args.out_json and not args.out_md:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

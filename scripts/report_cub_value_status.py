#!/usr/bin/env python3
"""Report CUB value-sprint fold/final/watcher status in a compact table."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time


def exists(path: Path) -> str:
    return "Y" if path.exists() else "-"


def _expected_policy_epochs(root: Path) -> int | None:
    config_path = root / "config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text())
    except Exception:
        return None
    try:
        return int(config["learning"]["policy_epochs"])
    except Exception:
        return None


def _live_pid(pid_path: Path) -> int | None:
    try:
        pid = int(pid_path.read_text().strip())
    except Exception:
        return None
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return None
    return pid


def _open_target_epoch(pid: int, root: Path) -> str | None:
    """Best-effort /proc hint for long target finalization phases."""
    fd_dir = Path("/proc") / str(pid) / "fd"
    if not fd_dir.exists():
        return None
    try:
        target_dir_name = str((root / "action_targets_records").resolve())
    except OSError:
        target_dir_name = str(root / "action_targets_records")
    try:
        fds = list(fd_dir.iterdir())
    except OSError:
        return None
    epochs: list[int] = []
    for fd in fds:
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith(target_dir_name):
            match = re.search(r"epoch_(\d+)\.jsonl$", target)
            if match:
                epochs.append(int(match.group(1)))
    if not epochs:
        return None
    return f"target_reader_fd_e{max(epochs):03d}"


def fold_stage(root: Path, pid_path: Path | None = None) -> str:
    if (root / "receipt.json").exists():
        try:
            receipt = json.loads((root / "receipt.json").read_text())
            return "complete:" + str(receipt.get("num_target_records", "?"))
        except Exception:
            return "complete"
    stages = []
    if (root / "outer/cache/manifest.json").exists():
        stages.append("outer_cache")
    elif (root / "outer/responder/receipt.json").exists():
        stages.append("outer_resp")
    elif (root / "outer").exists():
        stages.append("outer_started")
    for i in range(3):
        base = root / f"inner_{i:03d}"
        if (base / "cache/manifest.json").exists():
            stages.append(f"i{i}_cache")
        elif (base / "responder/receipt.json").exists():
            stages.append(f"i{i}_resp")
        elif base.exists():
            stages.append(f"i{i}_started")
    if (root / "head/receipt.json").exists():
        stages.append("head")
    if (root / "action_targets.json").exists():
        stages.append("targets")
    target_dir = root / "action_targets_records"
    if target_dir.exists():
        epochs = []
        newest = None
        for path in target_dir.glob("epoch_*.jsonl"):
            match = re.fullmatch(r"epoch_(\d+)\.jsonl", path.name)
            if match:
                epochs.append(int(match.group(1)))
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            newest = mtime if newest is None else max(newest, mtime)
        if epochs:
            expected = _expected_policy_epochs(root)
            latest_epoch = max(epochs)
            if expected:
                detail = f"target_e{latest_epoch:03d}/{expected}"
            else:
                detail = f"target_e{latest_epoch:03d}"
            if newest is not None:
                age = max(0, int(time.time() - newest))
                detail += f"@{age}s"
            stages.append(detail)
            if expected and len(set(epochs)) >= expected and latest_epoch + 1 >= expected and pid_path is not None:
                pid = _live_pid(pid_path)
                if pid is not None:
                    fd_hint = _open_target_epoch(pid, root)
                    if fd_hint is not None:
                        if (root / "action_targets.json").exists():
                            fd_hint = fd_hint.replace("target_reader", "validating_targets")
                        stages.append(fd_hint)
    return ",".join(stages) if stages else "started" if root.exists() else "missing"


def final_stage(root: Path) -> str:
    responder = root / "responder"
    if (responder / "receipt.json").exists():
        return "complete_nested_layout"
    if (root / "receipt.json").exists():
        return "complete_flat_layout"
    if responder.exists():
        return "started_nested_layout"
    if root.exists():
        return "started"
    return "missing"


def final_responder_stage(project_root: Path, seed: int, root: Path, current_root: Path) -> tuple[str, str]:
    if current_root.exists():
        stage = final_stage(current_root)
        if stage == "missing":
            stage = "started_current_layout"
        pid = first_pid(
            project_root / f"logs/cub_final_balanced_seed{seed}_current_v1.pid",
            project_root / f"logs/cub_final_balanced_seed{seed}_current_v1_responder.pid",
        )
        return f"current:{stage}", pid
    return final_stage(root), first_pid(
        project_root / f"logs/cub_final_balanced_seed{seed}_v1.pid",
        project_root / f"logs/cub_final_balanced_seed{seed}_v1_responder.pid",
    )


def artifact_stage(root: Path, *, receipt: str = "receipt.json", metrics: str | None = None) -> str:
    if metrics is not None and (root / metrics).exists():
        try:
            payload = json.loads((root / metrics).read_text())
            policies = payload.get("policies")
            if isinstance(policies, dict):
                return f"complete_metrics:{len(policies)}policies"
        except Exception:
            pass
        return "complete_metrics"
    if (root / receipt).exists():
        return "complete"
    if root.exists():
        files = 0
        newest = None
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            files += 1
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            newest = mtime if newest is None else max(newest, mtime)
        if newest is not None:
            return f"started:{files}files@{max(0, int(time.time() - newest))}s"
        return "started"
    return "missing"


def maybe_pid(path: Path) -> str:
    if not path.exists():
        return "-"
    try:
        return path.read_text().strip() or "-"
    except OSError:
        return "?"


def pid_label(path: Path) -> str:
    pid = maybe_pid(path)
    if pid in {"-", "?"}:
        return pid
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return f"{pid}(done)"
    except PermissionError:
        return f"{pid}(live?)"
    except ValueError:
        return f"{pid}(bad)"
    return pid


def first_pid(*paths: Path) -> str:
    for path in paths:
        pid = pid_label(path)
        if pid != "-":
            return pid
    return "-"


def live_cmd_pid(*needles: str) -> str:
    proc = Path("/proc")
    if not proc.exists():
        return "-"
    matches: list[int] = []
    for item in proc.iterdir():
        if not item.name.isdigit():
            continue
        try:
            raw = (item / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        cmdline = raw.replace(b"\0", b" ").decode("utf-8", "replace")
        if all(needle in cmdline for needle in needles):
            matches.append(int(item.name))
    if not matches:
        return "-"
    return ",".join(str(pid) for pid in sorted(matches))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", nargs="+", type=int, default=[60, 61, 62, 63])
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for seed in args.seeds:
        for fold in range(3):
            run = root / f"runs/cub_outer{fold}_value_seed{seed}_v1"
            pid_path = root / f"logs/cub_outer{fold}_value_seed{seed}_v1.pid"
            rows.append([str(seed), f"outer{fold}", fold_stage(run, pid_path),
                         pid_label(pid_path)])
        final_stage_label, final_pid = final_responder_stage(
            root,
            seed,
            root / f"runs/cub_final_balanced_seed{seed}_v1",
            root / f"runs/cub_final_balanced_seed{seed}_current_v1/responder",
        )
        rows.append([str(seed), "final_responder", final_stage_label, final_pid])
        rows.append([str(seed), "merged_controller",
                     artifact_stage(root / f"runs/cub_merged_value_seed{seed}_v1"),
                     live_cmd_pid("merge-crossfit-outer", f"cub_merged_value_seed{seed}_v1")])
        rows.append([str(seed), "static_order",
                     artifact_stage(root / f"runs/cub_static_value_seed{seed}_v1"),
                     live_cmd_pid("fit-crossfit-static", f"cub_static_value_seed{seed}_v1")])
        final_cache_current = root / f"runs/cub_final_balanced_seed{seed}_validation_cache_v1"
        final_cache_legacy = root / f"runs/cub_final_cache_value_seed{seed}_v1"
        final_cache_root = final_cache_current if final_cache_current.exists() else final_cache_legacy
        final_cache_needle = final_cache_root.name
        rows.append([str(seed), "final_cache",
                     artifact_stage(final_cache_root, receipt="manifest.json"),
                     live_cmd_pid("cache-crossfit", final_cache_needle)])
        rows.append([str(seed), "value_eval",
                     artifact_stage(root / f"runs/cub_eval_value_budget_grid_seed{seed}_v1",
                                    metrics="metrics.json"),
                     live_cmd_pid("evaluate_crossfit_budget_grid.py",
                                  f"cub_eval_value_budget_grid_seed{seed}_v1")])
        rows.append([str(seed), "value_watcher",
                     exists(root / f"logs/cub_value_seed{seed}_watcher.log"),
                     pid_label(root / f"logs/cub_value_seed{seed}_watcher.pid")])
        rows.append([str(seed), "profile_watcher",
                     exists(root / f"logs/cub_final_profile_seed{seed}_watcher.log"),
                     pid_label(root / f"logs/cub_final_profile_seed{seed}_watcher.pid")])
    widths = [max(len(row[i]) for row in rows + [["seed", "component", "stage", "pid"]])
              for i in range(4)]
    print(" ".join(name.ljust(widths[i]) for i, name in enumerate(["seed", "component", "stage", "pid"])))
    print(" ".join("-" * width for width in widths))
    for row in rows:
        print(" ".join(row[i].ljust(widths[i]) for i in range(4)))


if __name__ == "__main__":
    main()

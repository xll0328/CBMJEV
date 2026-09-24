#!/usr/bin/env python3
"""Local visual validation workflow; no download, resume, test scoring or certification.

The launcher never changes cbmjev model/runtime semantics. CPU-only orchestration
and audits hide GPUs; CUDA stages reuse the existing UUID-pinned allocator guard.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from cbmjev.config import resolve_config
from cbmjev.contracts import DeclaredCost, Schema

GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
ROLES = {"responder_fit", "head_fit", "policy_fit", "validation", "calibration", "test"}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prepared", required=True)
    result.add_argument("--raw-root", required=True)
    result.add_argument("--out", required=True)
    result.add_argument("--config", default=str(PROJECT / "configs/vision_cache.json"))
    initialization = result.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--backbone", help="existing local torchvision ResNet18 state_dict; never downloaded")
    initialization.add_argument("--random-init", action="store_true", help="explicitly train ResNet18 from random initialization")
    result.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    result.add_argument("--gpu", help="one physical GPU index or UUID; mapped to logical cuda:0")
    result.add_argument("--gpu-memory-gib", type=float, default=8.0)
    result.add_argument("--gpu-reserve-gib", type=float, default=8.0)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--epochs", type=int, default=30, help="responder epochs only; f/controller epochs come from config")
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--learning-rate", type=float, default=0.001)
    result.add_argument("--image-size", type=int, default=224)
    result.add_argument("--python", default=sys.executable, help="Python used for every stage")
    result.add_argument("--live", action="store_true", help="also run validation live episodes; not isolated latency evidence")
    result.add_argument("--dry-run", action="store_true", help="validate and print JSON plan; no output directory or GPU queries")
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def iter_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("expected JSON object at {}:{}".format(path, line_number))
                yield value


def check_prepared(prepared, raw_root):
    paths = [prepared / name for name in ("schema.json", "samples.jsonl", "membership.jsonl", "audit.json")]
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths):
        raise ValueError("prepared requires nonempty schema/samples/membership/audit artifacts")
    schema = Schema.from_dict(read_json(paths[0]))
    if schema.dataset not in {"cub", "isic2018_task2", "isic2018_task2_binary", "derm7pt"}:
        raise ValueError("supported visual prepared datasets: cub, isic2018_task2, isic2018_task2_binary, derm7pt")
    membership, groups, counts = {}, {}, Counter()
    for row in iter_jsonl(prepared / "membership.jsonl"):
        sid, gid, role = row.get("sample_id"), row.get("group_id"), row.get("split")
        if not isinstance(sid, str) or not sid or not isinstance(gid, str) or not gid or role not in ROLES:
            raise ValueError("invalid prepared membership ID/group/role")
        if sid in membership or (gid in groups and groups[gid] != role):
            raise ValueError("duplicate membership or group crossing fitting/evaluation roles")
        membership[sid], groups[gid] = (gid, role), role
        counts[role] += 1
    if any(not counts[role] for role in ("responder_fit", "head_fit", "policy_fit", "validation")):
        raise ValueError("responder_fit/head_fit/policy_fit/validation must all be nonempty")
    seen, image_count = set(), 0
    for row in iter_jsonl(prepared / "samples.jsonl"):
        sid = row.get("sample_id")
        if not isinstance(sid, str) or sid in seen or sid not in membership or membership[sid][0] != row.get("group_id"):
            raise ValueError("prepared samples and membership IDs/groups do not align")
        seen.add(sid)
        payload = row.get("input", {})
        if not isinstance(payload, dict):
            raise ValueError("visual chain requires image-only prepared inputs")
        paths = payload.get("image_paths")
        if payload.get("modality") != "image" or payload.get("text") is not None or not isinstance(paths, list) or not paths:
            raise ValueError("visual chain requires image-only prepared inputs")
        for relative in paths:
            if (not isinstance(relative, str) or not relative or ":" in relative or "\\" in relative
                    or Path(relative).is_absolute() or ".." in Path(relative).parts):
                raise ValueError("image path must stay relative to raw-root")
            target = (raw_root / relative).resolve()
            try:
                target.relative_to(raw_root)
            except ValueError as exc:
                raise ValueError("image symlink escapes raw-root") from exc
            if not target.is_file() or target.stat().st_size == 0:
                raise ValueError("missing/empty image: " + str(target))
            image_count += 1
    if seen != set(membership):
        raise ValueError("prepared samples and membership IDs do not align")
    audit = read_json(prepared / "audit.json")
    if not isinstance(audit.get("source_revision"), str) or not audit["source_revision"]:
        raise ValueError("prepared audit requires an explicit source_revision")
    return schema, {"dataset": schema.dataset, "num_atoms": schema.num_atoms, "num_query_groups": schema.num_groups,
                    "samples": len(seen), "image_paths_checked": image_count, "role_counts": dict(counts),
                    "source_revision": audit["source_revision"], "adapter_status": audit.get("status"),
                    "scope": "shape/roles/path/existence only; not image decoding, semantic or paper acceptance"}


def build_plan(args):
    unresolved_out = Path(args.out).expanduser().absolute()
    if unresolved_out.exists() or unresolved_out.is_symlink():
        raise FileExistsError("run output already exists; use a new run directory")
    prepared, raw_root, out, config_path = (Path(value).expanduser().resolve() for value in
                                         (args.prepared, args.raw_root, args.out, args.config))
    if out.exists() or out.is_symlink():
        raise FileExistsError("run output already exists; use a new run directory")
    if not out.parent.is_dir() or not raw_root.is_dir() or not prepared.is_dir():
        raise ValueError("prepared/raw-root/output parent must already exist")
    if out == prepared or out == raw_root or prepared in out.parents or raw_root in out.parents:
        raise ValueError("run output must not be inside prepared or raw data")
    if args.seed < 0 or args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        raise ValueError("seed >= 0, epochs/batch-size >= 1, image-size >= 32 required")
    for name in ("learning_rate", "gpu_memory_gib", "gpu_reserve_gib"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(name.replace("_", "-") + " must be finite and positive")
    if (args.device == "cuda:0") != (args.gpu is not None):
        raise ValueError("CUDA requires explicit --gpu and --device cuda:0; CPU must not select a GPU")
    if args.gpu is not None and not (re.fullmatch(r"0|[1-9][0-9]*", args.gpu) or GPU_UUID.fullmatch(args.gpu)):
        raise ValueError("gpu must be one nonnegative index or physical GPU UUID")
    python_exe = shutil.which(args.python)
    if python_exe is None:
        raise ValueError("Python executable unavailable: " + args.python)
    python_exe = str(Path(python_exe).absolute())
    backbone = Path(args.backbone).expanduser().resolve() if args.backbone else None
    if backbone is not None and (not backbone.is_file() or backbone.stat().st_size == 0):
        raise ValueError("backbone must be an existing nonempty local ResNet18 state_dict")
    schema, data = check_prepared(prepared, raw_root)
    config = resolve_config(read_json(config_path))
    config["device"], config["seed"] = args.device, args.seed
    methods = config["evaluation"]["methods"]
    if "nano_risk" in methods:
        raise ValueError("this visual chain does not fit a Nano controller")
    if ({"risk", "value"} - {config["learning"]["objective"]}) & set(methods):
        raise ValueError("controller objective and evaluation methods differ")
    if "lookahead" in methods and schema.num_groups > 7:
        raise ValueError("empirical lookahead supports at most 7 query groups")
    cap = config["policy"]["max_groups"]
    if cap is not None and cap > schema.num_groups:
        raise ValueError("max_groups exceeds the prepared schema")
    if "all" in methods:
        full_cost = DeclaredCost(**config["cost"])(schema.empty_state(), tuple(range(schema.num_groups)))
        if not config["learning"]["include_all"] or cap not in (None, schema.num_groups):
            raise ValueError("all-at-once requires include_all and a full query-group budget")
        if config["policy"]["max_cost"] is not None and full_cost > config["policy"]["max_cost"]:
            raise ValueError("all-at-once is infeasible under max_cost")
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device == "cuda:0" and workspace not in {":4096:8", ":16:8"}:
        raise ValueError("deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    stages = []

    def stage(name, arguments, expected, uses_device=False):
        stages.append({"name": name, "arguments": [str(value) for value in arguments],
                       "device": args.device if uses_device else "cpu",
                       "log": str(out / "logs" / ("{:02d}_{}.log".format(len(stages) + 1, name))),
                       "expected_artifacts": [str(out / relative) for relative in expected]})

    stage("doctor", ["doctor", "--check-device", args.device, "--out", out / "doctor.json"], ["doctor.json"], True)
    responder = ["train-responder", "--kind", "resnet18", "--prepared", prepared, "--raw-root", raw_root,
                 "--out", out / "responder", "--device", args.device, "--seed", args.seed,
                 "--epochs", args.epochs, "--batch-size", args.batch_size, "--learning-rate", args.learning_rate,
                 "--image-size", args.image_size]
    if backbone is not None:
        responder += ["--backbone", backbone]
    stage("responder", responder, ["responder/responder.pt", "responder/receipt.json", "responder/training.json"], True)
    stage("cache", ["cache", "--prepared", prepared, "--raw-root", raw_root, "--responder", out / "responder",
                    "--out", out / "cache", "--device", args.device],
          ["cache/responses.jsonl", "cache/manifest.json", "cache/schema.json"], True)
    stage("semantic_audit", ["audit-responses", "--prepared", prepared, "--cache", out / "cache",
                             "--out", out / "semantic_audit", "--split", "validation"], ["semantic_audit/audit.json"])
    stage("train", ["train", "--cache", out / "cache", "--config", config_path, "--out", out / "models",
                    "--seed", args.seed, "--device", args.device],
          ["models/models.pt", "models/receipt.json", "models/config.json"], True)
    common = ["--models", out / "models", "--cache", out / "cache", "--split", "validation", "--device", args.device]
    stage("validation_replay", ["evaluate", *common, "--out", out / "validation_replay"],
          ["validation_replay/metrics.json", "validation_replay/traces.jsonl"], True)
    summary_runs = [out / "validation_replay"]
    if args.live:
        stage("validation_live", ["evaluate", *common, "--out", out / "validation_live", "--prepared", prepared,
                                   "--raw-root", raw_root, "--responder", out / "responder", "--warmup", "3"],
              ["validation_live/metrics.json", "validation_live/traces.jsonl"], True)
        summary_runs.append(out / "validation_live")
    stage("summarize", ["summarize", "--runs", *summary_runs, "--out", out / "tables"], ["tables/metrics.csv", "tables/table.json"])
    tracked = [prepared / name for name in ("schema.json", "samples.jsonl", "membership.jsonl", "audit.json")]
    tracked += [config_path, Path(__file__), PROJECT / "scripts/run_vision.sh", PROJECT / "tools/run_gpu_limited.py"]
    tracked += sorted((PROJECT / "cbmjev").glob("*.py"))
    if backbone is not None:
        tracked.append(backbone)
    return {"format": "cbmjev-vision-chain-plan-v1", "out": str(out), "project": str(PROJECT),
            "python": python_exe, "prepared": str(prepared), "raw_root": str(raw_root), "data": data,
            "seed": args.seed, "device": args.device, "gpu_selector": args.gpu,
            "resource_guard": {"allocator_budget_gib": args.gpu_memory_gib, "reserve_gib": args.gpu_reserve_gib,
                               "scope": "PyTorch allocator only, not total GPU/process memory or a reservation"},
            "cublas_workspace_config": workspace, "initialization": "LOCAL_CHECKPOINT" if backbone else "EXPLICIT_RANDOM",
            "backbone": str(backbone) if backbone else None, "resolved_training_config": config, "stages": stages,
            "tracked_files": {str(p): {"sha256": file_hash(p), "bytes": p.stat().st_size,
                                         "mtime_ns": p.stat().st_mtime_ns} for p in tracked},
            "test_task_evaluated": False, "calibration_used": False,
            "cache_scope": "existing cache command also computes held-out/test automatic responses; no test task scoring",
            "evidence_status": "EXECUTION_PLAN_NOT_EXPERIMENT_RESULT"}


def resolve_gpu(selector):
    if GPU_UUID.fullmatch(selector):
        return selector
    result = subprocess.run(["nvidia-smi", "-i", selector, "--query-gpu=uuid", "--format=csv,noheader"],
                            check=True, capture_output=True, text=True, timeout=15)
    value = result.stdout.strip()
    if not GPU_UUID.fullmatch(value):
        raise ValueError("nvidia-smi must resolve exactly one physical GPU UUID")
    return value


def stage_command(plan, stage, gpu_uuid):
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=plan["cublas_workspace_config"],
                       CUDA_VISIBLE_DEVICES=gpu_uuid if stage["device"] == "cuda:0" else "")
    command = [plan["python"], "-u"]
    if stage["device"] == "cuda:0":
        command += [str(PROJECT / "tools/run_gpu_limited.py"), "--memory-gib", str(plan["resource_guard"]["allocator_budget_gib"]),
                    "--reserve-gib", str(plan["resource_guard"]["reserve_gib"]), "--"]
    return command + ["-m", "cbmjev", *stage["arguments"]], environment


def check_unchanged(plan, full=False):
    for name, recorded in plan["tracked_files"].items():
        path = Path(name)
        if (not path.is_file() or path.stat().st_size != recorded["bytes"]
                or path.stat().st_mtime_ns != recorded["mtime_ns"]
                or (full and file_hash(path) != recorded["sha256"])):
            raise ValueError("input/config/code/checkpoint changed during chain: " + name)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value, replace=False):
    if not replace:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".pending", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class RunInterrupted(Exception):
    def __init__(self, signum):
        self.signum = signum
        super().__init__("interrupted by signal " + str(signum))


class StageFailed(Exception):
    def __init__(self, name, code):
        self.code = code
        super().__init__("stage {} exited with {}".format(name, code))


def terminate_child(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=10)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def execute(plan):
    # GPU identification is read-only and happens only for actual execution.
    gpu_uuid = resolve_gpu(plan["gpu_selector"]) if plan["device"] == "cuda:0" else None
    check_unchanged(plan)
    out = Path(plan["out"])
    out.mkdir()  # Atomic no-overwrite even if another process raced the preflight.
    (out / "logs").mkdir()
    plan = {**plan, "resolved_gpu_uuid": gpu_uuid, "started_at": utc_now()}
    write_json(out / "plan.json", plan)
    status = {"status": "RUNNING", "started_at": plan["started_at"], "stage": None,
              "completed_stages": [], "plan_sha256": file_hash(out / "plan.json")}
    child = None
    previous_handlers = {}

    def event(kind, **fields):
        with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": kind, "at": utc_now(), **fields}, allow_nan=False) + "\n")

    def interrupt(signum, _frame):
        raise RunInterrupted(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[signum] = signal.signal(signum, interrupt)
        event("CHAIN_START", plan_sha256=status["plan_sha256"])
        for stage in plan["stages"]:
            check_unchanged(plan)
            status["stage"] = stage["name"]
            write_json(out / "status.json", status, replace=True)
            command, environment = stage_command(plan, stage, gpu_uuid)
            event("STAGE_START", name=stage["name"], command=command, log=stage["log"],
                  cuda_visible_devices=environment["CUDA_VISIBLE_DEVICES"])
            print("[{}] start; log={}".format(stage["name"], stage["log"]), flush=True)
            started = time.monotonic()
            with Path(stage["log"]).open("xb") as log:
                child = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                code = child.wait()
                child = None
            event("STAGE_EXIT", name=stage["name"], exit_code=code, elapsed_seconds=time.monotonic() - started)
            if code:
                raise StageFailed(stage["name"], code)
            if any(not Path(name).is_file() or not Path(name).stat().st_size for name in stage["expected_artifacts"]):
                raise ValueError("stage succeeded but required artifact is missing/empty: " + stage["name"])
            status["completed_stages"].append(stage["name"])
        check_unchanged(plan, full=True)
        status.update(status="COMPLETED", exit_code=0, finished_at=utc_now(),
                      evidence_status="VALIDATION_WORKFLOW_COMPLETED_NOT_PAPER_RESULT")
        write_json(out / "status.json", status, replace=True)
        event("CHAIN_COMPLETED", completed_stages=status["completed_stages"])
        write_json(out / "COMPLETED.json", status)
        print("Completed validation workflow: " + str(out / "COMPLETED.json"), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        terminate_child(child)
        interrupted = isinstance(exc, (RunInterrupted, KeyboardInterrupt))
        code = 128 + getattr(exc, "signum", signal.SIGINT) if interrupted else getattr(exc, "code", 1)
        if code < 0:
            code = 128 - code
        status.update(status="INTERRUPTED" if interrupted else "FAILED", exit_code=code,
                      finished_at=utc_now(), error=str(exc))
        write_json(out / "status.json", status, replace=True)
        event("CHAIN_" + status["status"], stage=status["stage"], exit_code=code, error=str(exc))
        write_json(out / (status["status"] + ".json"), status)
        print("{}; partial artifacts retained at {}".format(exc, out), file=sys.stderr, flush=True)
        return code
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
            return 0
        return execute(plan)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print("Preflight/launch failed: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

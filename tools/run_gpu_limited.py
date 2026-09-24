#!/usr/bin/env python3
"""Run one Python module on one visible GPU with an explicit allocator budget.

This limits PyTorch's caching allocator, not driver/library allocations or
other users' processes. Keep additional free-memory headroom and measure usage.
"""
import argparse
import json
import math
import os
import resource
import runpy
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-gib", type=float, default=4.0)
    parser.add_argument("--reserve-gib", type=float, default=8.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if len(command) < 2 or command[0] != "-m":
        parser.error("expected -- -m module [arguments]")
    if any(not math.isfinite(v) or v <= 0 for v in (args.memory_gib, args.reserve_gib)):
        parser.error("memory and reserve GiB must be finite and positive")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"):
        parser.error("pin one physical GPU UUID before invoking this runner")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("exactly one usable CUDA device must be visible")
    free, total = torch.cuda.mem_get_info(0)
    budget = int(args.memory_gib * 2**30)
    reserve = int(args.reserve_gib * 2**30)
    if budget >= total or free < budget + reserve:
        parser.error("insufficient free GPU memory for this budget plus reserve")
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats(0)
    context = {"gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"], "logical_device": "cuda:0",
               "allocator_budget_bytes": budget, "required_free_reserve_bytes": reserve,
               "free_bytes_before": free, "total_bytes": total,
               "scope": "PyTorch allocator only; driver/libraries may allocate additional memory",
               "shared_gpu_timing_is_not_paper_evidence": True}
    print(json.dumps({"event": "RESOURCE_GUARD_START", **context}), file=sys.stderr, flush=True)
    started, success = time.monotonic(), False
    # Match python -m import semantics even though this bootstrap is a script.
    sys.path.insert(0, os.getcwd())
    sys.argv = command[1:]
    try:
        runpy.run_module(command[1], run_name="__main__", alter_sys=True)
        success = True
    except SystemExit as exc:
        success = exc.code in (None, 0)
        raise
    finally:
        print(json.dumps({"event": "RESOURCE_GUARD_END", "success": success,
                          "elapsed_seconds": time.monotonic() - started,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                          "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                          "process_max_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          "rss_unit": "KiB on Linux", **context}), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

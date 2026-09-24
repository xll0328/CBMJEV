#!/usr/bin/env python3
"""Short shared-device diagnostic; not a paper latency benchmark."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from cbmjev.contracts import Schema
from cbmjev.io import environment, read_json, write_json
from cbmjev.learning import (encode_actions, encode_states, MaskedHead,
                             ActionController, normalize_config, action_targets)


def bulk_actions(actions, schema, device):
    rows = [[0.0] * (schema.num_groups + 1) for _ in actions]
    for row, action in zip(rows, actions):
        for index in action or (-1,):
            row[index] = 1.0
    return torch.tensor(rows, dtype=torch.float32, device=device).reshape(
        len(actions), schema.num_groups + 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schema", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    p.add_argument("--iterations", type=int, default=100)
    args = p.parse_args()
    if Path(args.out).exists() or args.iterations < 1:
        raise ValueError("new output and positive iterations required")
    schema = Schema.from_dict(read_json(args.schema))
    torch.set_num_threads(1)
    def sync():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
    measurements = []
    for size in (1, 64, 256):
        actions = [() if i % 3 == 0 else (i % schema.num_groups,) for i in range(size)]
        assert torch.equal(encode_actions(actions, schema, args.device),
                           bulk_actions(actions, schema, args.device))
        result = {"batch_size": size, "exact_equal": True}
        for name, fn in (("existing", encode_actions), ("bulk", bulk_actions)):
            for _ in range(5):
                fn(actions, schema, args.device)
            sync()
            timings = []
            for _ in range(3):
                start = time.perf_counter()
                for _ in range(args.iterations):
                    fn(actions, schema, args.device)
                    sync()
                timings.append((time.perf_counter() - start) / args.iterations)
            result[name + "_seconds"] = timings
        result["median_ratio_existing_over_bulk"] = (
            statistics.median(result["existing_seconds"]) / statistics.median(result["bulk_seconds"]))
        measurements.append(result)
    step_timings = {}
    cfg = normalize_config({"device": args.device, "hidden": 128, "objective": "value"}, schema)
    actions = [() if i % 3 == 0 else (i % schema.num_groups,) for i in range(64)]
    before = [schema.empty_state()] * 64
    examples = []
    for i, action in enumerate(actions):
        after = list(before[i])
        for atom in schema.expand(action):
            after[atom] = 0
        examples.append((before[i], action, tuple(after), i % schema.num_classes))
    for name, fn in (("existing", encode_actions), ("bulk", bulk_actions)):
        torch.manual_seed(17)
        head = MaskedHead(schema, cfg)
        head.network.eval().requires_grad_(False)
        controller = ActionController(schema, cfg)
        optimizer = torch.optim.AdamW(controller.network.parameters(), lr=0.001)
        timings = []
        for step in range(25):
            sync()
            start = time.perf_counter()
            target = action_targets(head, examples, "value")
            x = torch.cat((encode_states(before, schema, args.device),
                           fn(actions, schema, args.device)), dim=-1)
            loss = torch.nn.functional.mse_loss(controller.network(x).flatten(), target)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite diagnostic loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            float(loss.detach())
            sync()
            elapsed = time.perf_counter() - start
            if step >= 5:
                timings.append(elapsed)
        step_timings[name] = {"median_seconds": statistics.median(timings),
                              "seconds": timings, "final_loss": float(loss.detach())}
    report = {"scope": "ENCODING_MICROBENCHMARK_SHARED_DEVICE_NOT_END_TO_END_OR_PAPER_SPEEDUP",
              "representative_controller_step": step_timings,
              "device": args.device, "environment": environment(), "measurements": measurements}
    write_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

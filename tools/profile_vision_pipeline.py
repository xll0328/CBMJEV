#!/usr/bin/env python3
"""Small real-image preprocessing diagnostic; does not train or access test."""
import argparse
import io
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from cbmjev.io import write_json, environment
from cbmjev.pipeline import load_prepared
from cbmjev.responders import SharedVisionResponder
from cbmjev.runtime import load_payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if Path(args.out).exists():
        raise ValueError("output exists")
    schema, rows, membership = load_prepared(args.prepared)
    selected = [r for r in rows if membership[r["sample_id"]]["split"] == "responder_fit"][:4]
    start = time.perf_counter()
    payloads = [load_payload(r["input"], args.raw_root) for r in selected]
    read_seconds = time.perf_counter() - start
    model = SharedVisionResponder(schema).to("cuda:0").eval()
    default_threads = torch.get_num_threads()
    timings, reference = [], None
    for threads in dict.fromkeys((default_threads, 1, 4)):
        torch.set_num_threads(threads)
        start = time.perf_counter()
        images = []
        for payload in payloads:
            for raw in payload.images:
                with Image.open(io.BytesIO(raw)) as image:
                    images.append(model.transform(image.convert("RGB")))
        batch = torch.stack(images)
        preprocessing_seconds = time.perf_counter() - start
        if reference is None:
            reference = batch.clone()
        equal = torch.equal(reference, batch)
        with torch.no_grad():
            model.encoder(batch.to("cuda:0"))
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(3):
                model.encoder(batch.to("cuda:0"))
        torch.cuda.synchronize()
        timings.append({"threads": threads, "preprocessing_seconds": preprocessing_seconds,
                        "three_forward_seconds": time.perf_counter() - start,
                        "preprocessed_tensor_exact_equal": equal})
    report = {"scope": "FOUR_TRAIN_IMAGES_SHARED_GPU_DIAGNOSTIC_NOT_TRAINING_BENCHMARK",
              "read_seconds": read_seconds, "measurements": timings, "environment": environment()}
    write_json(args.out, report)
    print(report)


if __name__ == "__main__":
    main()

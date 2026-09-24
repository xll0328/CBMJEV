#!/usr/bin/env python3
"""Real training-subset IO diagnostic; never a formal throughput result."""
import argparse
import hashlib
from functools import partial
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from cbmjev.io import environment, file_hash, write_json
from cbmjev.pipeline import LazyConceptExamples, load_prepared, code_fingerprint
from cbmjev.responders import SharedVisionResponder, fit_responder
from cbmjev.runtime import load_payload


def profile_answer_transfer(schema, rows, raw_root):
    """Compare fresh sessions on the same pixels/random model; no task metrics."""
    from cbmjev.responders import _bulk_semantic_answers
    def legacy(distributions, counts, threshold):
        return tuple(count if threshold is not None and float(p.max()) < threshold
                     else int(p.argmax()) for p, count in zip(distributions, counts))
    torch.manual_seed(17)
    model = SharedVisionResponder(schema).to("cuda:0").eval()
    payloads = [load_payload(row["input"], raw_root) for row in rows]
    atoms = tuple(range(schema.num_atoms))
    results, expected = [], None
    # Warm both paths; alternate order across the two measured pairs.
    for name, fn in (("legacy", legacy), ("bulk", _bulk_semantic_answers)):
        with patch("cbmjev.responders._bulk_semantic_answers", fn):
            model.respond(payloads[0], atoms)
    for name, fn in (("legacy", legacy), ("bulk", _bulk_semantic_answers),
                     ("bulk", _bulk_semantic_answers), ("legacy", legacy)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with patch("cbmjev.responders._bulk_semantic_answers", fn):
            answers = [model.respond(payload, atoms) for payload in payloads]
        torch.cuda.synchronize()
        if expected is None:
            expected = answers
        if answers != expected:
            raise ValueError("bulk answer extraction changed discrete model responses")
        results.append({"extraction": name, "seconds": time.perf_counter() - start,
                        "images": len(payloads), "answers_equal": True})
    return {"scope": "RANDOM_MODEL_TRAIN_IMAGES_SHARED_GPU_FRESH_SESSION_DIAGNOSTIC_NOT_PAPER_SPEEDUP",
            "results": results, "all_answers_exact_equal": True,
            "model_initialization": "random seed17; not trained checkpoint or quality evidence",
            "timing_excludes": "file IO and payload loading; includes encode and all heads each session",
            "model_sha256": digest(model), "source_code_hash": code_fingerprint(),
            "script_sha256": file_hash(__file__), "environment": environment()}


def digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--out", required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--compare-compression", action="store_true",
                   help="compare lossless PNG level 6/0 with synchronous/prefetched reads")
    mode.add_argument("--compare-answer-transfer", action="store_true",
                      help="compare scalar and bulk GPU answer extraction on identical fresh sessions")
    args = p.parse_args()
    if Path(args.out).exists():
        raise ValueError("output exists")
    schema, records, membership = load_prepared(args.prepared)
    rows = [r for r in records if membership[r["sample_id"]]["split"] == "responder_fit"][:16]
    if len(rows) < 16:
        raise ValueError("requires sixteen training examples")
    if args.compare_answer_transfer:
        output = profile_answer_transfer(schema, rows, args.raw_root)
        write_json(args.out, output)
        print(output)
        return
    examples = LazyConceptExamples(rows, schema, args.raw_root)
    torch.set_num_threads(2)
    results = []
    variants = ((0, 6), (0, 0), (2, 0), (2, 6)) if args.compare_compression else (
        (0, 0), (2, 0), (2, 0), (0, 0))
    for workers, compression in variants:
        torch.manual_seed(17)
        model = SharedVisionResponder(schema)
        torch.cuda.synchronize()
        start = time.perf_counter()
        # Diagnostic-only loader override. No running production source is edited;
        # the prefetch executor exits before the next variant changes this hook.
        with patch("cbmjev.pipeline.load_payload", partial(load_payload, png_compress_level=compression)):
            model, report = fit_responder(model, examples, epochs=1, batch_size=4,
                                         seed=17, device="cuda:0", learning_rate=0.0001,
                                         input_workers=workers, input_prefetch_batches=2)
        torch.cuda.synchronize()
        results.append({"workers": workers, "png_compress_level": compression,
                        "seconds": time.perf_counter() - start,
                        "model_sha256": digest(model), "training_loss": report["training_loss"]})
        del model
    output = {"scope": "16_TRAIN_IMAGES_SHARED_GPU_DIAGNOSTIC_INCLUDES_LABEL_VALIDATION_NOT_PAPER_SPEEDUP",
              "results": results, "all_weights_exact_equal": len({r["model_sha256"] for r in results}) == 1,
              "all_losses_exact_equal": all(r["training_loss"] == results[0]["training_loss"] for r in results),
              "sample_ids": [r["sample_id"] for r in rows], "source_code_hash": code_fingerprint(),
              "profiler_sha256": file_hash(__file__), "compare_compression": args.compare_compression,
              "prepared_samples_sha256": file_hash(Path(args.prepared) / "samples.jsonl"),
              "environment": environment()}
    write_json(args.out, output)
    print(output)
    if not output["all_weights_exact_equal"] or not output["all_losses_exact_equal"]:
        raise ValueError("prefetch changed deterministic training results")


if __name__ == "__main__":
    main()

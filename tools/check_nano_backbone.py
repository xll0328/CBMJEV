"""Local-only backbone readiness check; not a downstream performance experiment."""
import argparse
import json
from pathlib import Path

import torch

from cbmjev.nanojev import load_local_nano
from cbmjev.pipeline import code_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    source = code_fingerprint()
    torch.manual_seed(17)
    scorer = load_local_nano(args.backbone, device=args.device, max_length=512)
    scorer.eval()
    prompts = ["Observed concepts: none. Candidate: stop. Estimate classification error.",
               "Observed concepts: food=positive; service=unknown. Candidate: acquire service. "
               "Estimate the classification error after acquiring this concept."]
    outputs = {}
    with torch.no_grad():
        single = torch.cat([scorer.score_prompts([p]) for p in prompts])
        for padding in ("left", "right"):
            scorer.tokenizer.padding_side = padding
            batch = scorer.score_prompts(prompts)
            torch.testing.assert_close(single, batch, atol=2e-4, rtol=2e-4)
            outputs[padding] = {"max_abs_batch_difference": float((single-batch).abs().max()),
                                "scores": batch.cpu().tolist()}
    # Frozen backbone must still permit the trainable decision head to learn.
    scorer.train()
    logits = scorer.score_prompts(prompts)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([0., 1.], device=logits.device))
    loss.backward()
    if not torch.isfinite(loss) or scorer.head.weight.grad is None:
        raise ValueError("invalid trainable-head gradient")
    if not torch.isfinite(scorer.head.weight.grad).all():
        raise ValueError("nonfinite head gradient")
    if any(p.grad is not None for p in scorer.backbone.parameters()):
        raise ValueError("frozen backbone received gradients")
    if source != code_fingerprint():
        raise ValueError("source changed during readiness check")
    report = {"status": "BACKBONE_READINESS_ONLY", "paper_evidence": False,
              "official_nanojev_reproduction": False, "seed": 17,
              "backbone_manifest": scorer.backbone_source_manifest,
              "source_code_hash": source, "device": args.device,
              "dtype": str(scorer.head.weight.dtype), "batch_padding_checks": outputs,
              "head_gradient_norm": float(scorer.head.weight.grad.norm()),
              "frozen_backbone_gradient_check": True,
              "decoder_cache_disabled": hasattr(scorer.backbone.config, "use_cache"),
              "prompts": prompts, "tolerance": {"atol": 2e-4, "rtol": 2e-4}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"out": str(out), "status": report["status"]}))


if __name__ == "__main__":
    main()

"""One bounded Qwen3 prompt/format audit on public policy_fit pilot cases."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_cebab_second_source import ASPECTS, _prompt, _read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("pilot", "prepared", "model", "out"):
        parser.add_argument("--" + field, required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise ValueError("output already exists")
    pilot = list(_read_jsonl(args.pilot))
    if any(r["split"] != "policy_fit" for r in pilot):
        raise ValueError("audit accepts policy_fit pilot only")
    ids = {r["sample_id"] for r in pilot}
    by_id = {r["sample_id"]: r for r in _read_jsonl(args.prepared) if r["sample_id"] in ids}
    selected = []
    for aspect_id in range(4):
        for gold in range(3):
            candidates = sorted((r for r in pilot if r["gold"][aspect_id] == gold), key=lambda r: r["sample_id"])
            if candidates:
                selected.append((candidates[0], aspect_id))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if torch.cuda.device_count() != 1:
        raise ValueError("isolate physical GPU1 with CUDA_VISIBLE_DEVICES=1")
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
             trust_remote_code=False, torch_dtype=torch.bfloat16).to("cuda:0").eval()
    digits = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(3)]
    observations = []
    for row, aspect_id in selected:
        prompt = _prompt(tokenizer, by_id[row["sample_id"]]["input"]["text"], ASPECTS[aspect_id])
        encoded = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            logits = model(**encoded, use_cache=False).logits[0, -1].float()
            top = logits.topk(8)
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=4,
                                       use_cache=True, pad_token_id=tokenizer.eos_token_id)
        tokens = generated[0, encoded["input_ids"].shape[1]:]
        observations.append({"sample_id": row["sample_id"], "aspect": ASPECTS[aspect_id],
          "gold": row["gold"][aspect_id], "scored": row["source_b"][aspect_id],
          "raw_generated": tokenizer.decode(tokens, skip_special_tokens=False),
          "digit_logits": [float(logits[i]) for i in digits],
          "digit_mass": float(logits.softmax(-1)[digits].sum()),
          "top_tokens": [tokenizer.decode([int(i)], skip_special_tokens=False) for i in top.indices]})
    report = {"scope": "12 deterministic policy_fit pilot gold-stratified examples; engineering audit, not model selection on validation",
              "observations": observations}
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

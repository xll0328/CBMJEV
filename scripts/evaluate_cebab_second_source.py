"""Generate a second, real typed CEBaB measurement from a local Qwen model.

This is a fixed zero-shot candidate scorer, not official Jev/NanoJev. The
model sees only the review text and one public aspect definition per query.
No task label, gold aspect, split, or first-source answer enters its prompt.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

ASPECTS = ("food", "noise", "ambiance", "service")
GLOSSES = {
    "food": "food quality and taste; negative means bad food, positive means good food",
    "noise": "noise level; negative means unpleasantly noisy, positive means pleasantly quiet",
    "ambiance": "restaurant atmosphere and decor; negative means unpleasant, positive means pleasant",
    "service": "staff service; negative means poor service, positive means good service",
}
PROMPT_VERSION = "cebab-qwen3-typed-digits-v2-explicit-evidence"
SYSTEM = (
    "You classify only the named aspect of a restaurant review. Use explicit "
    "evidence about that aspect in the quoted review. If the aspect is not "
    "discussed, answer 2, even when the overall review sounds positive or "
    "negative. A complaint about a different aspect is not evidence for this "
    "aspect. Return exactly one digit: 0 for explicit negative evidence, "
    "1 for explicit positive evidence, 2 for absent, mixed, or unclear evidence. "
    "Do not predict the overall rating."
)


_ROLE_FIELD = re.compile(r'(?<!\\)"split"\s*:\s*"([^"]+)"')
_ID_FIELD = re.compile(r'(?<!\\)"sample_id"\s*:\s*"([^"]+)"')


def _read_jsonl(path, *, roles=None, ids=None):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                if roles is not None:
                    match = _ROLE_FIELD.search(line)
                    if match is None:
                        raise ValueError("JSONL line lacks split field")
                    if match.group(1) not in roles:
                        continue
                if ids is not None:
                    match = _ID_FIELD.search(line)
                    if match is None:
                        raise ValueError("JSONL line lacks sample_id field")
                    if match.group(1) not in ids:
                        continue
                yield json.loads(line)


def _family_key(group_id):
    return hashlib.sha256(("cbmjev-second-source-v1|" + group_id).encode()).hexdigest()


def select_cases(cache_path, prepared_path, roles, limit_per_role):
    allowed = {"head_fit", "policy_fit", "validation"}
    if not roles or set(roles) - allowed or len(set(roles)) != len(roles):
        raise ValueError("roles must be distinct head_fit/policy_fit/validation only")
    cache = list(_read_jsonl(cache_path, roles=set(roles)))
    selected = []
    for role in roles:
        candidates = [r for r in cache if r["split"] == role]
        candidates.sort(key=lambda r: (_family_key(r["group_id"]), r["sample_id"]))
        if limit_per_role is not None:
            # One row per family in pilot, so its reported n is a family n.
            seen = set()
            family_cases = []
            for row in candidates:
                if row["group_id"] not in seen:
                    seen.add(row["group_id"])
                    family_cases.append(row)
                if len(family_cases) == limit_per_role:
                    break
            candidates = family_cases
        selected.extend(candidates)
    groups = defaultdict(set)
    for row in selected:
        groups[row["split"]].add(row["group_id"])
    for left in roles:
        for right in roles:
            if left != right and groups[left] & groups[right]:
                raise ValueError("family crosses selected roles")
    ids = {r["sample_id"] for r in selected}
    if len(ids) != len(selected):
        raise ValueError("duplicate cache sample")
    prepared = {r["sample_id"]: r for r in _read_jsonl(prepared_path, ids=ids)}
    if prepared.keys() != ids:
        raise ValueError("prepared/cache sample IDs differ")
    for row in selected:
        source = prepared[row["sample_id"]]
        if row["group_id"] != source["group_id"] or row["split"] not in allowed:
            raise ValueError("family or role mismatch")
        if len(row["z"]) != 4 or len(source["concepts"]) != 4:
            raise ValueError("not four-aspect CEBaB")
        if source["input"]["modality"] != "text" or not source["input"]["text"]:
            raise ValueError("non-text or empty review")
    return selected, prepared


def _prompt(tokenizer, text, aspect):
    user = (
        "Aspect: " + aspect + " (" + GLOSSES[aspect] + ")\n"
        "Review:\n" + text + "\n\nDoes this review explicitly discuss this aspect? "
        "If not, answer 2. Otherwise classify its explicit sentiment. "
        "Answer (0, 1, or 2):"
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def classify_batch(model, tokenizer, cases, *, max_tokens):
    import torch

    texts = [_prompt(tokenizer, text, aspect) for text, aspect in cases]
    encoded = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
    lengths = encoded["attention_mask"].sum(1).tolist()
    if max(lengths) > max_tokens:
        raise ValueError(f"prompt length {max(lengths)} exceeds declared max_tokens={max_tokens}")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in encoded.items()}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[:, -1, :]
        digit_ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(3)]
        if any(len(x) != 1 for x in digit_ids):
            raise ValueError("typed digit candidate is not one token")
        choice_logits = logits[:, [x[0] for x in digit_ids]].float()
        choices = choice_logits.argmax(1).cpu().tolist()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    return choices, lengths, seconds


def aggregate(rows):
    totals = Counter()
    by_role = defaultdict(Counter)
    by_aspect = {a: Counter() for a in ASPECTS}
    for row in rows:
        c = by_role[row["split"]]
        c["cases"] += 1
        totals["cases"] += 1
        for j, aspect in enumerate(ASPECTS):
            gold = row["gold"][j]
            if gold is None:
                continue
            for target in (totals, c, by_aspect[aspect]):
                target["labeled"] += 1
                target["source_a_correct"] += row["source_a"][j] == gold
                target["source_b_correct"] += row["source_b"][j] == gold
                target["a_wrong_b_correct"] += row["source_a"][j] != gold and row["source_b"][j] == gold
                target["a_correct_b_wrong"] += row["source_a"][j] == gold and row["source_b"][j] != gold
                target["both_wrong"] += row["source_a"][j] != gold and row["source_b"][j] != gold
                target["disagree"] += row["source_a"][j] != row["source_b"][j]
    return {"overall": dict(totals), "by_role": {k: dict(v) for k, v in by_role.items()},
            "by_aspect": {k: dict(v) for k, v in by_aspect.items()}}


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    roles = tuple(args.roles.split(","))
    cases, prepared = select_cases(args.cache, args.prepared, roles, args.limit_per_role)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
               trust_remote_code=False, torch_dtype=torch.bfloat16).to(args.device).eval()
    if args.device.startswith("cuda") and torch.cuda.device_count() != 1:
        raise ValueError("isolate exactly one allowed physical GPU through CUDA_VISIBLE_DEVICES")
    encoded_cases = []
    for row in cases:
        text = prepared[row["sample_id"]]["input"]["text"]
        encoded_cases.extend((text, aspect) for aspect in ASPECTS)
    answers, lengths, call_timings = [], [], []
    elapsed = 0.0
    for start in range(0, len(encoded_cases), args.batch_size):
        choices, prompt_lengths, seconds = classify_batch(
            model, tokenizer, encoded_cases[start:start + args.batch_size], max_tokens=args.max_tokens)
        answers.extend(choices)
        lengths.extend(prompt_lengths)
        elapsed += seconds
        call_timings.append({"call_index": len(call_timings), "questions": len(choices),
                             "seconds": seconds, "prompt_tokens": prompt_lengths})
        if (start // args.batch_size + 1) % 25 == 0:
            print(json.dumps({"processed_questions": len(answers), "total_questions": len(encoded_cases),
                              "forward_seconds": round(elapsed, 2)}), flush=True)
    output = []
    for index, old in enumerate(cases):
        source = prepared[old["sample_id"]]
        output.append({"sample_id": old["sample_id"], "group_id": old["group_id"],
            "split": old["split"], "y": old["y"],
            "gold": [c["value"] for c in source["concepts"]],
            "source_a": old["z"], "source_b": answers[4*index:4*(index+1)],
            "prompt_tokens": lengths[4*index:4*(index+1)]})
    with (out / "responses.jsonl").open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "call_timings.jsonl").open("w", encoding="utf-8") as handle:
        for row in call_timings:
            handle.write(json.dumps(row) + "\n")
    report = {"format": "cbmjev-cebab-qwen-second-source-v2", "status":
              "PILOT" if args.limit_per_role is not None else "EXPLORATORY_DEVELOPMENT",
              "test_evaluated": False, "source_b": "Qwen/Qwen3-0.6B zero-shot digit candidate scorer",
              "source_b_model_path": args.model, "model_revision": Path(args.model).name,
              "model_commit": "c1899de289a04d12100db370d81485cdf75e47ca",
              "model_license": "Apache-2.0", "prompt_version": PROMPT_VERSION,
              "category_mapping": {"0": "Negative", "1": "Positive", "2": "unknown"},
              "information": "hard-only answers; normalized digit logits are neither exposed nor fused",
              "roles": list(roles), "limit_per_role": args.limit_per_role,
              "role_families": {role: len({r["group_id"] for r in output if r["split"] == role}) for role in roles},
              "batch_size": args.batch_size, "max_tokens": args.max_tokens,
              "cost_scope": "one GPU forward per batch of candidate questions, excludes model load and tokenizer time; four concept calls for a review can share a batch; serial policies may incur separate forwards",
              "forward_seconds": elapsed, "questions_per_forward_second": len(answers)/elapsed,
              "prompt_token_max": max(lengths), "prompt_token_mean": sum(lengths)/len(lengths),
              "metrics": aggregate(output)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "report": report}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("cache", "prepared", "model", "out"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--roles", default="policy_fit")
    parser.add_argument("--limit-per-role", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.limit_per_role is not None and args.limit_per_role < 1:
        parser.error("--limit-per-role must be positive")
    if args.batch_size < 1 or args.max_tokens < 1:
        parser.error("batch-size and max-tokens must be positive")
    run(args)


if __name__ == "__main__":
    main()

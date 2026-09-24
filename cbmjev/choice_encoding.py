"""Bounded frozen-backbone encoding of visible Choice questions, no disk cache."""
import copy
import hashlib
import json

import torch

from .choice_features import CHOICE_FEATURE_VERSION, choice_prompts
from .contracts import stable_hash
from .nanojev import NanoCandidateScorer


CHOICE_ENCODING_VERSION = "cbmjev-frozen-choice-encoding-v1"


def _frozen_eval(scorer):
    if not isinstance(scorer, NanoCandidateScorer):
        raise ValueError("NanoCandidateScorer required")
    if (not scorer.freeze_backbone or any(m.training for m in scorer.backbone.modules())
            or any(p.requires_grad for p in scorer.backbone.parameters())):
        raise ValueError("encoding requires a frozen eval-mode backbone")


def _hash_item(hasher, value):
    blob = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    hasher.update(len(blob).to_bytes(8, "big")); hasher.update(blob)


def encode_frozen_choice_features(scorer, questions, schema, *, max_questions,
                                  max_padded_tokens, max_candidates_per_batch):
    """Return (tuple[CPU raw_features[K,D]], accounting) for a bounded tuple.

    Every prompt is length-checked BEFORE the first backbone call. Candidate
    encoding may cross question boundaries, but no candidate/question is dropped
    or reordered. Caller must retain complete sets for downstream Choice loss.
    Features precede trainable normalization. No full-dataset accumulation,
    checkpoint load, tokenizer download, or runtime-policy evaluation occurs.
    """
    _frozen_eval(scorer)
    for value in (max_questions, max_padded_tokens, max_candidates_per_batch):
        if type(value) is not int or value < 1:
            raise ValueError("explicit positive integer encoding limits required")
    if type(questions) is not tuple or not 1 <= len(questions) <= max_questions:
        raise ValueError("nonempty question tuple must satisfy max_questions")
    prompts, lengths, counts = [], [], []
    prompt_digest, token_digest = hashlib.sha256(), hashlib.sha256()
    for question in questions:
        paths = choice_prompts(question, schema)
        counts.append(len(paths))
        for prompt in paths:
            # Exact scorer.features tokenizer flags, applied one path at a time
            # so preflight itself cannot allocate an unbounded padded matrix.
            tokens = scorer.tokenizer([prompt], padding=True, truncation=False, return_tensors="pt")
            ids, mask = tokens["input_ids"], tokens["attention_mask"]
            if (ids.ndim != 2 or ids.shape[0] != 1 or ids.shape != mask.shape
                    or ids.shape[1] < 1 or not (mask == 1).all()):
                raise ValueError("single-path tokenization must contain only valid unpadded tokens")
            length = ids.shape[1]
            if length > scorer.max_length:
                raise ValueError("prompt exceeds scorer.max_length; no truncation permitted")
            if length > max_padded_tokens:
                raise ValueError("one candidate exceeds max_padded_tokens")
            prompts.append(prompt); lengths.append(length)
            _hash_item(prompt_digest, prompt); _hash_item(token_digest, ids[0].tolist())
    chunks, start, width = [], 0, 0
    for index, length in enumerate(lengths):
        candidate_width = max(width, length)
        size = index - start + 1
        if size > max_candidates_per_batch or candidate_width * size > max_padded_tokens:
            chunks.append((start, index, width)); start, width = index, length
        else:
            width = candidate_width
    chunks.append((start, len(prompts), width))
    manifest = copy.deepcopy(getattr(scorer, "backbone_source_manifest", None))
    manifest_hash = stable_hash(manifest) if manifest is not None else None
    tokenizer_options = {"padding_side": getattr(scorer.tokenizer, "padding_side", None),
        "pad_token_id": getattr(scorer.tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(scorer.tokenizer, "eos_token_id", None),
        "class": type(scorer.tokenizer).__module__ + "." + type(scorer.tokenizer).__qualname__}
    encoded = []
    with torch.no_grad():
        for first, last, width in chunks:
            _frozen_eval(scorer)
            values = scorer.features(prompts[first:last])
            if (values.shape != (last-first, scorer.hidden_size)
                    or not values.is_floating_point() or not torch.isfinite(values).all()):
                raise ValueError("backbone returned invalid raw Choice features")
            encoded.append(values.detach().cpu().clone())
    _frozen_eval(scorer)
    if stable_hash(getattr(scorer, "backbone_source_manifest", None)) != stable_hash(manifest):
        raise ValueError("backbone source manifest changed during encoding")
    flat = torch.cat(encoded, dim=0)
    outputs = tuple(flat.split(counts, dim=0))
    report = {"format": CHOICE_ENCODING_VERSION, "serializer": CHOICE_FEATURE_VERSION,
        "schema_hash": schema.hash, "questions": len(questions), "candidate_paths": len(prompts),
        "question_candidate_counts": counts, "actual_tokens": sum(lengths),
        "padded_tokens": sum((b-a)*w for a,b,w in chunks), "forward_calls": len(chunks),
        "max_prompt_tokens": max(lengths), "scorer_max_length": scorer.max_length,
        "max_questions": max_questions, "max_padded_tokens": max_padded_tokens,
        "max_candidates_per_batch": max_candidates_per_batch,
        "chunks": [{"first_candidate": a, "end_candidate_exclusive": b,
                    "max_length": w, "padded_tokens": (b-a)*w} for a,b,w in chunks],
        "prompt_sequence_sha256": prompt_digest.hexdigest(),
        "token_id_sequence_sha256": token_digest.hexdigest(),
        "tokenizer_options": tokenizer_options, "tokenizer_options_sha256": stable_hash(tokenizer_options),
        "backbone_manifest": manifest, "backbone_manifest_record_sha256": manifest_hash,
        "raw_feature_dtype": str(flat.dtype), "raw_feature_storage_bytes": flat.numel()*flat.element_size(),
        "normalization_cached": False, "gradients_enabled": False, "paper_speed_evidence": False,
        "provenance_scope": "Attached load-time HF manifest includes tokenizer/backbone files when present; not a live weight hash. Token IDs bind these exact inputs, not arbitrary tokenizer behavior. Fake/injected scorers may have no manifest.",
        "memory_scope": "Bounded supplied question block; stores its prompts, raw CPU chunks plus concatenated CPU features (temporary duplication), and one GPU encoding chunk. This token budget is not an allocator/activation guarantee.",
        "precision_scope": "Chunk batch shapes may change floating-point roundoff; semantic order preserved, bitwise equality not guaranteed."}
    return outputs, report

"""Local candidate-scalar adaptation, NOT official Jev or full NanoJev reproduction."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import ModelInput
from .responders import sanitize_model_input, validate_atoms, validate_examples, schema_digest

UPSTREAM_COMMIT = "76fdfc9ecdca45a9bcef17991a07d3041a87685a"
UPSTREAM_URL = f"https://github.com/TianyuCodings/NanoJev/tree/{UPSTREAM_COMMIT}"


def backbone_manifest(local_directory):
    """Fingerprint standard HF root model/tokenizer artifacts, never output subtrees."""
    directory = Path(local_directory)
    if not directory.is_dir():
        raise ValueError("local backbone directory is missing")
    files = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix not in {".json", ".safetensors", ".bin", ".model", ".txt"}:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files[path.name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    if not files:
        raise ValueError("no local HF model/tokenizer artifacts to fingerprint")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"mode": "LOCAL_HF_ROOT_CONTENT_MANIFEST", "files": files,
            "sha256": hashlib.sha256(encoded).hexdigest()}


def last_valid_pool(hidden, attention_mask):
    """Index maximum valid position, NOT sum(mask)-1 (which breaks left padding)."""
    mask = attention_mask.bool()
    if hidden.ndim != 3 or tuple(hidden.shape[:2]) != tuple(mask.shape) or not mask.any(1).all():
        raise ValueError("invalid hidden/mask shape or all-padding input")
    indices = torch.arange(mask.shape[1], device=mask.device).expand_as(mask).masked_fill(~mask, -1).max(1).values
    return hidden[torch.arange(len(hidden), device=hidden.device), indices]


class NanoCandidateScorer(nn.Module):
    def __init__(self, backbone, tokenizer, hidden_size=None, max_length=512, freeze_backbone=True,
                 feature_normalization="none"):
        super().__init__()
        self.backbone, self.tokenizer = backbone, tokenizer
        width = hidden_size or getattr(backbone.config, "hidden_size", None)
        if type(width) is not int or width < 1 or type(max_length) is not int or max_length < 1:
            raise ValueError("hidden_size and max_length must be positive integers")
        self.hidden_size, self.max_length = width, max_length
        self.freeze_backbone = freeze_backbone
        self.backbone_local_path = None
        self.backbone_source_manifest = None
        if feature_normalization not in ("none", "layernorm"):
            raise ValueError("feature_normalization must be none or layernorm")
        self.feature_normalization = feature_normalization
        self.head = nn.Linear(width, 1)
        self.norm = nn.LayerNorm(width) if feature_normalization == "layernorm" else nn.Identity()
        parameter = next(backbone.parameters(), None)
        if parameter is not None:
            self.head.to(device=parameter.device, dtype=parameter.dtype)
            self.norm.to(device=parameter.device, dtype=parameter.dtype)
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def score_prompts(self, prompts):
        return self.score_features(self.features(prompts))

    def score_features(self, features):
        """Apply trainable normalization and scalar head to RAW backbone features.

        Frozen feature caching must never cache normalization outputs: affine
        LayerNorm parameters require gradients and change on every training step.
        """
        if features.ndim != 2 or features.shape[1] != self.hidden_size:
            raise ValueError("raw feature shape differs from scorer hidden size")
        return self.head(self.norm(features)).squeeze(-1).float()

    def features(self, prompts):
        """Pooled backbone states, without evaluating or caching the trainable head."""
        if not prompts or any(not isinstance(p, str) or not p for p in prompts):
            raise ValueError("nonempty prompt strings required")
        tokens = self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
        if tokens["input_ids"].shape[1] > self.max_length:
            raise ValueError("prompt exceeds max_length; refusing to silently truncate evidence/candidates")
        device = self.head.weight.device
        ids = tokens["input_ids"].to(device)
        mask = tokens["attention_mask"].to(device)
        positions = (mask.long().cumsum(-1)-1).clamp_min(0)
        # Candidate scoring consumes the complete prompt once, never decodes.
        # Causal HF backbones otherwise return unused per-layer KV caches.
        cache_options = {"use_cache": False} if hasattr(self.backbone.config, "use_cache") else {}
        output = self.backbone(input_ids=ids, attention_mask=mask, position_ids=positions,
                               return_dict=True, **cache_options)
        hidden = output.last_hidden_state
        return last_valid_pool(hidden, mask)


def load_local_nano(backbone_path, device="cpu", *, max_length=512, freeze_backbone=True,
                    feature_normalization="none"):
    if feature_normalization not in ("none", "layernorm"):
        raise ValueError("feature_normalization must be none or layernorm")
    path = Path(backbone_path)
    if not path.is_dir():
        raise ValueError("backbone_path must be an existing local Hugging Face model directory")
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("optional transformers is not installed; no dependency or weights are downloaded automatically") from exc
    source_manifest = backbone_manifest(path)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("local tokenizer requires pad_token or eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(str(path), local_files_only=True, trust_remote_code=False).to(device)
    scorer = NanoCandidateScorer(backbone, tokenizer, max_length=max_length, freeze_backbone=freeze_backbone,
                                 feature_normalization=feature_normalization)
    scorer.backbone_local_path = str(path.resolve())
    scorer.backbone_source_manifest = source_manifest
    return scorer


def semantic_prompts(payload, schema, atom_ids):
    payload = sanitize_model_input(payload)
    if payload.text is None:
        raise ValueError("this local NanoJev-style adapter is TEXT ONLY, not a VLM")
    validate_atoms(atom_ids, schema.num_atoms)
    return [[json.dumps({"input_text": payload.text, "concept": schema.concepts[j].description,
                         "candidate_value": value}, ensure_ascii=False)
             for value in schema.concepts[j].values] for j in atom_ids]


def risk_prompts(schema, observed, actions):
    schema.validate_state(observed)
    known = []
    for j, value in enumerate(observed):
        if value < 0:
            continue
        concept = schema.concepts[j]
        name = concept.values[value] if value < len(concept.values) else (
            "UNCERTAIN" if value == len(concept.values) else "NOT_APPLICABLE")
        known.append({"concept": concept.description, "value": name})
    acquired = schema.group_mask(observed)
    prompts = []
    for action in actions:
        atoms = schema.expand(action)
        if any(acquired[g] for g in action):
            raise ValueError("candidate attempts to buy acquired group")
        candidate = [schema.concepts[j].description for j in atoms] if action else "STOP"
        prompts.append(json.dumps({"observed_evidence": known, "candidate": candidate,
            "event": "The frozen classifier will be wrong after this acquisition and then stopping."}, ensure_ascii=False))
    return prompts


@dataclass(frozen=True)
class RiskTrainingExample:
    observed: tuple[int, ...]
    action: tuple[int, ...]
    error: float
    split: str = "policy_fit"


class NanoRiskController:
    objective = "risk"

    def __init__(self, scorer, schema):
        self.scorer, self.schema = scorer, schema

    def score(self, observed, actions):
        self.scorer.eval()
        with torch.inference_mode():
            values = self.scorer.score_prompts(risk_prompts(self.schema, observed, actions)).sigmoid()
        return tuple(float(value) for value in values)

    predict = score


class NanoSemanticSession:
    def __init__(self, responder, payload):
        self.responder, self.payload = responder, sanitize_model_input(payload)
        self.last_stats = {}

    def respond(self, atom_ids):
        groups = semantic_prompts(self.payload, self.responder.schema, atom_ids)
        if not groups:
            self.last_stats = {"encoder_forwards": 0, "atoms_computed": 0, "candidate_paths": 0, "cache_hit": False}
            return ()
        model = self.responder.scorer
        model.eval()
        with torch.inference_mode():
            logits = model.score_prompts([p for group in groups for p in group])
        values, offset = [], 0
        for prompts in groups:
            probs = logits[offset:offset+len(prompts)].softmax(0)
            threshold = self.responder.threshold
            values.append(len(prompts) if threshold is not None and float(probs.max()) < threshold else int(probs.argmax()))
            offset += len(prompts)
        self.last_stats = {"encoder_forwards": 1, "atoms_computed": len(atom_ids), "candidate_paths": offset,
                           "cache_hit": False, "cost_mode": "TYPED_CANDIDATE_PATHS_NO_PREFIX_SHARING"}
        return tuple(values)


class NanoSemanticResponder:
    def __init__(self, scorer, schema, threshold=None):
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError("invalid uncertainty threshold")
        self.scorer, self.schema, self.threshold = scorer, schema, threshold

    def start_session(self, payload):
        return NanoSemanticSession(self, payload)

    def respond(self, payload: ModelInput, atom_ids: tuple[int, ...]) -> tuple[int, ...]:
        return self.start_session(payload).respond(atom_ids)


def _optimizer(scorer, learning_rate):
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    return torch.optim.Adam([p for p in scorer.parameters() if p.requires_grad], lr=learning_rate)


def _frozen_risk_features(scorer, examples, schema, batch_size, max_cache_bytes):
    if not isinstance(scorer, NanoCandidateScorer) or not scorer.freeze_backbone:
        raise ValueError("feature caching requires a frozen NanoCandidateScorer backbone")
    if any(p.requires_grad for p in scorer.backbone.parameters()):
        raise ValueError("feature caching requires every backbone parameter frozen")
    if type(batch_size) is not int or batch_size < 1 or type(max_cache_bytes) is not int or max_cache_bytes < 1:
        raise ValueError("feature batch size and max_cache_bytes must be positive integers")
    scorer.train()
    if any(module.training for module in scorer.backbone.modules()):
        raise ValueError("feature caching requires the entire backbone in eval mode")
    # Preserve first-occurrence order; duplicate events still train separately.
    prompts, prompt_index, event_indices = [], {}, []
    row_bytes = scorer.hidden_size * scorer.head.weight.element_size()
    for example in examples:
        prompt = risk_prompts(schema, example.observed, [example.action])[0]
        if prompt not in prompt_index:
            if (len(prompts) + 1) * row_bytes > max_cache_bytes:
                raise ValueError("unique frozen features exceed max_cache_bytes")
            prompt_index[prompt] = len(prompts)
            prompts.append(prompt)
        event_indices.append(prompt_index[prompt])
    # Allocate once: no full-sized concatenation copy and no autograd graph.
    features = torch.empty((len(prompts), scorer.hidden_size), dtype=scorer.head.weight.dtype,
                           device="cpu", requires_grad=False)
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            encoded = scorer.features(prompts[start:start + batch_size])
            if encoded.shape != features[start:start + batch_size].shape or encoded.dtype != features.dtype:
                raise ValueError("backbone feature shape/dtype differs from bounded cache allocation")
            features[start:start + batch_size].copy_(encoded.detach())
    return features, torch.tensor(event_indices, dtype=torch.long)


def fit_nano_risk(scorer, examples, schema, *, epochs=5, batch_size=16, learning_rate=.001, seed=0,
                  cache_features=False, feature_batch_size=None, max_cache_bytes=256 * 1024 * 1024):
    if not examples or epochs < 1 or batch_size < 1:
        raise ValueError("nonempty examples and positive training hyperparameters required")
    for e in examples:
        if not isinstance(e, RiskTrainingExample) or e.split != "policy_fit" or not 0 <= e.error <= 1:
            raise ValueError("risk training requires policy_fit examples with bounded independent event targets")
        risk_prompts(schema, e.observed, [e.action])  # Validate, but do not retain O(N * prompt_length) text.
    generator = torch.Generator().manual_seed(seed)
    optimizer = _optimizer(scorer, learning_rate)
    cached, event_indices = None, None
    if cache_features:
        cached, event_indices = _frozen_risk_features(
            scorer, examples, schema, batch_size if feature_batch_size is None else feature_batch_size,
            max_cache_bytes)
    history, weighted_history = [], []
    for _ in range(epochs):
        scorer.train()
        total, weighted_total, steps = 0., 0., 0
        for ids in torch.randperm(len(examples), generator=generator).split(batch_size):
            batch = [examples[int(i)] for i in ids]
            if cached is None:
                prompts = [risk_prompts(schema, e.observed, [e.action])[0] for e in batch]
                logits = scorer.score_prompts(prompts)
            else:
                features = cached.index_select(0, event_indices[ids]).to(scorer.head.weight.device)
                logits = scorer.score_features(features)
            target = torch.tensor([e.error for e in batch], device=logits.device)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            weighted_total += float(loss.detach()) * len(batch)
            steps += 1
        history.append(total / steps)
        weighted_history.append(weighted_total / len(examples))
    scorer.eval()
    return scorer, {"split": "policy_fit", "n": len(examples), "seed": seed,
                    "loss": "independent_event_BCE", "training_loss": history, "aggregation": "mean event BCE",
                    "feature_normalization": getattr(scorer, "feature_normalization", "unspecified_custom_scorer"),
                    "training_loss_aggregation": "unweighted mean of minibatch mean BCE (legacy)",
                    "sample_weighted_training_loss": weighted_history,
                    "examples_per_epoch": len(examples), "total_event_exposures": epochs * len(examples),
                    "optimizer_steps": epochs * ((len(examples) + batch_size - 1) // batch_size),
                    "feature_cache": {"enabled": bool(cache_features),
                        "unique_prompts": len(cached) if cached is not None else None,
                        "storage_bytes": cached.numel() * cached.element_size() if cached is not None else 0,
                        "max_cache_bytes": max_cache_bytes if cache_features else None,
                        "encoding_batch_size": (batch_size if feature_batch_size is None else feature_batch_size) if cache_features else None,
                        "dtype": str(cached.dtype) if cached is not None else None,
                        "bound_scope": "persistent pooled-feature tensor; excludes prompt strings, event indices, and batch activations",
                        "scope": "policy_fit RAW pooled states only; never normalization outputs or logits",
                        "precision_limitation": "Different encoding batch shapes may cause floating-point roundoff; assumes deterministic eval backbone",
                        "formal_speed_claim": False},
                    "freeze_backbone": scorer.freeze_backbone, "calibrated": False,
                    "head_initialization": "random_unless_explicit_checkpoint_loaded",
                    "baseline_comparator": None, "paper_evidence": False}


def fit_nano_semantics(scorer, examples, schema, *, epochs=5, learning_rate=.001, seed=0):
    validate_examples(examples, schema)
    if epochs < 1:
        raise ValueError("positive epochs required")
    optimizer = _optimizer(scorer, learning_rate)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(epochs):
        scorer.train()
        total, updates = 0., 0
        for index in torch.randperm(len(examples), generator=generator):
            example = examples[int(index)]
            atoms = tuple(j for j, target in enumerate(example.concepts) if target is not None)
            if not atoms:
                continue
            groups = semantic_prompts(example.payload, schema, atoms)
            logits = scorer.score_prompts([p for group in groups for p in group])
            terms, offset = [], 0
            for j, group in zip(atoms, groups):
                target = torch.tensor([example.concepts[j]], device=logits.device)
                terms.append(F.cross_entropy(logits[offset:offset+len(group)][None], target))
                offset += len(group)
            loss = torch.stack(terms).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            updates += 1
        history.append(total / max(updates, 1))
    scorer.eval()
    return scorer, {"split": "responder_fit", "n": len(examples), "seed": seed,
                    "loss": "masked_semantic_candidate_CE", "training_loss": history,
                    "task_label_gradient": False, "aggregation": "mean CE across observed concepts per example",
                    "baseline_comparator": None, "paper_evidence": False}


def save_nano_head(scorer, path, schema, task="risk"):
    if not scorer.freeze_backbone:
        raise ValueError("head-only export requires frozen backbone; do not discard trained backbone weights")
    if task not in ("risk", "semantic"):
        raise ValueError("task must be risk or semantic")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = backbone_manifest(scorer.backbone_local_path) if scorer.backbone_local_path else {
        "mode": "INJECTED_TEST_NO_LOCAL"}
    if scorer.backbone_source_manifest is not None and manifest != scorer.backbone_source_manifest:
        raise ValueError("local backbone/tokenizer content changed after model loading")
    torch.save({"format": "cbmjev-nano-head-v1", "head": scorer.head.state_dict(), "task": task,
                "feature_normalization": scorer.feature_normalization, "norm": scorer.norm.state_dict(),
                "schema_digest": schema_digest(schema), "hidden_size": scorer.hidden_size,
                "max_length": scorer.max_length, "backbone_local_path": scorer.backbone_local_path,
                "backbone_manifest": manifest,
                "freeze_backbone": scorer.freeze_backbone, "calibrated": False,
                "upstream_reference_commit": UPSTREAM_COMMIT,
                "implementation": "candidate_scalar_adaptation_not_full_upstream"}, path)


def load_nano_head(path, schema, device="cpu", *, scorer=None, expected_task=None):
    data = torch.load(Path(path), map_location="cpu", weights_only=True)
    if data.get("format") != "cbmjev-nano-head-v1" or data["schema_digest"] != schema_digest(schema):
        raise ValueError("Nano head format/schema mismatch")
    if expected_task is not None and data["task"] != expected_task:
        raise ValueError("semantic head and independent risk head are not interchangeable")
    normalization = data.get("feature_normalization", "none")
    if normalization not in ("none", "layernorm") or not data.get("freeze_backbone", False):
        raise ValueError("invalid normalization or nonfrozen head-only export")
    if normalization == "layernorm" and "norm" not in data:
        raise ValueError("LayerNorm export is missing normalization weights")
    if data["backbone_local_path"] is not None:
        if backbone_manifest(data["backbone_local_path"]) != data.get("backbone_manifest"):
            raise ValueError("local backbone/tokenizer content changed since head export")
        if scorer is not None:
            raise ValueError("injected scorer is only permitted for INJECTED_TEST_NO_LOCAL exports")
    elif data.get("backbone_manifest", {}).get("mode") != "INJECTED_TEST_NO_LOCAL":
        raise ValueError("missing backbone provenance manifest")
    if scorer is None:
        if data["backbone_local_path"] is None:
            raise ValueError("injected test backbone export requires explicitly injected scorer to reload")
        scorer = load_local_nano(data["backbone_local_path"], device, max_length=data["max_length"],
                                 feature_normalization=normalization)
    if scorer.hidden_size != data["hidden_size"]:
        raise ValueError("backbone hidden size mismatch")
    if scorer.feature_normalization != normalization:
        raise ValueError("injected scorer normalization mismatch")
    if not scorer.freeze_backbone or any(p.requires_grad for p in scorer.backbone.parameters()):
        raise ValueError("head-only reload requires a frozen backbone")
    scorer.norm.load_state_dict(data.get("norm", {}), strict=True)
    scorer.head.load_state_dict(data["head"])
    return scorer.to(device).eval()

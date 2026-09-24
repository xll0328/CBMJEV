"""Semantic-only trainable responders; no task-label gradient or metadata input."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from .contracts import ModelInput
from .input_prefetch import ordered_input_batches, validate_input_prefetch


def sanitize_model_input(value: Mapping | ModelInput) -> ModelInput:
    if isinstance(value, ModelInput):
        if type(value) is not ModelInput or set(vars(value)) != {"text", "images"}:
            raise ValueError("ModelInput must contain content only, without added metadata")
        if not isinstance(value.images, tuple):
            raise TypeError("images must be an immutable tuple of bytes")
        return value
    if not isinstance(value, Mapping):
        raise TypeError("expected ModelInput or strict input mapping")
    if set(value) - {"text", "images"}:
        raise ValueError("metadata/path/gold fields forbidden in responder input")
    return sanitize_model_input(ModelInput(**value))


@dataclass(frozen=True)
class ConceptTrainingExample:
    payload: ModelInput
    concepts: tuple[int | None, ...]
    split: str = "responder_fit"


def schema_digest(schema):
    return schema.hash


def _counts(schema):
    return tuple(len(concept.values) for concept in schema.concepts)


def validate_atoms(atom_ids, width):
    if not isinstance(atom_ids, tuple) or any(type(j) is not int or not 0 <= j < width for j in atom_ids):
        raise ValueError("atom_ids must be a tuple of valid integer indices")
    if len(set(atom_ids)) != len(atom_ids):
        raise ValueError("duplicate requested atom")


def validate_examples(examples, schema, *, return_class_counts=False):
    counts = _counts(schema)
    if not examples:
        raise ValueError("empty responder_fit examples")
    observed = [0] * len(counts)
    class_counts = [[0] * count for count in counts]
    for example in examples:
        if not isinstance(example, ConceptTrainingExample) or example.split != "responder_fit":
            raise ValueError("semantic training requires ConceptTrainingExample on responder_fit only")
        sanitize_model_input(example.payload)
        if len(example.concepts) != len(counts):
            raise ValueError("concept target width mismatch")
        for j, (value, count) in enumerate(zip(example.concepts, counts)):
            if value is not None and (type(value) is not int or not 0 <= value < count):
                raise ValueError("gold target must be semantic category or None; runtime states are not gold")
            observed[j] += value is not None
            if value is not None:
                class_counts[j][value] += 1
    if not sum(observed):
        raise ValueError("no observed semantic labels")
    return (observed, class_counts) if return_class_counts else observed


def _bulk_semantic_answers(distributions, value_counts, threshold):
    """Extract discrete answers with at most two device-to-host transfers.

    Keep per-head argmax/max and Python-float threshold comparison identical to
    the scalar path. Comparing threshold on-device could round a Python float
    to the probability dtype and change decisions near the threshold boundary.
    """
    if not distributions:
        return ()
    answers = torch.stack([p.argmax() for p in distributions]).cpu().tolist()
    if threshold is None:
        return tuple(answers)
    confidences = torch.stack([p.max() for p in distributions]).cpu().tolist()
    return tuple(count if confidence < threshold else answer
                 for answer, confidence, count in zip(answers, confidences, value_counts))


class CheapSession:
    """Episode-local cache; first query computes ALL heads and reports that fact."""
    def __init__(self, model, payload):
        self.model = model
        self.payload = sanitize_model_input(payload)
        self._values = None
        self.last_stats = {}
        self.cost_mode = getattr(model, "shared_cost_mode", "SHARED_CHEAP_ALL_HEADS")

    def respond(self, atom_ids: tuple[int, ...]) -> tuple[int, ...]:
        validate_atoms(atom_ids, len(self.model.value_counts))
        if not atom_ids:
            self.last_stats = {"encoder_forwards": 0, "atoms_computed": 0, "cache_hit": False,
                               "cost_mode": self.cost_mode}
            return ()
        computed = self._values is None
        if computed:
            self.model.eval()
            with torch.inference_mode():
                distributions = [x.softmax(-1)[0] for x in self.model([self.payload])]
                self._values = _bulk_semantic_answers(distributions, self.model.value_counts,
                                                     self.model.threshold)
        self.last_stats = {"encoder_forwards": int(computed),
                           "atoms_computed": len(self.model.value_counts) if computed else 0,
                           "images_encoded": len(self.payload.images) if computed else 0,
                           "cache_hit": not computed, "cost_mode": self.cost_mode}
        return tuple(self._values[j] for j in atom_ids)


class SharedResponder(nn.Module):
    def __init__(self, schema, threshold=None):
        super().__init__()
        if threshold is not None and (not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1):
            raise ValueError("threshold must lie in [0,1]")
        self.schema, self.threshold = schema, threshold
        self.value_counts = _counts(schema)

    def forward(self, payloads: Sequence[ModelInput]):
        features = self.encode([sanitize_model_input(p) for p in payloads])
        return tuple(head(features) for head in self.heads)

    def start_session(self, payload):
        return CheapSession(self, payload)

    def respond(self, payload: ModelInput, atom_ids: tuple[int, ...]) -> tuple[int, ...]:
        # Convenience path is a NEW episode. Live sequential execution must retain a session.
        return self.start_session(payload).respond(atom_ids)


class HashingTextResponder(SharedResponder):
    def __init__(self, schema, vocab_size=2048, embedding_dim=64, threshold=None):
        super().__init__(schema, threshold)
        if type(vocab_size) is not int or vocab_size < 2 or type(embedding_dim) is not int or embedding_dim < 1:
            raise ValueError("invalid text encoder dimensions")
        self.vocab_size, self.embedding_dim = vocab_size, embedding_dim
        self.encoder = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")
        self.heads = nn.ModuleList(nn.Linear(embedding_dim, count) for count in self.value_counts)

    def token_ids(self, text):
        tokens = re.findall(r"[\u4e00-\u9fff]|[^\W_]+", text.casefold())
        return [1 + int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little") % (self.vocab_size-1)
                for t in tokens] or [0]

    def encode(self, payloads):
        if any(p.text is None for p in payloads):
            raise ValueError("HashingTextResponder accepts text only")
        tokens, offsets = [], []
        for payload in payloads:
            offsets.append(len(tokens))
            tokens.extend(self.token_ids(payload.text))
        device = self.encoder.weight.device
        return self.encoder(torch.tensor(tokens, dtype=torch.long, device=device),
                            torch.tensor(offsets, dtype=torch.long, device=device))

    def config(self):
        return {"kind": "hashing_text", "vocab_size": self.vocab_size,
                "embedding_dim": self.embedding_dim, "threshold": self.threshold}


class HFTextResponder(SharedResponder):
    """Shared pretrained encoder and masked-mean concept heads, never task labels.

    Injection supports offline contract tests; deployment uses load_hf_text_responder.
    The first query still computes all heads, as with the cheap shared responder.
    """
    shared_cost_mode = "SHARED_HF_TEXT_ALL_HEADS"

    def __init__(self, schema, encoder, tokenizer, *, max_length=512,
                 freeze_backbone=False, threshold=None, initialization=None):
        super().__init__(schema, threshold)
        width = getattr(encoder.config, "hidden_size", None)
        if type(width) is not int or width < 1 or type(max_length) is not int or max_length < 1:
            raise ValueError("HF hidden_size and max_length must be positive integers")
        if type(freeze_backbone) is not bool:
            raise ValueError("freeze_backbone must be boolean")
        if getattr(encoder.config, "is_decoder", False) or getattr(encoder.config, "is_encoder_decoder", False):
            raise ValueError("hf_text requires an encoder-only model")
        if getattr(tokenizer, "pad_token_id", None) is None or tokenizer.padding_side != "right":
            raise ValueError("hf_text requires a pad token and right padding")
        capacity = getattr(encoder.config, "max_position_embeddings", None)
        if type(capacity) is int and max_length > capacity:
            raise ValueError("max_length exceeds encoder positional capacity")
        self.encoder, self.tokenizer = encoder, tokenizer
        self.max_length, self.freeze_backbone = max_length, freeze_backbone
        self.initialization = dict(initialization or {"kind": "INJECTED_TEST_NO_PRETRAINED_SOURCE"})
        self.heads = nn.ModuleList(nn.Linear(width, count) for count in self.value_counts)
        parameter = next(encoder.parameters(), None)
        if parameter is not None:
            self.heads.to(device=parameter.device, dtype=parameter.dtype)
        if freeze_backbone:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def encode(self, payloads):
        if not payloads or any(p.text is None for p in payloads):
            raise ValueError("HFTextResponder accepts nonempty batches of text only")
        if self.tokenizer.padding_side != "right":
            raise ValueError("hf_text requires right padding")
        tokens = self.tokenizer([p.text for p in payloads], padding=True, truncation=False,
                                return_attention_mask=True, return_tensors="pt")
        if set(tokens) - {"input_ids", "attention_mask", "token_type_ids"}:
            raise ValueError("unexpected HF tokenizer model input fields")
        if "input_ids" not in tokens or "attention_mask" not in tokens:
            raise ValueError("HF tokenizer must return input_ids and attention_mask")
        ids, mask = tokens["input_ids"], tokens["attention_mask"]
        if ids.ndim != 2 or mask.shape != ids.shape or ids.shape[0] != len(payloads):
            raise ValueError("invalid HF tokenizer input/mask shape")
        if ids.shape[1] > self.max_length:
            raise ValueError("text exceeds max_length; refusing to silently truncate semantic evidence")
        if not torch.all((mask == 0) | (mask == 1)) or not mask.bool().any(1).all():
            raise ValueError("invalid or all-padding HF attention mask")
        device = self.heads[0].weight.device
        tokens = {key: value.to(device) for key, value in tokens.items()}
        hidden = self.encoder(**tokens, return_dict=True).last_hidden_state
        if hidden.ndim != 3 or tuple(hidden.shape[:2]) != tuple(ids.shape):
            raise ValueError("invalid HF encoder hidden-state shape")
        weights = tokens["attention_mask"].to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(1) / weights.sum(1)

    def config(self):
        return {"kind": "hf_text", "max_length": self.max_length,
                "freeze_backbone": self.freeze_backbone, "threshold": self.threshold,
                "pooling": "attention_masked_mean", "truncation": False,
                "attention_implementation": "eager", "initialization": self.initialization}


def _encoder_fingerprint(encoder):
    digest = hashlib.sha256()
    for name, value in sorted(encoder.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _hf_assets_manifest(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("HF checkpoint assets are missing or changed")
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("HF checkpoint assets must not contain symlinks")
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files[path.relative_to(directory).as_posix()] = digest.hexdigest()
    if "config.json" not in files or len(files) < 2:
        raise ValueError("HF checkpoint config/tokenizer assets are missing or changed")
    return files


def load_hf_text_responder(schema, model_name_or_path, *, revision=None, allow_download=False,
                           max_length=512, freeze_backbone=False, threshold=None):
    """Explicit local/cached/opt-in Hub initialization; no remote model code."""
    if not isinstance(model_name_or_path, (str, Path)) or not str(model_name_or_path).strip():
        raise ValueError("hf_text requires an explicit model_name_or_path")
    if type(allow_download) is not bool:
        raise ValueError("allow_download must be boolean")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("revision must be a nonempty string or None")
    source = str(model_name_or_path)
    local = Path(source).expanduser()
    is_local = local.is_dir()
    if not is_local and (local.is_absolute() or source.startswith((".", "~")) or local.exists()):
        raise ValueError("explicit local HF model directory does not exist or is not a directory")
    if is_local:
        source = str(local.resolve())
    try:
        import transformers
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("hf_text requires optional transformers; no toy fallback or automatic installation") from exc
    encoder = AutoModel.from_pretrained(source, revision=revision, local_files_only=not allow_download,
                                        trust_remote_code=False, attn_implementation="eager", weights_only=True)
    resolved = getattr(encoder.config, "_commit_hash", None)
    if not is_local and (not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-f]{40}", resolved)):
        if isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision):
            resolved = revision
        else:
            raise ValueError("Hub encoder resolved commit unavailable; supply an immutable 40-character revision")
    # A mutable Hub tag/main may move between calls. Pin the tokenizer to the
    # actual encoder commit instead of resolving the requested tag a second time.
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision if is_local else resolved,
                    local_files_only=not allow_download, trust_remote_code=False)
    tokenizer.padding_side = "right"
    initialization = {"kind": "PRETRAINED_ENCODER", "source_kind": "LOCAL_DIRECTORY" if is_local else "HUGGING_FACE_HUB",
        "model_name_or_path": source, "requested_revision": revision, "resolved_revision": resolved,
        "encoder_initial_state_sha256": _encoder_fingerprint(encoder),
        "tokenizer_class": type(tokenizer).__name__, "allow_download": allow_download,
        "transformers_version": str(getattr(transformers, "__version__", "UNKNOWN")),
        "trust_remote_code": False, "attention_implementation": "eager"}
    return HFTextResponder(schema, encoder, tokenizer, max_length=max_length,
                           freeze_backbone=freeze_backbone, threshold=threshold, initialization=initialization)


class SharedVisionResponder(SharedResponder):
    """Optional ResNet18. weights=None always; optional initialization is local only."""
    def __init__(self, schema, image_size=224, backbone_checkpoint=None, threshold=None):
        super().__init__(schema, threshold)
        if type(image_size) is not int or image_size < 32:
            raise ValueError("image_size must be an integer >=32")
        try:
            from torchvision import models, transforms
        except ImportError as exc:
            raise ImportError("torchvision is optional and must be installed by the environment owner") from exc
        self.image_size = image_size
        self.encoder = models.resnet18(weights=None)
        if backbone_checkpoint is not None:
            path = Path(backbone_checkpoint)
            if not path.is_file():
                raise ValueError("backbone_checkpoint must be an existing local state_dict")
            self.encoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        width = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.heads = nn.ModuleList(nn.Linear(width, count) for count in self.value_counts)
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225))])

    def encode(self, payloads):
        from PIL import Image
        images, ranges = [], []
        for p in payloads:
            if not p.images:
                raise ValueError("SharedVisionResponder accepts image bytes only")
            start = len(images)
            for image_bytes in p.images:
                with Image.open(io.BytesIO(image_bytes)) as image:
                    images.append(self.transform(image.convert("RGB")))
            ranges.append((start, len(images)))
        encoded = self.encoder(torch.stack(images).to(next(self.encoder.parameters()).device))
        return torch.stack([encoded[a:b].mean(0) for a, b in ranges])

    def config(self):
        return {"kind": "resnet18", "image_size": self.image_size, "threshold": self.threshold}


def _responder_training_device(device):
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("supported responder training devices are cpu and cuda")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _seed_responder_rng(seed, device):
    # torch.manual_seed seeds ALL accelerator RNGs, including unrelated GPUs.
    # CPU initialization/shuffle and selected-device dropout need only these RNGs.
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def fit_responder(model, examples, *, epochs=30, batch_size=32, learning_rate=.003, seed=0,
                  device="cpu", deterministic=True, input_workers=0, input_prefetch_batches=2,
                  concept_class_weighting="none"):
    validate_input_prefetch(input_workers, input_prefetch_batches)
    if type(deterministic) is not bool:
        raise ValueError("deterministic must be boolean")
    if concept_class_weighting not in ("none", "inverse_frequency"):
        raise ValueError("concept_class_weighting must be none or inverse_frequency")
    observed_per_atom, class_counts = validate_examples(examples, model.schema, return_class_counts=True)
    class_weights = []
    for counts in class_counts:
        supported = sum(count > 0 for count in counts)
        total = sum(counts)
        class_weights.append([1.0 if concept_class_weighting == "none" else
                              (total / (supported * count) if count else 0.0)
                              for count in counts])
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("positive training hyperparameters required")
    # Seed training-time randomness as well as the independent shuffle generator.
    # Initial weights remain the caller's responsibility; fit_text_responder seeds them.
    device = _responder_training_device(device)
    _seed_responder_rng(seed, device)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    generator = torch.Generator().manual_seed(seed)
    model.to(device)
    # Unit expected weight under each concept's observed TRAINING distribution.
    # Do not renormalize within minibatches: that erases balancing for pure-class
    # batches (especially batch_size=1). Missing targets never enter the loss.
    weight_tensors = [torch.tensor(weights, device=device, dtype=next(model.parameters()).dtype)
                      for weights in class_weights] if concept_class_weighting != "none" else None
    if weight_tensors is not None and any(not torch.isfinite(w).all() for w in weight_tensors):
        raise ValueError("nonfinite responder class weights")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for _ in range(epochs):
        model.train()
        total, updates = 0., 0
        with ordered_input_batches(examples, torch.randperm(len(examples), generator=generator).split(batch_size),
                                   input_workers=input_workers,
                                   input_prefetch_batches=input_prefetch_batches) as batches:
            for batch in batches:
                logits = model([e.payload for e in batch])
                terms = []
                for j, pred in enumerate(logits):
                    valid = [i for i, e in enumerate(batch) if e.concepts[j] is not None]
                    if valid:
                        target = torch.tensor([batch[i].concepts[j] for i in valid], device=device)
                        if weight_tensors is None:
                            terms.append(F.cross_entropy(pred[valid], target))
                        else:
                            per_label = F.cross_entropy(pred[valid], target, reduction="none")
                            terms.append((per_label * weight_tensors[j][target]).mean())
                if not terms:
                    continue
                loss = torch.stack(terms).mean()  # Equal concept weight within each labelled batch.
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite semantic responder training loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                updates += 1
        losses.append(total / max(updates, 1))
    model.eval()
    return model, {"split": "responder_fit", "n": len(examples), "seed": seed,
                   "loss": ("masked_concept_mean_CE" if concept_class_weighting == "none" else
                            "masked_concept_mean_train_inverse_frequency_CE"),
                   "deterministic": torch.are_deterministic_algorithms_enabled(),
                   "device": str(next(model.parameters()).device), "torch_version": str(torch.__version__),
                   "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                   "input_workers": input_workers, "input_prefetch_batches": input_prefetch_batches,
                   "observed_labels_per_atom": observed_per_atom, "training_loss": losses,
                   "concept_class_weighting": concept_class_weighting,
                   "concept_class_counts": class_counts, "concept_class_weights": class_weights,
                   "concept_class_weight_fit_scope": "supplied_responder_fit_examples_only",
                   "concept_class_weight_normalization": ("unweighted" if concept_class_weighting == "none" else
                       "n_observed/(n_supported_classes*n_class); unsupported=0; mean over valid labels, not weight sum"),
                   "task_label_gradient": False, "aggregation": "mean over labelled concepts per batch",
                   "baseline_comparator": None, "paper_evidence": False}


def fit_text_responder(examples, schema, *, vocab_size=2048, embedding_dim=64, threshold=None, seed=0, **kwargs):
    device = _responder_training_device(kwargs.get("device", "cpu"))
    # Explicit devices=[] avoids fork_rng's default enumeration/initialization
    # of every visible CUDA GPU for a CPU-only training request.
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        torch.random.default_generator.manual_seed(seed)
        model = HashingTextResponder(schema, vocab_size, embedding_dim, threshold)
        return fit_responder(model, examples, seed=seed, **{**kwargs, "device": device})


def save_responder(model, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = model.config()
    if isinstance(model, HFTextResponder):
        assets = path.parent / (path.stem + "_hf_assets")
        if assets.exists():
            raise FileExistsError(assets)
        assets.mkdir()
        model.encoder.config.save_pretrained(assets)
        model.tokenizer.save_pretrained(assets)
        # The manifest is INSIDE responder.pt, so the existing R checkpoint hash
        # transitively binds every config/tokenizer byte used on offline reload.
        config["assets_dir"] = assets.name
        config["assets_manifest"] = _hf_assets_manifest(assets)
    torch.save({"format": "cbmjev-semantic-responder-v1", "config": config,
                "schema_digest": schema_digest(model.schema), "state_dict": model.state_dict()}, path)


def load_responder(path, schema, device="cpu"):
    path = Path(path)
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format") != "cbmjev-semantic-responder-v1" or data["schema_digest"] != schema_digest(schema):
        raise ValueError("checkpoint format/schema mismatch")
    config = dict(data["config"])
    kind = config.pop("kind")
    if kind not in ("hashing_text", "resnet18", "hf_text"):
        raise ValueError("unknown responder kind")
    if kind == "hf_text":
        assets_name = config.pop("assets_dir")
        if not isinstance(assets_name, str) or Path(assets_name).name != assets_name or assets_name in ("", ".", ".."):
            raise ValueError("invalid relative HF checkpoint assets directory")
        assets = path.parent / assets_name
        if assets.is_symlink() or _hf_assets_manifest(assets) != config.pop("assets_manifest"):
            raise ValueError("HF checkpoint assets changed since export")
        if (config.pop("pooling") != "attention_masked_mean" or config.pop("truncation") is not False
                or config.pop("attention_implementation") != "eager"):
            raise ValueError("unsupported HF checkpoint preprocessing configuration")
        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("hf_text reload requires optional transformers; no fallback") from exc
        encoder_config = AutoConfig.from_pretrained(str(assets), local_files_only=True, trust_remote_code=False)
        encoder = AutoModel.from_config(encoder_config, trust_remote_code=False, attn_implementation="eager")
        tokenizer = AutoTokenizer.from_pretrained(str(assets), local_files_only=True, trust_remote_code=False)
        model = HFTextResponder(schema, encoder, tokenizer, **config)
    else:
        cls = HashingTextResponder if kind == "hashing_text" else SharedVisionResponder
        model = cls(schema, **config)
    model.load_state_dict(data["state_dict"])
    return model.to(device).eval()

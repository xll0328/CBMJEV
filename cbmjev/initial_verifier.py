"""Source-only initial concepts for the new appraisal domain.

A single fine-tuned DistilRoBERTa encoding feeds native-category concept heads.
The pooling rule is last-valid-token, distinct from historical A's masked mean.
Task labels are absent from every model and training interface. Runtime files
contain private IDs/labels and belong on the data server, outside public results.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re
import time

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import Schema, stable_hash
from .io import file_hash, fresh_dir, read_json, write_json, write_jsonl
from .nanojev import last_valid_pool
from .responders import _seed_responder_rng
from .typed_verifier import (concept_metrics, load_source_records, record_metadata,
                             source_group_split)


PROTOCOL = "cbmjev-initial-native-concepts-v1"
MODEL_REVISION = "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b"
LEARNING_RATES = (2e-5, 5e-5)
TRAINING_SEEDS = (40, 41, 42)
MAX_LENGTH = 512


class InitialConceptEncoder(nn.Module):
    """Shared last-valid hidden state; one head per native concept vocabulary."""

    def __init__(self, schema, backbone, tokenizer, *, binding, max_length=MAX_LENGTH):
        super().__init__()
        width = getattr(backbone.config, "hidden_size", None)
        if type(width) is not int or width < 1 or type(max_length) is not int or max_length < 1:
            raise ValueError("hidden_size and max_length must be positive integers")
        if getattr(backbone.config, "is_decoder", False) or getattr(backbone.config, "is_encoder_decoder", False):
            raise ValueError("initial concepts require an encoder-only backbone")
        if getattr(tokenizer, "pad_token_id", None) is None:
            raise ValueError("initial tokenizer requires a pad token")
        if tokenizer.padding_side != "right" or tokenizer.truncation_side != "right":
            raise ValueError("initial source fixes padding and truncation to right")
        self.schema, self.backbone, self.tokenizer = schema, backbone, tokenizer
        self.max_length, self.value_counts = max_length, schema.value_counts
        self.backbone.float()
        self.heads = nn.ModuleList(nn.Linear(width, count) for count in self.value_counts)
        self.heads.to(device=next(backbone.parameters()).device, dtype=torch.float32)
        self.binding = {**binding, "schema_hash": schema.hash, "hidden_size": width,
                        "max_length": max_length, "pooling": "last_valid_token",
                        "dtype": "torch.float32", "padding_side": "right",
                        "truncation_side": "right", "backbone_frozen": False}

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def forward(self, texts):
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("initial source accepts nonempty text batches only")
        if self.tokenizer.padding_side != "right" or self.tokenizer.truncation_side != "right":
            raise ValueError("initial tokenizer condition changed")
        tokens = self.tokenizer(list(texts), padding=True, truncation=True,
                                max_length=self.max_length, return_attention_mask=True,
                                return_tensors="pt")
        if set(tokens) - {"input_ids", "attention_mask", "token_type_ids"}:
            raise ValueError("unexpected tokenizer model input fields")
        if "input_ids" not in tokens or "attention_mask" not in tokens:
            raise ValueError("tokenizer must return input_ids and attention_mask")
        ids, mask = tokens["input_ids"], tokens["attention_mask"]
        if ids.ndim != 2 or mask.shape != ids.shape or ids.shape[0] != len(texts) or ids.shape[1] > self.max_length:
            raise ValueError("invalid or untruncated tokenized batch")
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        hidden = self.backbone(**tokens).last_hidden_state
        pooled = last_valid_pool(hidden, tokens["attention_mask"])
        return tuple(head(pooled) for head in self.heads)

    def truncation_report(self, texts, *, batch_size=128):
        """Count original lengths once; this audit does not alter model inputs."""
        if not texts or type(batch_size) is not int or batch_size < 1:
            raise ValueError("nonempty texts and positive batch_size required")
        lengths = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(list(texts[start:start + batch_size]), padding=False,
                                    truncation=False, return_attention_mask=False)
            lengths.extend(len(ids) for ids in tokens["input_ids"])
        truncated = sum(length > self.max_length for length in lengths)
        return {"n_samples": len(lengths), "max_length": self.max_length,
                "length_includes_special_tokens": True, "max_original_tokens": max(lengths),
                "truncated_samples": truncated, "truncation_rate": truncated / len(lengths),
                "removed_tokens": sum(max(0, length - self.max_length) for length in lengths),
                "rule": "tokenizer right truncation, preserving special tokens"}


def load_initial_encoder(backbone_path, schema, *, model_revision, device="cuda:0", max_length=MAX_LENGTH):
    """Load only the existing pinned DistilRoBERTa snapshot; never download."""
    path = Path(backbone_path)
    if not path.is_dir() or not re.fullmatch(r"[a-f0-9]{40}", model_revision):
        raise ValueError("existing local backbone and immutable revision required")
    if model_revision != MODEL_REVISION or max_length != MAX_LENGTH:
        raise ValueError("initial source recipe fixes the DistilRoBERTa revision and length 512")
    import transformers
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = tokenizer.truncation_side = "right"
    backbone = AutoModel.from_pretrained(str(path), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation="eager", add_pooling_layer=False).to(device)
    if (getattr(backbone.config, "model_type", None) != "roberta"
            or getattr(backbone.config, "num_hidden_layers", None) != 6
            or getattr(backbone.config, "hidden_size", None) != 768):
        raise ValueError("source recipe requires the pinned six-layer DistilRoBERTa")
    resolved = getattr(backbone.config, "_commit_hash", None)
    if resolved is not None and resolved != model_revision:
        raise ValueError("loaded model revision differs from the declared snapshot")
    metadata_files = {name: file_hash(path / name) for name in
                      ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
                      if (path / name).is_file()}
    weight_files = {item.name: item.stat().st_size for item in path.iterdir()
                    if item.is_file() and item.suffix in (".safetensors", ".bin")}
    binding = {"model_id": "distilroberta-base", "model_revision": model_revision,
               "tokenizer_revision": model_revision, "local_backbone_path": str(path.resolve()),
               "revision_verification": "hf_config_commit" if resolved else "operator_pinned_local_snapshot",
               "metadata_sha256": metadata_files, "pretrained_weight_bytes": weight_files,
               "attention_implementation": "eager", "torch_version": str(torch.__version__),
               "transformers_version": transformers.__version__}
    return InitialConceptEncoder(schema, backbone, tokenizer, binding=binding, max_length=max_length)


def masked_concept_loss(logits, labels):
    """Mean CE over observed concept annotations; -1 is missing, never unknown."""
    if labels.dtype != torch.long or labels.ndim != 2 or labels.shape[1] != len(logits) or not logits:
        raise ValueError("labels must be a batch by concept int64 tensor")
    losses, observed = [], 0
    for j, head in enumerate(logits):
        if head.ndim != 2 or head.shape[0] != labels.shape[0]:
            raise ValueError("concept logits and label shape differ")
        target = labels[:, j].to(head.device)
        if torch.any((target < -1) | (target >= head.shape[1])):
            raise ValueError("concept target is not a native category or missing")
        valid = target >= 0
        if valid.any():
            losses.append(F.cross_entropy(head[valid], target[valid], reduction="sum"))
            observed += int(valid.sum())
    if not losses:
        return sum(head.sum() * 0 for head in logits)
    return torch.stack(losses).sum() / observed


def _source_labels(records, schema):
    if not records or any(record.role != "responder_fit" for record in records):
        raise ValueError("A fitting permits responder_fit concepts only")
    for record in records:
        if len(record.concepts) != schema.num_atoms:
            raise ValueError("source concept width differs from schema")
        for value, count in zip(record.concepts, schema.value_counts):
            if value is not None and (type(value) is not int or not 0 <= value < count):
                raise ValueError("source concept must be a native category or missing")
    return torch.tensor([[value if value is not None else -1 for value in record.concepts]
                         for record in records], dtype=torch.long)


@contextmanager
def _seeded(seed, device):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        raise ValueError("choose an explicit logical CUDA device")
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        _seed_responder_rng(seed, device)
        yield


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_initial_records(model, records, *, batch_size=16):
    if not records or type(batch_size) is not int or batch_size < 1:
        raise ValueError("nonempty records and a positive batch size required")
    model.eval()
    probabilities = [[] for _ in model.value_counts]
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            logits = model([record.text for record in records[start:start + batch_size]])
            for j, head in enumerate(logits):
                if not torch.isfinite(head).all():
                    raise ValueError("nonfinite initial-source logits")
                probabilities[j].append(head.softmax(-1).cpu())
    probabilities = tuple(torch.cat(parts) for parts in probabilities)
    return torch.stack([p.argmax(1) for p in probabilities], 1), probabilities


def _fit_recipe(model_factory, records, labels, schema, fit_indices, *, learning_rate,
                epochs, batch_size, seed, device, tune_indices=None):
    with _seeded(seed, device):
        model = model_factory().to(device=device, dtype=torch.float32)
        if model.schema.hash != schema.hash:
            raise ValueError("model factory schema differs from training schema")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.)
        generator = torch.Generator().manual_seed(seed)
        indices = torch.tensor(fit_indices, dtype=torch.long)
        history, best = [], None
        for epoch in range(1, epochs + 1):
            model.train()
            order = indices[torch.randperm(len(indices), generator=generator)]
            total_loss, observed = 0., 0
            for start in range(0, len(order), batch_size):
                subset = order[start:start + batch_size]
                targets = labels[subset]
                n = int((targets >= 0).sum())
                if not n:
                    continue
                logits = model([records[i].text for i in subset.tolist()])
                loss = masked_concept_loss(logits, targets)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite source concept loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * n
                observed += n
            result = {"epoch": epoch, "observed_label_exposures": observed,
                      "training_ce": total_loss / observed}
            if tune_indices is not None:
                values, _ = predict_initial_records(model, [records[i] for i in tune_indices],
                                                    batch_size=batch_size)
                metrics = concept_metrics(values, labels[tune_indices], schema)
                result["tune_concept_macro_f1"] = metrics["concept_macro_f1"]
                if best is None or metrics["concept_macro_f1"] > best["metrics"]["concept_macro_f1"]:
                    best = {"epoch": epoch, "learning_rate": learning_rate, "metrics": metrics}
            history.append(result)
        return model.eval(), history, best


def fit_initial_source(records, schema, output, *, model_factory, seed=40, epochs=5,
                       batch_size=16, split_seed=20260930, device="cpu", provenance=None):
    """Select on shared source groups, then reinitialize and refit all source rows."""
    if seed not in TRAINING_SEEDS or split_seed != 20260930:
        raise ValueError("source recipe fixes seeds 40/41/42 and group split seed 20260930")
    if type(epochs) is not int or not 1 <= epochs <= 5 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("source recipe permits 1..5 epochs and a positive batch size")
    labels = _source_labels(records, schema)
    manifest = [record_metadata(record) for record in records]
    fit, tune, split = source_group_split(manifest, split_seed=split_seed)
    if torch.any((labels[fit] >= 0).sum(0) == 0):
        raise ValueError("each concept needs an observed source-internal-fit label")
    if not (labels[tune] >= 0).any():
        raise ValueError("source-internal-tune has no observed concept labels")
    output = fresh_dir(output)
    plan = {"protocol": PROTOCOL, "schema_hash": schema.hash, "seed": seed, "source_split": split,
            "learning_rates": list(LEARNING_RATES), "epochs_per_trial": epochs,
            "selection_metric": "concept_macro_f1", "selection_role": "source_internal_tune",
            "tie_break": "first declared learning rate, then earliest epoch", "no_task_labels": True}
    write_json(output / "training_plan.json", plan)
    write_jsonl(output / "source_manifest.jsonl", manifest)
    majority_values = []
    for j, count in enumerate(schema.value_counts):
        targets = labels[fit, j]
        majority_values.append(int(torch.bincount(targets[targets >= 0], minlength=count).argmax()))
    majority = concept_metrics(torch.tensor(majority_values).repeat(len(tune), 1), labels[tune], schema)
    trials, selected, truncation, expected_binding = [], None, None, None
    started = time.perf_counter()
    for learning_rate in LEARNING_RATES:
        model, history, best = _fit_recipe(model_factory, records, labels, schema, fit,
            learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, seed=seed,
            device=device, tune_indices=tune)
        if expected_binding is None:
            expected_binding = model.binding
            truncation = model.truncation_report([record.text for record in records])
        elif model.binding != expected_binding:
            raise ValueError("pretrained model binding changed across source trials")
        trials.append({"learning_rate": learning_rate, "history": history, "best": best})
        if selected is None or best["metrics"]["concept_macro_f1"] > selected["metrics"]["concept_macro_f1"]:
            selected = best
        del model
    model, refit_history, _ = _fit_recipe(model_factory, records, labels, schema, list(range(len(records))),
        learning_rate=selected["learning_rate"], epochs=selected["epoch"], batch_size=batch_size,
        seed=seed, device=device)
    if model.binding != expected_binding:
        raise ValueError("pretrained model binding changed for full-source refit")
    _sync(model.device)
    elapsed = time.perf_counter() - started
    metadata = {"protocol": PROTOCOL, "schema": schema.to_dict(), "binding": model.binding,
        "seed": seed, "source_split": split, "fit_sample_ids": [r.sample_id for r in records],
        "fit_group_ids": sorted({r.group_id for r in records}), "provenance": provenance or {},
        "source_manifest_sha256": stable_hash(manifest), "training_plan_sha256": stable_hash(plan),
        "supervision": "responder_fit OBSERVED concept labels only; task labels never provided",
        "objective": "masked CE, mean over observed concept labels", "weight_decay": 0.,
        "pooling_note": "last-valid-token; historical A used masked mean and is not this model",
        "selected_recipe": {"learning_rate": selected["learning_rate"], "epochs": selected["epoch"]},
        "training_batch_size": batch_size, "training_device": str(torch.device(device)),
        "parameters": sum(p.numel() for p in model.parameters()), "max_length": model.max_length,
        "truncation": truncation, "selection_and_refit_seconds": elapsed,
        "artifact_visibility": "PRIVATE_RUNTIME", "paper_evidence": False,
        "pilot_reduced_epochs": epochs < 5}
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({"protocol": PROTOCOL, "schema_hash": schema.hash,
                "binding_hash": stable_hash(model.binding), "state_dict": state}, output / "model.pt")
    metadata["checkpoint_bytes"] = (output / "model.pt").stat().st_size
    metadata["checkpoint_identity"] = stable_hash(metadata)
    report = {"trials": trials, "selected": selected, "majority_comparator": majority,
              "full_source_refit_history": refit_history, "full_source_refit_n": len(records),
              "selection_role": "source_internal_tune", "truncation": truncation}
    write_json(output / "checkpoint.json", metadata)
    write_json(output / "selection_report.json", report)
    return metadata, report


def train_initial_source(prepared, output, backbone_path, *, model_revision, seed=40,
                         epochs=5, batch_size=16, split_seed=20260930, device="cuda:0"):
    schema, records, provenance = load_source_records(prepared, ("responder_fit",))
    return fit_initial_source(records, schema, output, model_factory=lambda: load_initial_encoder(
        backbone_path, schema, model_revision=model_revision, device=device), seed=seed,
        epochs=epochs, batch_size=batch_size, split_seed=split_seed, device=device, provenance=provenance)


def load_initial_checkpoint(checkpoint, *, device="cpu", model_factory=None):
    checkpoint = Path(checkpoint)
    metadata = read_json(checkpoint / "checkpoint.json")
    if metadata.get("protocol") != PROTOCOL:
        raise ValueError("unknown initial-source checkpoint protocol")
    payload = {k: v for k, v in metadata.items() if k != "checkpoint_identity"}
    if stable_hash(payload) != metadata.get("checkpoint_identity"):
        raise ValueError("initial checkpoint metadata identity differs")
    schema = Schema.from_dict(metadata["schema"])
    if schema.hash != metadata["binding"].get("schema_hash"):
        raise ValueError("initial checkpoint schema binding differs")
    if (checkpoint / "model.pt").stat().st_size != metadata["checkpoint_bytes"]:
        raise ValueError("initial checkpoint size differs from its receipt")
    if model_factory is None:
        binding = metadata["binding"]
        model_factory = lambda: load_initial_encoder(binding["local_backbone_path"], schema,
            model_revision=binding["model_revision"], device=device, max_length=metadata["max_length"])
    model = model_factory().to(device=device, dtype=torch.float32)
    if model.binding != metadata["binding"]:
        raise ValueError("initial model/tokenizer binding differs from checkpoint")
    stored = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    if (stored.get("protocol") != PROTOCOL or stored.get("schema_hash") != schema.hash
            or stored.get("binding_hash") != stable_hash(model.binding)):
        raise ValueError("stored model identity differs from checkpoint metadata")
    model.load_state_dict(stored["state_dict"], strict=True)
    return metadata, model.eval()


def predict_initial_source(prepared, checkpoint, output, *, roles=("head_fit", "policy_fit", "policy_tune"),
                           allow_protected=False, batch_size=16, device="cuda:0"):
    schema, records, provenance = load_source_records(prepared, roles, allow_protected=allow_protected)
    checkpoint, output = Path(checkpoint), Path(output)
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("initial prediction output already exists")
    metadata = read_json(checkpoint / "checkpoint.json")
    if schema.hash != Schema.from_dict(metadata["schema"]).hash:
        raise ValueError("prepared and checkpoint native schemas differ")
    metadata, model = load_initial_checkpoint(checkpoint, device=device)
    truncation = model.truncation_report([r.text for r in records])
    _sync(model.device)
    started = time.perf_counter()
    values, probabilities = predict_initial_records(model, records, batch_size=batch_size)
    _sync(model.device)
    elapsed = time.perf_counter() - started
    rows = [{"sample_id": record.sample_id, "group_id": record.group_id, "split": record.role,
             "A": values[i].tolist(), "A_probabilities": [p[i].tolist() for p in probabilities]}
            for i, record in enumerate(records)]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    report = {"protocol": PROTOCOL, "n": len(records), "roles": list(roles), "seed": metadata["seed"],
              "checkpoint_identity": metadata["checkpoint_identity"], "binding": model.binding,
              "fit_sample_ids": metadata["fit_sample_ids"], "fit_group_ids": metadata["fit_group_ids"],
              "source_training_provenance": metadata["provenance"],
              "source_training_membership_sha256": metadata["provenance"].get("membership_sha256"),
              "checkpoint_metadata_sha256": file_hash(checkpoint / "checkpoint.json"),
              "model_sha256": file_hash(checkpoint / "model.pt"),
              "predictions_sha256": file_hash(output),
              "provenance": provenance, "truncation": truncation, "batch_size": batch_size,
              "prediction_seconds": elapsed, "backbone_forwards": (len(records) + batch_size - 1) // batch_size,
              "timing_scope": "tokenization/backbone/concept heads; excludes load, truncation audit, and downstream policy",
              "semantic_value_encoding": "zero-based schema-native category IDs; no runtime unknown category",
              "schema": schema.to_dict(), "artifact_visibility": "PRIVATE_RUNTIME", "paper_evidence": False}
    write_json(metadata_path, report)
    return report

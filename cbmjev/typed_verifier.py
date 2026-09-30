"""Frozen contextual candidate readouts and a shared-encoding concept reference.

This is Jev-inspired typed verification, not an official Jev implementation or
LM sequence log-likelihood. A scalar readout scores each complete candidate's
contextual hidden state. Both sources use the same public source-role labels.
No task labels are accepted by the fitting or feature-extraction interfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import shutil
import time

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import ModelInput, Schema, stable_hash
from .io import file_hash, fresh_dir, read_json, read_jsonl, write_json, write_jsonl
from .nanojev import last_valid_pool
from .responders import ConceptTrainingExample, validate_atoms, validate_examples


SOURCE_ROLES = frozenset(("responder_fit", "source_fit", "source-fit"))
PROTECTED_ROLES = frozenset(("test", "confirmation", "locked_confirmation", "locked-confirmation"))
KNOWN_ROLES = SOURCE_ROLES | PROTECTED_ROLES | frozenset((
    "head_fit", "policy_fit", "policy_tune", "validation", "calibration"))
PROTOCOL = "cbmjev-contextual-candidate-readout-v1"
TEMPLATE_VERSION = "nonthinking-segmented-full-candidate-v1"
LEARNING_RATES = (3e-4, 1e-3)


@dataclass(frozen=True)
class SourceRecord:
    sample_id: str
    group_id: str
    role: str
    text: str
    concepts: tuple[int | None, ...]


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be a nonempty string")
    return value


def load_source_records(prepared, roles=("responder_fit",), *, allow_protected=False):
    """Join standard prepared JSONL by ID; project away all task/extra fields.

    Labels from non-source roles are deliberately omitted. Protected inference
    roles require an explicit override; that override never authorizes fitting.
    """
    prepared = Path(prepared)
    if not roles or len(set(roles)) != len(roles):
        raise ValueError("roles must be nonempty and distinct")
    if set(roles) - KNOWN_ROLES:
        raise ValueError("unknown requested role")
    if set(roles) & PROTECTED_ROLES and not allow_protected:
        raise ValueError("protected test/confirmation extraction requires explicit override")
    schema = Schema.from_dict(read_json(prepared / "schema.json"))
    members = read_jsonl(prepared / "membership.jsonl")
    member_map, group_roles = {}, {}
    for member in members:
        sid = _identifier(member.get("sample_id"), "membership.sample_id")
        gid = _identifier(member.get("group_id"), "membership.group_id")
        role = _identifier(member.get("split"), "membership.split")
        if role not in KNOWN_ROLES:
            raise ValueError("unknown membership role")
        if sid in member_map:
            raise ValueError("duplicate membership sample_id")
        if gid in group_roles and group_roles[gid] != role:
            raise ValueError("group crosses prepared roles")
        group_roles[gid] = role
        member_map[sid] = member
    if set(roles) - {row["split"] for row in members}:
        raise ValueError("requested role is absent from membership")
    records, seen = [], set()
    for row in read_jsonl(prepared / "samples.jsonl"):
        sid = _identifier(row.get("sample_id"), "sample_id")
        if sid in seen or sid not in member_map:
            raise ValueError("duplicate sample or sample missing from membership")
        seen.add(sid)
        member = member_map[sid]
        if row.get("group_id") != member["group_id"]:
            raise ValueError("sample/membership group mismatch")
        role = member["split"]
        outer = member.get("outer_split")
        if outer is not None and row.get("split") != outer:
            raise ValueError("sample/membership outer_split mismatch")
        if role in SOURCE_ROLES and outer is not None and outer not in ("train", role):
            raise ValueError("source role cannot relabel held-out outer_split")
        if role not in roles:
            continue
        if row.get("dataset") != schema.dataset:
            raise ValueError("sample dataset differs from schema")
        content = row.get("input", {})
        text = content.get("text")
        if content.get("modality", "text") != "text" or not isinstance(text, str) or not text.strip():
            raise ValueError("typed verifier requires nonempty event/review text only")
        labels = [None] * schema.num_atoms
        if role in SOURCE_ROLES:
            annotations = row.get("concepts")
            if not isinstance(annotations, list) or len(annotations) != schema.num_atoms:
                raise ValueError("source concept label width mismatch")
            by_id = {item["concept_id"]: item for item in annotations}
            if len(by_id) != schema.num_atoms or set(by_id) != {c.id for c in schema.concepts}:
                raise ValueError("source concept IDs differ from schema")
            for j, concept in enumerate(schema.concepts):
                annotation = by_id[concept.id]
                value = annotation.get("value")
                status = annotation.get("annotation_status")
                if value is not None and (type(value) is not int or not 0 <= value < len(concept.values)):
                    raise ValueError("concept labels must be zero-based native categories or None")
                if value is not None and status != "OBSERVED":
                    raise ValueError("missing annotation cannot carry a semantic label")
                if value is None and status == "OBSERVED":
                    raise ValueError("observed annotation lacks a semantic label")
                labels[j] = value
        records.append(SourceRecord(sid, member["group_id"], role, text, tuple(labels)))
    if seen != set(member_map):
        raise ValueError("samples and membership IDs do not align")
    if not records:
        raise ValueError("no rows in requested roles")
    records.sort(key=lambda r: r.sample_id)
    provenance = {"schema_file_sha256": file_hash(prepared / "schema.json"),
                  "membership_sha256": file_hash(prepared / "membership.jsonl"),
                  "roles": list(roles), "protected_override": bool(allow_protected)}
    return schema, records, provenance


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class FrozenTypedEncoder:
    """Explicit token budgets keep all question/candidate tokens under truncation.

    Text is tokenized once and reused, unchanged, across all concept candidates
    and the shared encoding. Only evidence tokens may be truncated. Segmented
    tokenization is intentional and part of the versioned prompt contract.
    """
    def __init__(self, backbone, tokenizer, schema, *, binding, max_length=512):
        if type(max_length) is not int or max_length < 1:
            raise ValueError("max_length must be positive")
        self.backbone, self.tokenizer, self.schema = backbone, tokenizer, schema
        self.max_length = max_length
        self.hidden_size = getattr(backbone.config, "hidden_size", None)
        if type(self.hidden_size) is not int or self.hidden_size < 1:
            raise ValueError("backbone must declare hidden_size")
        self.backbone.requires_grad_(False).eval()
        self.device = next(backbone.parameters()).device
        self.dtype = next(backbone.parameters()).dtype
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer requires explicit pad_token_id")
        marker = "__CBMJEV_TEXT_TOKEN_BOUNDARY_0729__"
        system = ("Assess only the named concept from the quoted text. Use its native categories. "
                  "The assistant response is a complete candidate value. Do not infer a task label.")
        self.templates = []
        for concept in schema.concepts:
            templates = []
            for value in concept.values:
                question = ("Quoted text:\n" + marker + "\n\nConcept: " + concept.id + ". " +
                            concept.description + "\nNative values: " + "; ".join(concept.values) +
                            "\nWhich native value describes this concept in the text?")
                rendered = tokenizer.apply_chat_template(
                    [{"role": "system", "content": system}, {"role": "user", "content": question},
                     {"role": "assistant", "content": value}], tokenize=False,
                    add_generation_prompt=False, enable_thinking=False)
                templates.append(self._split_template(rendered, marker))
            self.templates.append(templates)
        shared = tokenizer.apply_chat_template(
            [{"role": "system", "content": "Represent the quoted text for concept measurement."},
             {"role": "user", "content": "Quoted text:\n" + marker + "\n\nEnd of quoted text."}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False)
        self.shared_template = self._split_template(shared, marker)
        all_templates = [pair for group in self.templates for pair in group] + [self.shared_template]
        overhead = max(len(prefix) + len(suffix) for prefix, suffix in all_templates)
        self.text_token_budget = max_length - overhead
        self.truncation_marker = tokenizer.encode("\n[TEXT TRUNCATED]\n", add_special_tokens=False)
        if self.text_token_budget <= len(self.truncation_marker) + 1:
            raise ValueError("max_length leaves no evidence token budget after complete question/candidate")
        self.binding = {**binding, "schema_hash": schema.hash, "template_version": TEMPLATE_VERSION,
                        "template_sha256": stable_hash(all_templates), "hidden_size": self.hidden_size,
                        "max_length": max_length, "text_token_budget": self.text_token_budget,
                        "text_truncation": "head_tail_with_explicit_marker_shared_across_sources",
                        "pooling": "last_valid_full_candidate_path", "position_ids": "mask_cumsum_minus_one",
                        "dtype": str(self.dtype), "thinking": False,
                        "device_type": self.device.type,
                        "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
                        "method": "contextual_candidate_readout_not_lm_log_likelihood"}

    def _split_template(self, rendered, marker):
        if rendered.count(marker) != 1:
            raise ValueError("chat template must retain exactly one evidence boundary")
        prefix, suffix = rendered.split(marker)
        return (self.tokenizer.encode(prefix, add_special_tokens=False),
                self.tokenizer.encode(suffix, add_special_tokens=False))

    def _encode_paths(self, paths, batch_size, padding_side, timing):
        features = []
        for start in range(0, len(paths), batch_size):
            begin = time.perf_counter()
            batch = paths[start:start + batch_size]
            length = max(map(len, batch))
            ids = torch.full((len(batch), length), self.tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, tokens in enumerate(batch):
                offset = length - len(tokens) if padding_side == "left" else 0
                ids[i, offset:offset + len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                mask[i, offset:offset + len(tokens)] = 1
            ids, mask = ids.to(self.device), mask.to(self.device)
            # Same explicit padding convention and pooling as NanoCandidateScorer.
            positions = (mask.cumsum(-1) - 1).clamp_min(0)
            _synchronize(self.device)
            timing["tokenization_and_batch_seconds"] += time.perf_counter() - begin
            begin = time.perf_counter()
            options = {"use_cache": False} if hasattr(self.backbone.config, "use_cache") else {}
            with torch.inference_mode():
                hidden = self.backbone(input_ids=ids, attention_mask=mask, position_ids=positions,
                                       return_dict=True, **options).last_hidden_state
                _synchronize(self.device)
                timing["model_forward_seconds"] += time.perf_counter() - begin
                begin = time.perf_counter()
                pooled = last_valid_pool(hidden, mask)
                _synchronize(self.device)
                timing["pooling_seconds"] += time.perf_counter() - begin
                begin = time.perf_counter()
                features.append(pooled.detach().to(device="cpu").clone())
                _synchronize(self.device)
                timing["host_transfer_seconds"] += time.perf_counter() - begin
            timing["encoder_forwards"] += 1
            timing["input_tokens"] += sum(map(len, batch))
            timing["padded_tokens"] += len(batch) * length
        return torch.cat(features)

    def encode_texts(self, texts, *, batch_size=16, padding_side="right", reverse_candidates=False,
                     atom_ids=None, include_shared=True):
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("nonempty text strings required")
        if type(batch_size) is not int or batch_size < 1 or padding_side not in ("left", "right"):
            raise ValueError("invalid batch size/padding side")
        if self.backbone.training or any(p.requires_grad for p in self.backbone.parameters()):
            raise ValueError("feature cache requires frozen eval-mode backbone")
        atom_ids = tuple(range(self.schema.num_atoms)) if atom_ids is None else atom_ids
        validate_atoms(atom_ids, self.schema.num_atoms)
        if not atom_ids and not include_shared:
            raise ValueError("at least one source encoding must be requested")
        wall = time.perf_counter()
        timing = {name: 0.0 for name in ("tokenization_and_batch_seconds", "model_forward_seconds",
                                        "pooling_seconds", "host_transfer_seconds")}
        timing.update(encoder_forwards=0, input_tokens=0, padded_tokens=0, text_tokens_original=0,
                      text_tokens_retained=0, truncated_texts=0, n=len(texts))
        begin = time.perf_counter()
        candidate_paths, shared_paths, restore = [], [], []
        total_candidates = sum(self.schema.value_counts[j] for j in atom_ids)
        for sample_i, text in enumerate(texts):
            evidence = self.tokenizer.encode(text, add_special_tokens=False)
            original = len(evidence)
            timing["text_tokens_original"] += original
            if original > self.text_token_budget:
                remaining = self.text_token_budget - len(self.truncation_marker)
                left = (remaining + 1) // 2
                right = remaining // 2
                evidence = evidence[:left] + self.truncation_marker + evidence[-right:]
                timing["truncated_texts"] += 1
                timing["text_tokens_retained"] += remaining
            else:
                timing["text_tokens_retained"] += original
            concept_offset = 0
            for j in atom_ids:
                templates = self.templates[j]
                order = list(range(len(templates)))
                if reverse_candidates:
                    order.reverse()
                for value in order:
                    prefix, suffix = templates[value]
                    candidate_paths.append(prefix + evidence + suffix)
                    restore.append(sample_i * total_candidates + concept_offset + value)
                concept_offset += len(templates)
            if include_shared:
                prefix, suffix = self.shared_template
                shared_paths.append(prefix + evidence + suffix)
        timing["tokenization_and_batch_seconds"] += time.perf_counter() - begin
        if max(map(len, candidate_paths + shared_paths)) > self.max_length:
            raise ValueError("internal token budget overflow")
        before = dict(timing)
        if candidate_paths:
            candidate = self._encode_paths(candidate_paths, batch_size, padding_side, timing)
            canonical = torch.empty_like(candidate)
            canonical[torch.tensor(restore)] = candidate
        else:
            canonical = torch.empty((0, self.hidden_size), dtype=self.dtype)
        component_fields = ("tokenization_and_batch_seconds", "model_forward_seconds", "pooling_seconds",
                            "host_transfer_seconds", "encoder_forwards", "input_tokens", "padded_tokens")
        candidate_timing = {key: timing[key] - before[key] for key in component_fields}
        before = dict(timing)
        shared = self._encode_paths(shared_paths, batch_size, padding_side, timing) if shared_paths else None
        shared_timing = {key: timing[key] - before[key] for key in component_fields}
        timing.update(wall_seconds=time.perf_counter() - wall, candidate_paths=len(candidate_paths),
                      shared_text_paths=len(shared_paths),
                      atom_ids=list(atom_ids), candidate_timing=candidate_timing, shared_timing=shared_timing,
                      scope="OFFLINE_FEATURE_EXTRACTION_NOT_END_TO_END_DEPLOYMENT_LATENCY")
        return canonical.reshape(len(texts), total_candidates, self.hidden_size), shared, timing


def load_frozen_encoder(backbone_path, schema, *, model_revision, device="cuda:0", max_length=512):
    """Local-only pinned Qwen3-4B BF16 load; no downloads or weight-file hashing."""
    path = Path(backbone_path)
    if not path.is_dir() or not re.fullmatch(r"[0-9a-f]{40}", model_revision or ""):
        raise ValueError("existing local backbone directory and pinned 40-hex revision required")
    config = read_json(path / "config.json")
    if any(config.get(key) != value for key, value in {
            "model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36,
            "num_attention_heads": 32, "num_key_value_heads": 8}.items()):
        raise ValueError("local config must match the fixed Qwen3-4B architecture")
    import transformers
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither padding nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    if str(device).startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise ValueError("this source recipe requires BF16-capable CUDA")
    backbone = AutoModel.from_pretrained(str(path), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    if getattr(backbone.config, "model_type", None) != "qwen3":
        raise ValueError("source recipe is pinned to Qwen3-4B, not a backbone search")
    resolved = getattr(backbone.config, "_commit_hash", None)
    if resolved is not None and resolved != model_revision:
        raise ValueError("resolved model revision differs from declared revision")
    metadata_files = {name: file_hash(path / name) for name in
                      ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
                      if (path / name).is_file()}
    binding = {"model_id": "Qwen/Qwen3-4B", "model_revision": model_revision,
               "revision_verification": "hf_config_commit" if resolved else "operator_pinned_local_snapshot",
               "tokenizer_revision": model_revision, "metadata_sha256": metadata_files,
               "attention_implementation": "eager", "torch_version": str(torch.__version__),
               "transformers_version": transformers.__version__, "backbone_frozen": True}
    return FrozenTypedEncoder(backbone, tokenizer, schema, binding=binding, max_length=max_length)


def record_metadata(record):
    return {"sample_id": record.sample_id, "group_id": record.group_id, "role": record.role,
            "text_sha256": hashlib.sha256(record.text.encode()).hexdigest(),
            "concepts": list(record.concepts)}


def extract_feature_cache(encoder, records, output, *, provenance=None, batch_size=16,
                          shard_size=32, padding_side="right", max_cache_bytes=64 * 1024**3,
                          progress=None):
    """Write bounded shards; estimate storage and check free disk before encoding."""
    if not records or type(shard_size) is not int or shard_size < 1:
        raise ValueError("nonempty records and positive shard_size required")
    if len({r.sample_id for r in records}) != len(records):
        raise ValueError("duplicate source sample_id")
    if any(r.role not in SOURCE_ROLES and any(x is not None for x in r.concepts) for r in records):
        raise ValueError("non-source labels cannot enter feature cache")
    element_bytes = torch.empty((), dtype=encoder.dtype).element_size()
    estimated = len(records) * (sum(encoder.schema.value_counts) + 1) * encoder.hidden_size * element_bytes
    output = Path(output)
    disk_parent = output
    while not disk_parent.exists():
        disk_parent = disk_parent.parent
    free_bytes = shutil.disk_usage(disk_parent).free
    required = math.ceil(estimated * 1.1) + 1024 * 1024
    if estimated > max_cache_bytes or required > free_bytes:
        raise ValueError("feature cache exceeds max_cache_bytes or available disk before allocation")
    output = fresh_dir(output)
    binding = {**encoder.binding, "batch_size": batch_size, "padding_side": padding_side,
               "shard_size": shard_size}
    shards, timings = [], []
    started = time.perf_counter()
    for start in range(0, len(records), shard_size):
        subset = records[start:start + shard_size]
        candidate, shared, timing = encoder.encode_texts([r.text for r in subset],
            batch_size=batch_size, padding_side=padding_side)
        name = "features-{:05d}.pt".format(len(shards))
        torch.save({"candidate": candidate, "shared": shared}, output / name)
        shards.append({"file": name, "start": start, "count": len(subset),
                       "sha256": file_hash(output / name)})
        timings.append(timing)
        if progress is not None:
            completed = start + len(subset)
            elapsed = time.perf_counter() - started
            progress({"event": "feature_cache_progress", "completed_samples": completed,
                      "total_samples": len(records), "shards_written": len(shards),
                      "elapsed_seconds": elapsed, "last_shard_extraction_seconds": timing["wall_seconds"],
                      "estimated_remaining_seconds": elapsed / completed * (len(records) - completed)})
    index = {"protocol": PROTOCOL, "schema": encoder.schema.to_dict(), "binding": binding,
             "rows": [record_metadata(r) for r in records], "provenance": provenance or {},
             "shards": shards, "estimated_feature_bytes": estimated,
             "free_disk_bytes_before": free_bytes, "timing": timings,
             "timing_scope": "offline extraction includes tokenization/model/pooling; excludes readout/policy/fusion/task head"}
    index["index_sha256"] = stable_hash(index)
    write_json(output / "index.json", index)
    return index


def load_feature_cache(directory):
    directory = Path(directory)
    index = read_json(directory / "index.json")
    if index.get("protocol") != PROTOCOL:
        raise ValueError("unknown feature cache protocol")
    payload = {k: v for k, v in index.items() if k != "index_sha256"}
    if stable_hash(payload) != index.get("index_sha256"):
        raise ValueError("feature cache index binding mismatch")
    schema = Schema.from_dict(index["schema"])
    if schema.hash != index["binding"].get("schema_hash"):
        raise ValueError("cache schema binding mismatch")
    candidates, shared, count = [], [], 0
    width = index["binding"]["hidden_size"]
    for shard in index["shards"]:
        name = shard["file"]
        if Path(name).name != name or shard["start"] != count:
            raise ValueError("invalid cache shard path or order")
        if file_hash(directory / name) != shard["sha256"]:
            raise ValueError("feature shard content hash mismatch")
        tensors = torch.load(directory / name, map_location="cpu", weights_only=True)
        a, b = tensors["candidate"], tensors["shared"]
        expected = (shard["count"], sum(schema.value_counts), width)
        if tuple(a.shape) != expected or tuple(b.shape) != (shard["count"], width):
            raise ValueError("cached feature shape mismatch")
        if str(a.dtype) != index["binding"]["dtype"] or b.dtype != a.dtype:
            raise ValueError("cached feature dtype mismatch")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("nonfinite cached features")
        candidates.append(a)
        shared.append(b)
        count += shard["count"]
    if count != len(index["rows"]) or not count:
        raise ValueError("cache sample count mismatch")
    return index, torch.cat(candidates), torch.cat(shared)


class ConceptReadout(nn.Module):
    """Fixed small architecture: affine LayerNorm and linear concept scores."""
    def __init__(self, hidden_size, value_counts, kind):
        super().__init__()
        if kind not in ("typed", "shared"):
            raise ValueError("kind must be typed or shared")
        self.kind, self.value_counts = kind, tuple(value_counts)
        self.norm = nn.LayerNorm(hidden_size)
        if kind == "typed":
            self.head = nn.Linear(hidden_size, 1, bias=False)
        else:
            self.heads = nn.ModuleList(nn.Linear(hidden_size, c) for c in value_counts)

    def forward(self, features):
        hidden = self.norm(features.float())
        if self.kind == "typed":
            if features.ndim != 3 or features.shape[1] != sum(self.value_counts):
                raise ValueError("typed features must include every canonical candidate")
            return self.head(hidden).squeeze(-1).split(self.value_counts, dim=1)
        if features.ndim != 2:
            raise ValueError("shared source uses exactly one text encoding per sample")
        return tuple(head(hidden) for head in self.heads)

    def score_candidates(self, features):
        """Score only requested candidate paths, without computing other concepts."""
        if self.kind != "typed" or features.ndim != 3:
            raise ValueError("typed candidate features required")
        return self.head(self.norm(features.float())).squeeze(-1)


def source_group_split(rows, *, split_seed=20260930, tune_fraction=.2):
    """A label-independent group split, fixed across source-training seeds."""
    if not rows or any(row["role"] not in SOURCE_ROLES for row in rows):
        raise ValueError("source fitting requires only responder_fit/source_fit; protected roles forbidden")
    if not 0 < tune_fraction < 1:
        raise ValueError("tune_fraction must be between zero and one")
    ids, text_groups = set(), {}
    for row in rows:
        sid, gid = _identifier(row["sample_id"], "sample_id"), _identifier(row["group_id"], "group_id")
        if sid in ids:
            raise ValueError("duplicate cache sample_id")
        ids.add(sid)
        text_hash = _identifier(row["text_sha256"], "text_sha256")
        if text_hash in text_groups and text_groups[text_hash] != gid:
            raise ValueError("duplicate text crosses source groups")
        text_groups[text_hash] = gid
    groups = sorted({row["group_id"] for row in rows}, key=lambda group: (
        stable_hash([PROTOCOL, "source_internal_split", split_seed, group]), group))
    if len(groups) < 2:
        raise ValueError("source internal selection requires at least two groups")
    tune_count = min(len(groups) - 1, max(1, math.ceil(len(groups) * tune_fraction)))
    tune_groups = set(groups[:tune_count])
    fit = [i for i, row in enumerate(rows) if row["group_id"] not in tune_groups]
    tune = [i for i, row in enumerate(rows) if row["group_id"] in tune_groups]
    return fit, tune, {"split_seed": split_seed, "tune_fraction_by_group": tune_fraction,
        "split_rule": "label_independent_group_hash", "fit_sample_ids": [rows[i]["sample_id"] for i in fit],
        "tune_sample_ids": [rows[i]["sample_id"] for i in tune],
        "fit_group_ids": sorted(set(groups) - tune_groups), "tune_group_ids": sorted(tune_groups)}


def _labels(rows, schema):
    examples = [ConceptTrainingExample(ModelInput(text="cached source text"), tuple(row["concepts"]))
                for row in rows]
    validate_examples(examples, schema)
    return torch.tensor([[value if value is not None else -1 for value in row["concepts"]]
                         for row in rows], dtype=torch.long)


def concept_metrics(predictions, labels, schema):
    predictions, labels = torch.as_tensor(predictions), torch.as_tensor(labels)
    if predictions.shape != labels.shape or predictions.ndim != 2 or predictions.shape[1] != schema.num_atoms:
        raise ValueError("concept metric shape mismatch")
    metrics, macros = [], []
    for j, concept in enumerate(schema.concepts):
        valid = labels[:, j] >= 0
        gold, pred = labels[valid, j], predictions[valid, j]
        count = len(concept.values)
        if torch.any((pred < 0) | (pred >= count)) or torch.any(gold >= count):
            raise ValueError("metric labels/predictions differ from native categories")
        confusion = torch.bincount(gold * count + pred, minlength=count * count).reshape(count, count)
        true_count, pred_count = confusion.sum(1), confusion.sum(0)
        denominator = true_count + pred_count
        f1 = torch.where(denominator > 0, 2 * confusion.diag().double() / denominator.clamp_min(1), 0.)
        n = len(gold)
        macro = float(f1.mean()) if n else None
        if macro is not None:
            macros.append(macro)
        ordinal = concept.values == ("1", "2", "3", "4", "5")
        metrics.append({"concept_id": concept.id, "n": n, "native_values": list(concept.values),
            "accuracy": float((pred == gold).double().mean()) if n else None, "macro_f1": macro,
            "ordinal_mae": float((pred - gold).abs().double().mean()) if n and ordinal else None,
            "confusion": confusion.tolist(), "true_counts": true_count.tolist(),
            "prediction_counts": pred_count.tolist(),
            "prediction_distribution": (pred_count.double() / n).tolist() if n else None})
    if not macros:
        raise ValueError("concept metrics require observed labels")
    return {"n_samples": len(labels), "observed_labels": int((labels >= 0).sum()),
            "aggregation": "native-class macro-F1 per concept, equal mean over labeled concepts",
            "concept_macro_f1": sum(macros) / len(macros), "concepts": metrics}


def source_gate_definition(schema, split, cache_hash):
    """Freeze this object before any readout validation output is inspected."""
    return {"protocol": PROTOCOL, "schema_hash": schema.hash, "source_cache_sha256": cache_hash,
            "source_split": split, "selection_metric": "concept_macro_f1",
            "criterion": {"macro_f1_margin_over_majority": .02, "collapse_min_labels": 50,
                          "collapse_prediction_share": .95, "collapse_true_share_max": .80,
                          "collapse_concept_fraction": .25},
            "majority_fit_role": "source_internal_fit_only",
            "no_task_labels": True, "learning_rates": list(LEARNING_RATES), "max_epochs": 10,
            "future_upgrade_only": {"implemented": False, "rank": 8, "alpha": 16,
                                    "dropout": .05, "modules": ["q_proj", "v_proj"],
                                    "learning_rates": [1e-4, 3e-4], "max_epochs": 3,
                                    "trigger": "concept gate only; never task outcome"}}


def evaluate_source_gate(report, majority, gate):
    rule = gate["criterion"]
    eligible, collapsed = [], []
    for concept in report["concepts"]:
        n = concept["n"]
        if n < rule["collapse_min_labels"]:
            continue
        eligible.append(concept["concept_id"])
        winner = max(range(len(concept["prediction_counts"])), key=concept["prediction_counts"].__getitem__)
        if (concept["prediction_counts"][winner] / n >= rule["collapse_prediction_share"]
                and concept["true_counts"][winner] / n <= rule["collapse_true_share_max"]):
            collapsed.append(concept["concept_id"])
    macro_failure = report["concept_macro_f1"] <= majority["concept_macro_f1"] + rule["macro_f1_margin_over_majority"]
    collapse_failure = bool(eligible) and len(collapsed) / len(eligible) >= rule["collapse_concept_fraction"]
    return {"upgrade_triggered": bool(macro_failure or collapse_failure), "macro_trigger": macro_failure,
            "collapse_trigger": collapse_failure, "eligible_concepts": eligible,
            "collapsed_concepts": collapsed, "low_sample_concepts": [c["concept_id"] for c in
                report["concepts"] if c["n"] < rule["collapse_min_labels"]],
            "lora_executed": False, "scope": "engineering source screen, not evidence of task benefit"}


def _predict_readout(model, features, *, batch_size=256):
    model.eval()
    probabilities = [[] for _ in model.value_counts]
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            for j, logits in enumerate(model(features[start:start + batch_size])):
                probabilities[j].append(logits.softmax(-1))
    probabilities = [torch.cat(parts) for parts in probabilities]
    values = torch.stack([p.argmax(1) for p in probabilities], 1)
    return values, probabilities


def _fit_readout(features, labels, schema, kind, fit_indices, *, learning_rate,
                  epochs, batch_size, seed, tune_indices=None):
    """Only cached features are accepted; the backbone cannot be rerun here."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ConceptReadout(features.shape[-1], schema.value_counts, kind)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.)
        generator = torch.Generator().manual_seed(seed)
        history, best = [], None
        indices = torch.tensor(fit_indices, dtype=torch.long)
        for epoch in range(1, epochs + 1):
            model.train()
            order = indices[torch.randperm(len(indices), generator=generator)]
            total_loss, observed = 0., 0
            for start in range(0, len(order), batch_size):
                subset = order[start:start + batch_size]
                logits = model(features[subset])
                losses, n = [], 0
                for j, head_logits in enumerate(logits):
                    valid = labels[subset, j] >= 0
                    if valid.any():
                        losses.append(F.cross_entropy(head_logits[valid], labels[subset, j][valid], reduction="sum"))
                        n += int(valid.sum())
                if not losses:
                    continue
                loss = torch.stack(losses).sum() / n
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * n
                observed += n
            result = {"epoch": epoch, "observed_label_exposures": observed,
                      "training_ce": total_loss / observed if observed else None}
            if tune_indices is not None:
                values, _ = _predict_readout(model, features[tune_indices])
                metrics = concept_metrics(values, labels[tune_indices], schema)
                result["tune_concept_macro_f1"] = metrics["concept_macro_f1"]
                # Strict > makes ties prefer the first epoch and first declared LR.
                if best is None or metrics["concept_macro_f1"] > best["metrics"]["concept_macro_f1"]:
                    best = {"epoch": epoch, "metrics": metrics, "learning_rate": learning_rate}
            history.append(result)
        return model.eval(), history, best


def train_cached_sources(cache_directory, output, *, seed=40, epochs=10,
                         batch_size=64, split_seed=20260930):
    if type(epochs) is not int or not 1 <= epochs <= 10 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("source recipe permits 1..10 epochs and a positive batch size")
    index, typed_features, shared_features = load_feature_cache(cache_directory)
    schema = Schema.from_dict(index["schema"])
    rows = index["rows"]
    fit, tune, split = source_group_split(rows, split_seed=split_seed)
    labels = _labels(rows, schema)
    if torch.any((labels[fit] >= 0).sum(0) == 0):
        raise ValueError("each concept needs at least one valid source-internal-fit label")
    output = fresh_dir(output)
    gate = source_gate_definition(schema, split, index["index_sha256"])
    gate["requested_epochs"] = epochs
    gate["training_seed"] = seed
    # This immutable write MUST precede computing even the majority tune metric.
    write_json(output / "SOURCE_GATE.json", gate)
    majority_values = []
    for j, count in enumerate(schema.value_counts):
        values = labels[fit, j]
        majority_values.append(int(torch.bincount(values[values >= 0], minlength=count).argmax()))
    majority = concept_metrics(torch.tensor(majority_values).repeat(len(tune), 1), labels[tune], schema)
    reports, states, configs = {}, {}, {}
    started = time.perf_counter()
    for kind, features in (("typed", typed_features), ("shared", shared_features)):
        trials, selected = [], None
        for learning_rate in LEARNING_RATES:
            _, history, best = _fit_readout(features, labels, schema, kind, fit,
                learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, seed=seed, tune_indices=tune)
            trials.append({"learning_rate": learning_rate, "history": history, "best": best})
            if selected is None or best["metrics"]["concept_macro_f1"] > selected["metrics"]["concept_macro_f1"]:
                selected = best
        model, history, _ = _fit_readout(features, labels, schema, kind, list(range(len(rows))),
            learning_rate=selected["learning_rate"], epochs=selected["epoch"], batch_size=batch_size, seed=seed)
        states[kind] = model.state_dict()
        configs[kind] = {"kind": kind, "hidden_size": features.shape[-1],
                         "value_counts": list(schema.value_counts), "learning_rate": selected["learning_rate"],
                         "epochs": selected["epoch"], "parameters": sum(p.numel() for p in model.parameters())}
        reports[kind] = {"trials": trials, "selected": selected, "full_source_refit_history": history,
                         "full_source_refit_n": len(rows), "selection_role": "source_internal_tune",
                         "majority_comparator": majority, "gate": evaluate_source_gate(selected["metrics"], majority, gate)}
    metadata = {"protocol": PROTOCOL, "schema": schema.to_dict(), "binding": index["binding"],
                "source_cache_sha256": index["index_sha256"], "source_gate_sha256": stable_hash(gate),
                "source_provenance": index["provenance"],
                "source_provenance_scope": "prepared membership and schema used for source-role concept supervision",
                "source_split": split, "fit_sample_ids": [r["sample_id"] for r in rows],
                "fit_record_sha256": {r["sample_id"]: stable_hash(r) for r in rows},
                "fit_group_ids": sorted({r["group_id"] for r in rows}), "seed": seed,
                "supervision": "public source-role concept labels only; no task labels",
                "architecture": "affine LayerNorm plus linear scores; frozen backbone",
                "configs": configs, "training_batch_size": batch_size, "weight_decay": 0.,
                "cached_readout_training_seconds": time.perf_counter() - started,
                "backbone_forwards_during_training": 0, "paper_evidence": False}
    torch.save(states, output / "readouts.pt")
    metadata["weights_sha256"] = file_hash(output / "readouts.pt")
    write_json(output / "checkpoint.json", metadata)
    write_json(output / "selection_report.json", reports)
    return metadata, reports


def load_readouts(checkpoint, *, expected_binding=None):
    checkpoint = Path(checkpoint)
    metadata = read_json(checkpoint / "checkpoint.json")
    if metadata.get("protocol") != PROTOCOL:
        raise ValueError("unknown readout checkpoint protocol")
    schema = Schema.from_dict(metadata["schema"])
    if schema.hash != metadata["binding"].get("schema_hash") or set(metadata["configs"]) != {"typed", "shared"}:
        raise ValueError("checkpoint schema/source configuration mismatch")
    if expected_binding is not None and metadata["binding"] != expected_binding:
        raise ValueError("checkpoint and cache model/schema/batch binding differ; run audit before changing conditions")
    if file_hash(checkpoint / "readouts.pt") != metadata["weights_sha256"]:
        raise ValueError("readout weight binding mismatch")
    gate = read_json(checkpoint / "SOURCE_GATE.json")
    if stable_hash(gate) != metadata["source_gate_sha256"]:
        raise ValueError("frozen source gate changed")
    if gate["schema_hash"] != schema.hash or gate["source_cache_sha256"] != metadata["source_cache_sha256"]:
        raise ValueError("source gate/schema/cache binding mismatch")
    states = torch.load(checkpoint / "readouts.pt", map_location="cpu", weights_only=True)
    models = {}
    with torch.random.fork_rng(devices=[]):
        for kind, config in metadata["configs"].items():
            if (config["kind"] != kind or tuple(config["value_counts"]) != schema.value_counts
                    or config["hidden_size"] != metadata["binding"]["hidden_size"]):
                raise ValueError("readout configuration differs from bound schema/backbone")
            model = ConceptReadout(config["hidden_size"], config["value_counts"], kind)
            model.load_state_dict(states[kind], strict=True)
            models[kind] = model.eval()
    return metadata, models


def predict_cached_sources(cache_directory, checkpoint, output, *, allow_protected=False):
    index, typed_features, shared_features = load_feature_cache(cache_directory)
    if any(row["role"] in PROTECTED_ROLES for row in index["rows"]) and not allow_protected:
        raise ValueError("protected test/confirmation prediction requires explicit override")
    metadata, models = load_readouts(checkpoint, expected_binding=index["binding"])
    started = time.perf_counter()
    typed, typed_probs = _predict_readout(models["typed"], typed_features)
    shared, shared_probs = _predict_readout(models["shared"], shared_features)
    elapsed = time.perf_counter() - started
    rows = []
    for i, record in enumerate(index["rows"]):
        rows.append({"sample_id": record["sample_id"], "group_id": record["group_id"], "split": record["role"],
                     "typed_values": typed[i].tolist(), "shared_values": shared[i].tolist(),
                     "typed_probabilities": [p[i].tolist() for p in typed_probs],
                     "shared_probabilities": [p[i].tolist() for p in shared_probs]})
    output = Path(output)
    if output.exists() or output.with_suffix(output.suffix + ".metadata.json").exists():
        raise FileExistsError("prediction output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    report = {"protocol": PROTOCOL, "n": len(rows), "source_cache_sha256": index["index_sha256"],
              "fit_source_cache_sha256": metadata["source_cache_sha256"],
              "source_checkpoint_sha256": file_hash(Path(checkpoint) / "checkpoint.json"),
              "predictions_sha256": file_hash(output),
              "readout_weights_sha256": metadata["weights_sha256"], "binding": index["binding"],
              "fit_sample_ids": metadata["fit_sample_ids"], "fit_group_ids": metadata["fit_group_ids"],
              "source_split": metadata["source_split"],
              "provenance": index["provenance"], "fit_provenance": metadata.get("source_provenance"),
              "provenance_scope": ("provenance identifies prediction-cache prepared membership; "
                  "fit_provenance identifies source-supervision prepared membership; "
                  "null fit_provenance means a legacy checkpoint did not store it; "
                  "fit IDs cover final full-source refit, source_split records internal selection; "
                  "hashes bind artifacts but do not authenticate externally supplied role histories"),
              "cached_readout_replay_seconds": elapsed,
              "timing_scope": "CACHE_REPLAY_ONLY_NOT_END_TO_END_LATENCY", "backbone_forwards": 0}
    write_json(output.with_suffix(output.suffix + ".metadata.json"), report)
    return report


def predict_text_sources(encoder, models, texts, *, atom_ids=None, include_shared=True,
                          batch_size=16, padding_side="right"):
    """Live source inference, including tokenization/backbone/readout time.

    ``atom_ids=(j,), include_shared=False`` is real singleton verification.
    ``atom_ids=(), include_shared=True`` encodes each text once for all strong
    heads. The returned source time excludes A/policy/fusion/final task head.
    """
    atom_ids = tuple(range(encoder.schema.num_atoms)) if atom_ids is None else atom_ids
    for kind in (("typed",) if not include_shared else ("typed", "shared")):
        if models[kind].value_counts != encoder.schema.value_counts:
            raise ValueError("live readout/schema category mismatch")
    started = time.perf_counter()
    candidate, shared, timing = encoder.encode_texts(texts, batch_size=batch_size, padding_side=padding_side,
                                                    atom_ids=atom_ids, include_shared=include_shared)
    begin = time.perf_counter()
    result = {"atom_ids": list(atom_ids)}
    with torch.inference_mode():
        if atom_ids:
            models["typed"].eval()
            logits = models["typed"].score_candidates(candidate)
            probabilities = [values.softmax(-1) for values in
                             logits.split([encoder.schema.value_counts[j] for j in atom_ids], dim=1)]
            result["typed_values"] = torch.stack([p.argmax(1) for p in probabilities], 1).tolist()
            result["typed_probabilities"] = [[p[i].tolist() for p in probabilities] for i in range(len(texts))]
        if include_shared:
            values, probabilities = _predict_readout(models["shared"], shared)
            result["shared_values"] = values.tolist()
            result["shared_probabilities"] = [[p[i].tolist() for p in probabilities] for i in range(len(texts))]
    timing["readout_seconds"] = time.perf_counter() - begin
    timing["source_prediction_seconds"] = time.perf_counter() - started
    timing["scope"] = "LIVE_SOURCE_INFERENCE_ONLY_EXCLUDES_INITIAL_SOURCE_POLICY_FUSION_AND_TASK_HEAD"
    result["cost_record"] = timing
    return result


def audit_source_conditions(encoder, records, checkpoint, *, batch_size=16, limit=100):
    """Compare discrete predictions for fixed internal-tune IDs, without labels.

    BF16 may differ numerically across batch conditions. Any changed hard answer
    fails this audit; exact extraction bindings remain mandatory even on pass.
    """
    metadata, models = load_readouts(checkpoint)
    core_binding = {k: v for k, v in metadata["binding"].items()
                    if k not in ("batch_size", "padding_side", "shard_size")}
    if encoder.binding != core_binding:
        raise ValueError("audit encoder differs from trained backbone/schema/template binding")
    by_id = {record.sample_id: record for record in records if record.role in SOURCE_ROLES}
    ids = metadata["source_split"]["tune_sample_ids"][:limit]
    if not ids or any(sid not in by_id for sid in ids):
        raise ValueError("audit requires the fixed source internal-tune samples")
    if any(stable_hash(record_metadata(by_id[sid])) != metadata["fit_record_sha256"].get(sid) for sid in ids):
        raise ValueError("audit source text/role/label binding differs from training input")
    texts = [by_id[sid].text for sid in ids]
    baseline, results = None, []
    for name, batch, side, reverse in (
            ("singleton_right", 1, "right", False), ("batch_right", batch_size, "right", False),
            ("repeat_batch_right", batch_size, "right", False), ("batch_left", batch_size, "left", False),
            ("reversed_candidates", batch_size, "right", True)):
        typed, shared, timing = encoder.encode_texts(texts, batch_size=batch, padding_side=side,
                                                    reverse_candidates=reverse)
        predictions, probabilities = {}, {}
        for kind, features in (("typed", typed), ("shared", shared)):
            predictions[kind], probabilities[kind] = _predict_readout(models[kind], features)
        if baseline is None:
            baseline = predictions, probabilities
        mismatch = {kind: int((predictions[kind] != baseline[0][kind]).sum()) for kind in models}
        delta = {kind: max(float((p - q).abs().max()) for p, q in
                          zip(probabilities[kind], baseline[1][kind])) for kind in models}
        results.append({"condition": name, "batch_size": batch, "padding_side": side,
                        "canonicalized_reversed_candidates": reverse, "changed_hard_answers": mismatch,
                        "max_probability_delta": delta, "feature_extraction_timing": timing})
    return {"protocol": PROTOCOL, "scope": "engineering consistency audit; not a concept-quality or latency claim",
            "sample_ids": ids, "n": len(ids), "conditions": results,
            "passed": all(sum(row["changed_hard_answers"].values()) == 0 for row in results),
            "on_failure": "investigate numeric/semantic differences; never mix extraction condition caches",
            "readout_weights_sha256": metadata["weights_sha256"]}

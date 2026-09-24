"""Development-only paired Choice heads on structured or frozen language features."""
import math
from collections import Counter
from pathlib import Path
import random

import torch

from .choice_features import CHOICE_FEATURE_VERSION, choice_feature_width, encode_choice_features
from .choice_targets import singleton_choice_example
from .choice_training import _weights, fit_choice_head_pair
from .choice_language import fit_frozen_language_choice_pair
from .config import learning_config, resolve_config
from .contracts import stable_hash
from .crossfit_targets import DiskRecords, open_target_package
from .decision_sets import iter_decision_sets
from .io import file_hash, fresh_dir, read_json, write_json
from .nano_choice import NanoChoiceHead
from .nanojev import backbone_manifest, load_local_nano
from .pipeline import load_prepared


def _source_binding():
    names = ("choice_artifacts.py", "choice_features.py", "choice_targets.py", "choice_training.py",
             "nano_choice.py", "decision_sets.py", "crossfit_targets.py", "crossfit_training.py",
             "provenance.py", "learning.py", "contracts.py", "config.py", "pipeline.py", "io.py",
             "choice_runtime.py", "runtime.py", "choice_language.py", "choice_encoding.py", "nanojev.py")
    return {name: file_hash(Path(__file__).parent / name) for name in names}


def _inputs(prepared, targets, config):
    paths = [Path(prepared) / name for name in ("schema.json", "samples.jsonl", "membership.jsonl")]
    paths.append(Path(config))
    for target in targets:
        paths.append(Path(target))
        records = open_target_package(target)["records"]
        if isinstance(records, DiskRecords):
            paths.extend(records._path(shard) for shard in records.shards)
    return {str(path.resolve()): file_hash(path) for path in paths}


def _configuration(path):
    config = read_json(path)
    return learning_config(resolve_config(config)) if "learning" in config else config


def _preflight(prepared, targets, config, *, validate_decisions=True,
               validate_record_roles=True):
    schema, rows, roles = load_prepared(prepared)
    cfg = _configuration(config)
    if not targets or len({str(Path(p).resolve()) for p in targets}) != len(targets):
        raise ValueError("distinct nonempty target manifest paths required")
    fit_roles = {"responder_fit", "head_fit", "policy_fit"}
    fit_groups = {role["group_id"] for role in roles.values() if role["split"] in fit_roles}
    groups, samples, packages = set(), set(), []
    for path in targets:
        package = open_target_package(path)
        # Training's later decision iterator performs its complete validation
        # before yielding the first occurrence. Reload has no later iterator,
        # so it retains the independent full validation here.
        if validate_decisions:
            iter_decision_sets(package, schema, cfg)
        if groups.intersection(package["target_group_ids"]) or samples.intersection(package["target_sample_ids"]):
            raise ValueError("target packages must have disjoint samples and groups")
        groups.update(package["target_group_ids"])
        samples.update(package["target_sample_ids"])
        if validate_record_roles:
            for record in package["records"]:
                role = roles.get(record["sample_id"])
                if (role is None or role["split"] not in fit_roles
                        or role["group_id"] != record["group_id"]):
                    raise ValueError("target sample must retain its original prepared training role/group")
        for producer in package["provenance"]:
            if not set(producer["supervised_group_ids"]).issubset(fit_groups):
                raise ValueError("target ancestry includes groups outside prepared training roles")
        packages.append(package)
    return schema, cfg, packages


def _reservoir(decisions, maximum, seed):
    """Uniform occurrence reservoir; equal states remain distinct observations."""
    rng, selected, count = random.Random(seed), [], 0
    for count, example in enumerate(decisions, 1):
        if count <= maximum:
            selected.append((count - 1, example))
        else:
            slot = rng.randrange(count)
            if slot < maximum:
                selected[slot] = (count - 1, example)
    return tuple(example for _, example in sorted(selected)), count


def train_choice_pair(prepared, targets, config, out, *, max_questions=512,
                      epochs=5, batch_size=16, learning_rate=.001, seed=60,
                      cost_weight=.03, temperature=.5, remaining_groups=None,
                      budget_mode="fixed", feature_backend="structured", backbone_path=None,
                      encoding_device=None, max_length=None, max_padded_tokens=None,
                      max_candidates_per_batch=None, capacity_control=False):
    if type(capacity_control) is not bool:
        raise ValueError("capacity_control must be boolean")
    if feature_backend not in ("structured", "frozen_language"):
        raise ValueError("unknown Choice feature_backend")
    language = feature_backend == "frozen_language"
    if language and capacity_control:
        raise ValueError("capacity control currently requires structured features")
    if language:
        if backbone_path is None or encoding_device is None:
            raise ValueError("explicit local backbone_path and encoding_device required")
        for value in (max_length, max_padded_tokens, max_candidates_per_batch):
            if type(value) is not int or value < 1:
                raise ValueError("explicit positive language encoding limits required")
        try:
            encoder_device = torch.device(encoding_device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("invalid encoding_device") from error
        if (encoder_device.type not in ("cpu", "cuda") or
                (encoder_device.type == "cuda" and (not torch.cuda.is_available() or
                 (encoder_device.index is not None and encoder_device.index >= torch.cuda.device_count())))):
            raise ValueError("encoding_device unavailable or unsupported")
        backbone_path = str(Path(backbone_path).resolve())
    elif any(v is not None for v in (backbone_path, encoding_device, max_length,
                                     max_padded_tokens, max_candidates_per_batch)):
        raise ValueError("language encoding options require frozen_language backend")
    for name, value in (("max_questions", max_questions), ("epochs", epochs), ("batch_size", batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be nonnegative integer")
    if budget_mode not in ("fixed", "uniform_remaining"):
        raise ValueError("budget_mode must be fixed or uniform_remaining")
    if budget_mode == "uniform_remaining" and remaining_groups is not None:
        raise ValueError("uniform_remaining cannot also specify remaining_groups")
    for name, value, allow_zero in (("learning_rate", learning_rate, False),
                                     ("temperature", temperature, False), ("cost_weight", cost_weight, True)):
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (not allow_zero and value == 0)):
            raise ValueError("invalid " + name)
    targets = tuple(targets)
    def current_binding():
        result = {"inputs": _inputs(prepared, targets, config), "source": _source_binding()}
        if language:
            result["backbone"] = backbone_manifest(backbone_path)
        return result

    before = current_binding()
    schema, cfg, packages = _preflight(prepared, targets, config,
                                       validate_decisions=False)
    budget = schema.num_groups if remaining_groups is None else remaining_groups
    if type(budget) is not int or not 0 <= budget <= schema.num_groups:
        raise ValueError("remaining_groups outside schema bounds")

    def examples():
        # Separate from reservoir RNG: labels and sampling outcomes cannot alter
        # budget draws. Every original occurrence still appears exactly once.
        budget_rng = random.Random("choice-budget-v1:" + str(seed))
        for package in packages:
            for decision in iter_decision_sets(package, schema, cfg):
                available = schema.num_groups - sum(schema.group_mask(decision.model_inputs.observed))
                current_budget = budget_rng.randrange(available + 1) if budget_mode == "uniform_remaining" else budget
                example = singleton_choice_example(decision, schema, remaining_groups=current_budget,
                    cost_weight=cost_weight, temperature=temperature)
                # Validate even occurrences not chosen by the reservoir.
                from .choice_features import validate_choice_inputs
                validate_choice_inputs(example.model_inputs, schema)
                yield example

    selected, count = _reservoir(examples(), max_questions, seed)
    if before != current_binding():
        raise ValueError("Choice inputs/source changed during preflight")
    destination = Path(out)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("output must be new or empty")
    language_fit = None
    if language:
        scorer = load_local_nano(backbone_path, device=str(encoder_device), max_length=max_length,
            freeze_backbone=True, feature_normalization="none")
        if (scorer.backbone_source_manifest != before["backbone"] or before != current_binding()):
            raise ValueError("Choice backbone/inputs/source changed before language fit")
        width = scorer.hidden_size
        if read_json(Path(backbone_path) / "config.json").get("hidden_size") != width:
            raise ValueError("local backbone config hidden_size differs from loaded backbone")
        scalar, attention, language_fit = fit_frozen_language_choice_pair(scorer, selected, schema,
            max_questions=max_questions, max_padded_tokens=max_padded_tokens,
            max_candidates_per_batch=max_candidates_per_batch, epochs=epochs,
            batch_size=batch_size, lr=learning_rate, seed=seed, device=cfg.get("device", "cpu"))
        fit = language_fit["trainer"]
    else:
        features = tuple(encode_choice_features(example.model_inputs, schema) for example in selected)
        width = choice_feature_width(schema)
        *fitted_heads, fit = fit_choice_head_pair(selected, features, epochs=epochs,
            batch_size=batch_size, lr=learning_rate, seed=seed, device=cfg.get("device", "cpu"),
            capacity_control=capacity_control)
        scalar, attention = fitted_heads[:2]
    head_names = ("scalar", "attention", "independent_mlp") if capacity_control else ("scalar", "attention")
    heads = (scalar, attention, fitted_heads[2]) if capacity_control else (scalar, attention)
    settings = {"format": ("cbmjev-structured-choice-capacity-config-v1" if capacity_control
                           else "cbmjev-structured-choice-pair-config-v1"), "schema_hash": schema.hash,
        "feature_version": CHOICE_FEATURE_VERSION, "feature_width": width,
        "feature_backend": feature_backend,
        "backend": ("frozen-language-choice-not-fullbody-NanoJev" if language else
                    "structured-raw-features-not-MLP-or-pretrained-JEV"),
        "full_body_training": False,
        "max_questions": max_questions, "seed": seed,
        "remaining_groups": budget if budget_mode == "fixed" else None,
        "budget_mode": budget_mode,
        "cost_weight": cost_weight, "temperature": temperature,
        "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "device": cfg.get("device", "cpu"), "allowed_evaluation_splits": ["validation"],
        "head_names": list(head_names), "capacity_control": capacity_control,
        "paper_evidence": False}
    if language:
        settings["language_encoding"] = dict(backbone_path=backbone_path,
            backbone_manifest=before["backbone"], encoding_device=str(encoder_device),
            max_length=max_length, max_padded_tokens=max_padded_tokens,
            max_candidates_per_batch=max_candidates_per_batch,
            feature_normalization="none", local_files_only=True)
    report = {"format": ("cbmjev-structured-choice-capacity-report-v1" if capacity_control
                         else "cbmjev-structured-choice-pair-report-v1"),
        "evidence_status": "DEVELOPMENT_ONLY",
        "upstream_provenance_verified": True, "original_prepared_roles_preserved": True,
        "sampling": "uniform-complete-occurrence-reservoir-without-deduplication",
        "available_occurrences": count, "selected_occurrences": len(selected),
        "selected_budget_counts": dict(sorted(Counter(str(e.model_inputs.remaining_groups)
                                                     for e in selected).items())),
        "budget_sampling": ("fixed-budget" if budget_mode == "fixed" else
                            "uniform-integer-zero-through-unobserved-groups-per-occurrence"),
        "budget_coverage_scope": "Observed marginal counts only; not joint state-budget or rollout distribution coverage. Positive budgets share a one-step teacher, not budget-dependent long-horizon values.",
        "selected_derived_ids": [example.derived_id for example in selected],
        "package_bindings": [{"path": str(Path(path).resolve()), "package_sha256": package["package_sha256"],
                              "target_artifact_id": package["target_artifact_id"]}
                             for path, package in zip(targets, packages)],
        "interpretation": "One-step realized-error-plus-cost soft imitation, not calibrated risk or Bellman Q; no MLP-vs-JEV-pretraining claim",
        "fit": fit}
    report.update(feature_backend=feature_backend, backend=settings["backend"], full_body_training=False)
    if language:
        report["language_fit"] = language_fit
        report["provenance_scope"] = "Verified target/prepared ancestry and local backbone file content, not official JEV provenance or model training ancestry."
    if before != current_binding():
        raise ValueError("Choice inputs/source changed during training")
    destination = fresh_dir(out)
    for name, head in zip(head_names, heads):
        torch.save({key: value.detach().cpu() for key, value in head.state_dict().items()}, destination / (name + ".pt"))
    write_json(destination / "config.json", settings)
    write_json(destination / "report.json", report)
    # Read-back the actual exported bytes before publishing completion.
    _read_heads(destination, settings, report, "cpu")
    if before != current_binding():
        raise ValueError("Choice inputs/source changed during export")
    receipt = {"format": ("cbmjev-structured-choice-capacity-artifact-v1" if capacity_control
                          else "cbmjev-structured-choice-pair-artifact-v1"), "status": "COMPLETE",
        "schema_hash": schema.hash, "bindings": before, "package_bindings": report["package_bindings"],
        "files": {name: file_hash(destination / name) for name in
                  (*(head_name + ".pt" for head_name in head_names), "config.json", "report.json")}}
    receipt["receipt_sha256"] = stable_hash(receipt)
    write_json(destination / "receipt.json", receipt)
    return receipt


def _read_heads(directory, settings, report, device):
    heads = []
    head_names = tuple(settings.get("head_names", ("scalar", "attention")))
    expected = (("scalar", "attention", "independent_mlp") if settings.get("capacity_control", False)
                else ("scalar", "attention"))
    if head_names != expected or set(report["fit"]["heads"]) != set(expected):
        raise ValueError("Choice head manifest mismatch")
    with torch.random.fork_rng(devices=[]):
        for name in head_names:
            kind = "none" if name == "scalar" else name
            head = NanoChoiceHead(settings["feature_width"], kind)
            weights = torch.load(Path(directory) / (name + ".pt"), map_location="cpu", weights_only=True)
            if not isinstance(weights, dict) or any(not isinstance(value, torch.Tensor) for value in weights.values()):
                raise ValueError("tensor-only Choice state dict required")
            head.load_state_dict(weights, strict=True)
            if _weights(head) != report["fit"]["heads"][name]["final_weights_sha256"]:
                raise ValueError("Choice model tensor hash mismatch")
            heads.append(head.to(device).eval().requires_grad_(False))
    return tuple(heads)


def load_choice_pair(directory, *, prepared, targets, config, device="cpu", split="validation",
                     reuse_sealed_training_validation=False):
    """Reload frozen heads; optionally reuse the exact training-time source audit.

    The fast path is only for a source-bound evaluation. The signed-format
    receipt is not an authentication mechanism, but its content hashes bind
    every prepared/config/target-shard byte to a successful training run that
    consumed and validated the full decision stream before sealing.
    """
    if split != "validation":
        raise ValueError("Choice artifacts are development/validation only, not test")
    if type(reuse_sealed_training_validation) is not bool:
        raise ValueError("reuse_sealed_training_validation must be boolean")
    directory, targets = Path(directory), tuple(targets)
    receipt = read_json(directory / "receipt.json")
    unsigned = dict(receipt)
    if (unsigned.pop("receipt_sha256", None) != stable_hash(unsigned)
            or receipt.get("format") not in ("cbmjev-structured-choice-pair-artifact-v1",
                                             "cbmjev-structured-choice-capacity-artifact-v1")
            or receipt.get("status") != "COMPLETE"):
        raise ValueError("invalid Choice receipt")
    if (not isinstance(receipt.get("files"), dict)
            or any(receipt["files"].get(name) != file_hash(directory / name)
                   for name in ("config.json", "report.json"))):
        raise ValueError("Choice artifact config/report checksum mismatch")
    settings, report = read_json(directory / "config.json"), read_json(directory / "report.json")
    capacity_control = settings.get("capacity_control", False)
    head_names = ("scalar", "attention", "independent_mlp") if capacity_control else ("scalar", "attention")
    if (type(capacity_control) is not bool
            or tuple(settings.get("head_names", head_names)) != head_names
            or set(receipt["files"]) != {*(name + ".pt" for name in head_names), "config.json", "report.json"}
            or (receipt["format"], settings.get("format"), report.get("format")) !=
               (("cbmjev-structured-choice-capacity-artifact-v1",
                 "cbmjev-structured-choice-capacity-config-v1",
                 "cbmjev-structured-choice-capacity-report-v1") if capacity_control else
                ("cbmjev-structured-choice-pair-artifact-v1",
                 "cbmjev-structured-choice-pair-config-v1",
                 "cbmjev-structured-choice-pair-report-v1"))):
        raise ValueError("Choice artifact family mismatch")
    for name in (head_name + ".pt" for head_name in head_names):
        if receipt["files"][name] != file_hash(directory / name):
            raise ValueError("Choice artifact checksum mismatch")
    backend = settings.get("feature_backend", "structured")
    if backend not in ("structured", "frozen_language"):
        raise ValueError("unknown Choice feature backend in artifact")

    def current_binding():
        result = {"inputs": _inputs(prepared, targets, config), "source": _source_binding()}
        if backend == "frozen_language":
            # Hash files and read configuration only: no backbone construction,
            # network access, or allocation on a GPU is needed to reload heads.
            encoder = settings["language_encoding"]
            manifest = backbone_manifest(encoder["backbone_path"])
            if manifest != encoder["backbone_manifest"]:
                raise ValueError("Choice backbone manifest mismatch")
            width = read_json(Path(encoder["backbone_path"]) / "config.json").get("hidden_size")
            if type(width) is not int or width < 1 or width != settings["feature_width"]:
                raise ValueError("Choice backbone feature width mismatch")
            result["backbone"] = manifest
        return result

    if receipt["bindings"] != current_binding():
        raise ValueError("Choice source/input binding mismatch")
    schema, _, packages = _preflight(prepared, targets, config,
        validate_decisions=not reuse_sealed_training_validation,
        validate_record_roles=not reuse_sealed_training_validation)
    bindings = [{"path": str(Path(path).resolve()), "package_sha256": package["package_sha256"],
                 "target_artifact_id": package["target_artifact_id"]} for path, package in zip(targets, packages)]
    if (receipt["schema_hash"] != schema.hash or settings["schema_hash"] != schema.hash
            or settings["feature_version"] != CHOICE_FEATURE_VERSION
            or (backend == "structured" and settings["feature_width"] != choice_feature_width(schema))
            or bindings != receipt["package_bindings"] or bindings != report["package_bindings"]):
        raise ValueError("Choice schema/package binding mismatch")
    heads = _read_heads(directory, settings, report, device)
    if receipt["bindings"] != current_binding():
        raise ValueError("Choice sources changed during reload")
    return (*heads, settings, report)


def verify_choice_binding_unchanged(directory, *, prepared, targets, config):
    """Rehash sealed Choice inputs/models without replaying training decisions."""
    directory = Path(directory)
    receipt = read_json(directory / "receipt.json")
    settings = read_json(directory / "config.json")
    current = {"inputs": _inputs(prepared, targets, config), "source": _source_binding()}
    if settings.get("feature_backend", "structured") == "frozen_language":
        current["backbone"] = backbone_manifest(settings["language_encoding"]["backbone_path"])
    unsigned = dict(receipt)
    if (unsigned.pop("receipt_sha256", None) != stable_hash(unsigned)
            or receipt.get("status") != "COMPLETE"
            or receipt.get("bindings") != current):
        raise ValueError("Choice receipt/source/input binding changed")
    if (not isinstance(receipt.get("files"), dict)
            or any(receipt["files"].get(name) != file_hash(directory / name)
                   for name in receipt["files"])):
        raise ValueError("Choice artifact checksum changed during evaluation")
    return receipt

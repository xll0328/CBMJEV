"""Artifact-oriented research pipeline. Real datasets and weights remain local.

The default implementation uses disjoint responder/head/policy fitting roles.
It does not pretend that a nested cross-fit plan has already been executed.
"""
import hashlib
from pathlib import Path
import math
import random
import time

from .config import learning_config, resolve_config
from .contracts import Schema, DeclaredCost, stable_hash, validate_rows
from .io import read_json, read_jsonl, write_json, write_jsonl, fresh_dir, file_hash, environment
from .runtime import (ReplayEnvironment, LiveEnvironment, load_payload, run_episode,
                      synchronize)


def code_fingerprint():
    return stable_hash({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def responder_code_fingerprint():
    """Conservative identity for semantic prompts, decoding and preprocessing."""
    root = Path(__file__).parent
    return stable_hash({name: file_hash(root / name) for name in
                        ("contracts.py", "responders.py", "nanojev.py", "runtime.py")})


def component_digest(model, schema, component, *, objective=None, pairs=None):
    """Hash only a model component's tensors, not its mixed-supervision bundle."""
    import torch
    tensors = {}
    for name, value in sorted(model.network.state_dict().items()):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        tensors[name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                         "sha256": hashlib.sha256(raw).hexdigest()}
    return stable_hash({"component": component, "schema_hash": schema.hash,
                        "tensors": tensors, "objective": objective, "pairs": pairs})


def load_prepared(prepared):
    prepared = Path(prepared)
    schema = Schema.from_dict(read_json(prepared / "schema.json"))
    rows = read_jsonl(prepared / "samples.jsonl")
    membership = read_jsonl(prepared / "membership.jsonl")
    roles = {row["sample_id"]: row for row in membership}
    if len(roles) != len(membership) or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate sample IDs in prepared records/membership")
    if set(roles) != {row["sample_id"] for row in rows}:
        raise ValueError("prepared records and membership IDs do not align")
    groups = {}
    for row in rows:
        role = roles[row["sample_id"]]
        if role["group_id"] != row["group_id"]:
            raise ValueError("membership group mismatch")
        gid = row["group_id"]
        if gid in groups and groups[gid] != role["split"]:
            raise ValueError("prepared group leaks across roles")
        groups[gid] = role["split"]
    return schema, rows, roles


class LazyConceptExamples:
    """Load one content-only example at a time instead of retaining decoded CUB."""
    def __init__(self, rows, schema, raw_root):
        self.rows, self.schema, self.raw_root = rows, schema, raw_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from .responders import ConceptTrainingExample
        row = self.rows[index]
        labels = {c["concept_id"]: c for c in row["concepts"]}
        targets = tuple(labels[c.id]["value"] if labels[c.id]["annotation_status"] == "OBSERVED"
                        else None for c in self.schema.concepts)
        return ConceptTrainingExample(load_payload(row["input"], self.raw_root), targets,
                                      split="responder_fit")


def train_responder(prepared, out, *, kind="hashing_text", raw_root=None,
                    device="cpu", seed=17, epochs=30, batch_size=32,
                    learning_rate=None, backbone=None, image_size=224, revision=None,
                    allow_download=False, max_length=512, freeze_backbone=False,
                    input_workers=0, input_prefetch_batches=2, concept_class_weighting="none"):
    schema, rows, membership = load_prepared(prepared)
    selected = [r for r in rows if membership[r["sample_id"]]["split"] == "responder_fit"]
    if not selected:
        raise ValueError("no responder_fit rows; do not silently borrow head/policy/validation data")
    return _train_selected_responder(
        prepared, out, schema, selected, kind=kind, raw_root=raw_root, device=device,
        seed=seed, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        backbone=backbone, image_size=image_size, revision=revision,
        allow_download=allow_download, max_length=max_length, freeze_backbone=freeze_backbone,
        input_workers=input_workers, input_prefetch_batches=input_prefetch_batches,
        concept_class_weighting=concept_class_weighting)


def train_crossfit_responder(prepared, planned, out, *, outer_fold=None,
                            inner_fold=None, final=False, **training_options):
    """Fit exactly the verified nested-plan groups, without rewriting membership."""
    from .crossfit import verify_crossfit_prepared
    verified = verify_crossfit_prepared(prepared, planned)
    plan = read_json(Path(planned) / "plan.json")
    unsigned = {key: value for key, value in plan.items() if key != "plan_hash"}
    if plan.get("plan_hash") != verified["plan_hash"] or stable_hash(unsigned) != verified["plan_hash"]:
        raise ValueError("crossfit plan changed after verification")
    if type(final) is not bool:
        raise ValueError("final must be boolean")
    if final:
        if outer_fold is not None or inner_fold is not None:
            raise ValueError("final responder cannot specify outer_fold or inner_fold")
        stage, groups = "final", plan["eligible_group_ids"]
    else:
        if type(outer_fold) is not int or not 0 <= outer_fold < plan["outer_folds"]:
            raise ValueError("outer_fold must identify a planned outer fold")
        fold = plan["folds"][outer_fold]
        stage, groups = "outer", fold["fit_group_ids"]
        if inner_fold is not None:
            if type(inner_fold) is not int or not 0 <= inner_fold < plan["inner_folds"]:
                raise ValueError("inner_fold must identify a planned inner fold")
            stage, groups = "inner", fold["inner_folds"][inner_fold]["fit_group_ids"]
    schema, rows, _ = load_prepared(prepared)
    group_set = set(groups)
    selected = [row for row in rows if row["group_id"] in group_set]
    if not selected or {row["group_id"] for row in selected} != group_set:
        raise ValueError("planned responder fit groups are empty or missing")
    if any(row["split"] != "train" for row in selected):
        raise ValueError("crossfit responder cannot fit held-out data")
    crossfit = {"plan_hash": verified["plan_hash"],
                "plan_sha256": file_hash(Path(planned) / "plan.json"),
                "fold_manifest_hash": verified["fold_manifest_hash"],
                "stage": stage, "outer_fold": outer_fold, "inner_fold": inner_fold,
                "fit_group_ids": sorted(group_set),
                "fit_sample_ids": sorted(row["sample_id"] for row in selected),
                "prepared_files_sha256": plan["receipt"]["prepared_files_sha256"]}
    if "crossfit_receipt" in training_options:
        raise ValueError("crossfit receipt is derived from the verified plan")
    return _train_selected_responder(prepared, out, schema, selected,
                                     crossfit_receipt=crossfit, **training_options)


def _train_selected_responder(prepared, out, schema, selected, *, kind="hashing_text",
                              raw_root=None, device="cpu", seed=17, epochs=30,
                              batch_size=32, learning_rate=None, backbone=None,
                              image_size=224, revision=None, allow_download=False,
                              max_length=512, freeze_backbone=False, crossfit_receipt=None,
                              input_workers=0, input_prefetch_batches=2,
                              concept_class_weighting="none"):
    """Shared backend implementation; public entrypoints own training-set selection."""
    import torch
    from .responders import (fit_text_responder, SharedVisionResponder, fit_responder,
                             save_responder, load_hf_text_responder)
    from .provenance import make_fit_record
    if concept_class_weighting not in ("none", "inverse_frequency"):
        raise ValueError("unsupported concept class weighting")
    if concept_class_weighting != "none" and kind not in ("resnet18", "hf_text"):
        raise ValueError("concept class weighting currently requires resnet18 or hf_text")
    if type(input_workers) is not int or input_workers < 0:
        raise ValueError("input_workers must be a nonnegative integer")
    if type(input_prefetch_batches) is not int or input_prefetch_batches < 1:
        raise ValueError("input_prefetch_batches must be a positive integer")
    if kind == "nano_semantic" and (input_workers or input_prefetch_batches != 2):
        raise ValueError("input prefetch is not implemented for nano_semantic")
    if kind == "hf_text" and (backbone is None or not str(backbone).strip()):
        raise ValueError("hf_text requires an explicit --backbone local directory or Hub model ID")
    if kind != "hf_text" and (revision is not None or allow_download or freeze_backbone or max_length != 512):
        raise ValueError("revision/allow_download/max_length/freeze_backbone options require kind=hf_text")
    if learning_rate is None:
        learning_rate = 3e-5 if kind == "hf_text" else 0.003
    data_audit = read_json(Path(prepared) / "audit.json")
    if kind == "hf_text" and any(row["input"]["modality"] != "text" for row in selected):
        raise ValueError("hf_text requires text-only responder_fit inputs")
    examples = LazyConceptExamples(selected, schema, raw_root)
    out = fresh_dir(out)
    torch.manual_seed(seed)
    if kind == "hashing_text":
        model, report = fit_text_responder(examples, schema, epochs=epochs,
                                          batch_size=batch_size, learning_rate=learning_rate,
                                          seed=seed, device=device, input_workers=input_workers,
                                          input_prefetch_batches=input_prefetch_batches)
        save_responder(model, out / "responder.pt")
    elif kind == "hf_text":
        model = load_hf_text_responder(schema, backbone, revision=revision, allow_download=allow_download,
                                      max_length=max_length, freeze_backbone=freeze_backbone)
        model, report = fit_responder(model, examples, epochs=epochs, batch_size=batch_size,
                                     learning_rate=learning_rate, seed=seed, device=device,
                                     input_workers=input_workers, input_prefetch_batches=input_prefetch_batches,
                                     concept_class_weighting=concept_class_weighting)
        report.update(responder_kind="hf_text", freeze_backbone=freeze_backbone,
                      max_length=max_length, pooling="attention_masked_mean", truncation=False,
                      initialization=model.initialization)
        save_responder(model, out / "responder.pt")
    elif kind == "resnet18":
        model = SharedVisionResponder(schema, image_size=image_size,
                                      backbone_checkpoint=backbone)
        model, report = fit_responder(model, examples, epochs=epochs, batch_size=batch_size,
                                     learning_rate=learning_rate, seed=seed, device=device,
                                     input_workers=input_workers, input_prefetch_batches=input_prefetch_batches,
                                     concept_class_weighting=concept_class_weighting)
        save_responder(model, out / "responder.pt")
    elif kind == "nano_semantic":
        from .nanojev import load_local_nano, fit_nano_semantics, save_nano_head
        if backbone is None:
            raise ValueError("nano_semantic requires --backbone /local/model/directory")
        model = load_local_nano(backbone, device=device)
        model, report = fit_nano_semantics(model, examples, schema, epochs=epochs,
                                          learning_rate=learning_rate, seed=seed)
        save_nano_head(model, out / "responder.pt", schema, task="semantic")
    else:
        raise ValueError("unknown responder kind")
    fit_groups = sorted({r["group_id"] for r in selected})
    artifact_id = "R:" + file_hash(out / "responder.pt")
    provenance = make_fit_record(artifact_id, supervised_group_ids=fit_groups,
                                 metadata={"role": "responder_fit", "kind": kind,
                                           "sample_count": len(selected)})
    receipt = {"kind": kind, "artifact_id": artifact_id, "schema_hash": schema.hash,
               "checkpoint_sha256": file_hash(out / "responder.pt"),
               "supervised_group_ids": fit_groups, "provenance": [provenance],
               "prepared_samples_sha256": file_hash(Path(prepared) / "samples.jsonl"),
               "membership_sha256": file_hash(Path(prepared) / "membership.jsonl"),
               "source_code_hash": code_fingerprint(), "seed": seed,
               "semantic_code_hash": responder_code_fingerprint(),
               "source_revision": data_audit["source_revision"],
               "data_evidence": ("SYNTHETIC_FIXTURE" if data_audit["source_revision"].startswith("SYNTHETIC")
                                 else "USER_SUPPLIED_PUBLIC_DATA_NOT_INDEPENDENTLY_AUTHENTICATED"),
               "batch_independent": True,
               "batch_independence_scope": "deterministic_eval_reference_implementation_requires_backend_profile",
               "dataset_experiment_status": "SINGLE_TRAINING_ARTIFACT_NOT_PAPER_RESULT",
               "external_pretraining_overlap": "UNKNOWN_NOT_CERTIFIED"}
    if kind == "resnet18":
        receipt["initialization"] = {"kind": "local_checkpoint" if backbone else "random",
                     "checkpoint_sha256": file_hash(backbone) if backbone else None}
    elif kind == "hf_text":
        receipt["initialization"] = {**model.initialization, "freeze_backbone": freeze_backbone,
                                     "concept_heads": "RANDOM_INITIALIZATION_THEN_CONCEPT_ONLY_FIT"}
        receipt["shared_compute_mode"] = model.shared_cost_mode
        receipt["checkpoint_scope"] = "FULL_ENCODER_AND_CONCEPT_HEADS_WITH_HASH_BOUND_LOCAL_TOKENIZER_CONFIG"
    if crossfit_receipt is not None:
        if any(file_hash(Path(prepared) / name) != digest
               for name, digest in crossfit_receipt["prepared_files_sha256"].items()):
            raise ValueError("prepared data changed during crossfit responder training")
        receipt["crossfit"] = crossfit_receipt
        receipt["supervised_sample_ids"] = crossfit_receipt["fit_sample_ids"]
        receipt["training_options"] = {"kind": kind, "device": device, "seed": seed,
                "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                "backbone": str(backbone) if backbone is not None else None,
                "image_size": image_size, "revision": revision,
                "allow_download": allow_download, "max_length": max_length,
                "freeze_backbone": freeze_backbone, "input_workers": input_workers,
                "input_prefetch_batches": input_prefetch_batches,
                "concept_class_weighting": concept_class_weighting}
        receipt.setdefault("initialization", {
            "kind": "random" if kind == "hashing_text" else "local_nano_backbone",
            "seed": seed, "backbone": str(backbone) if backbone is not None else None})
        if kind == "nano_semantic":
            receipt["initialization"]["backbone_manifest"] = model.backbone_source_manifest
            receipt["initialization"]["concept_head"] = "RANDOM_INITIALIZATION_THEN_CONCEPT_ONLY_FIT"
    write_json(out / "schema.json", schema.to_dict())
    write_json(out / "receipt.json", receipt)
    write_json(out / "training.json", report)
    write_json(out / "environment.json", environment())
    return {"out": str(out), "responder_kind": kind, "fitted_rows": len(selected),
            "checkpoint_sha256": receipt["checkpoint_sha256"]}


def load_backend(responder_dir, schema, device="cpu"):
    from .responders import load_responder
    responder_dir = Path(responder_dir)
    receipt = read_json(responder_dir / "receipt.json")
    if receipt["schema_hash"] != schema.hash:
        raise ValueError("responder schema differs from prepared dataset")
    if receipt["checkpoint_sha256"] != file_hash(responder_dir / "responder.pt"):
        raise ValueError("responder checkpoint does not match its receipt")
    if receipt.get("semantic_code_hash") != responder_code_fingerprint():
        raise ValueError("responder prompts/preprocessing code changed or lacks an identity receipt; use the recorded code version or regenerate artifacts")
    if receipt["kind"] == "nano_semantic":
        from .nanojev import load_nano_head, NanoSemanticResponder
        scorer = load_nano_head(responder_dir / "responder.pt", schema, device=device,
                                expected_task="semantic")
        model = NanoSemanticResponder(scorer, schema)
    else:
        model = load_responder(responder_dir / "responder.pt", schema, device=device)
    return model, receipt


def cache_responses(prepared, responder_dir, out, *, raw_root=None, device="cpu"):
    """Full offline response tables are permitted for learning, never timing claims."""
    schema, records, membership = load_prepared(prepared)
    responder, receipt = load_backend(responder_dir, schema, device)
    if receipt["membership_sha256"] != file_hash(Path(prepared) / "membership.jsonl"):
        raise ValueError("responder was fitted under a different role manifest")
    if not receipt["batch_independent"]:
        raise ValueError("flat response cache requires batch-independent backend; use action-keyed storage")
    check_rows = [r for r in records if membership[r["sample_id"]]["split"] == "validation"]
    batch_check = verify_batch_invariance(schema, responder, check_rows, raw_root=raw_root)
    if not batch_check["passed"]:
        raise ValueError("batch-dependent answers detected; flat cache is invalid, action-keyed backend required")
    out = fresh_dir(out)
    responses, excluded = [], []
    for row in records:
        role = membership[row["sample_id"]]["split"]
        if role == "responder_fit":
            continue
        if row["target"]["status"] != "OBSERVED":
            excluded.append({"sample_id": row["sample_id"], "reason": "NO_OBSERVED_TASK_TARGET"})
            continue
        payload = load_payload(row["input"], raw_root)
        values = tuple(responder.respond(payload, tuple(range(schema.num_atoms))))
        schema.validate_state(values, complete=True)
        responses.append({"sample_id": row["sample_id"], "group_id": row["group_id"],
                          "split": role, "z": list(values), "y": row["target"]["value"],
                          "response_source": "automatic_model",
                          "input_digest": stable_hash({"text": payload.text,
                              "images": [hashlib.sha256(b).hexdigest() for b in payload.images]})})
    audit = validate_rows(responses, schema, required_roles=("head_fit", "policy_fit", "validation"))
    protected = {r["group_id"] for r in responses}
    if protected.intersection(receipt["supervised_group_ids"]):
        raise ValueError("responder supervision overlaps held-out response groups")
    write_jsonl(out / "responses.jsonl", responses)
    write_json(out / "schema.json", schema.to_dict())
    write_jsonl(out / "exclusions.jsonl", excluded)
    manifest = {"format": "cbmjev-cache-v1", "schema_hash": schema.hash,
                "responses_sha256": file_hash(out / "responses.jsonl"),
                "response_source": "automatic_model", "batch_independent": True,
                "batch_invariance_check": batch_check,
                "responder_checkpoint_sha256": receipt["checkpoint_sha256"],
                "semantic_code_hash": receipt["semantic_code_hash"], "producer_device": device,
                "responder_artifact_id": receipt["artifact_id"],
                "source_revision": receipt.get("source_revision", "UNSPECIFIED_HISTORICAL_ARTIFACT"),
                "data_evidence": receipt.get("data_evidence", "UNSPECIFIED_HISTORICAL_ARTIFACT"),
                "provenance": receipt["provenance"],
                "prepared_samples_sha256": file_hash(Path(prepared) / "samples.jsonl"),
                "membership_sha256": file_hash(Path(prepared) / "membership.jsonl"),
                "audit": audit, "num_excluded_targets": len(excluded),
                "evidence_status": "OFFLINE_RESPONSES_NOT_LATENCY_EVIDENCE"}
    write_json(out / "manifest.json", manifest)
    return {"out": str(out), **audit, "evidence_status": manifest["evidence_status"]}


def verify_batch_invariance(schema, responder, rows, *, raw_root=None, limit=8):
    """Empirical validation guard, not a distribution-wide invariance theorem."""
    if type(limit) is not int or limit < 1 or not rows:
        raise ValueError("batch invariance guard requires validation samples and positive limit")
    selected = sorted(rows, key=lambda r: stable_hash(r["sample_id"]))[:limit]
    mismatches = []
    all_groups = tuple(range(schema.num_groups))
    plans = [(all_groups,), tuple((g,) for g in all_groups),
             tuple((g,) for g in reversed(all_groups)),
             tuple(all_groups[i:i + 2] for i in range(0, len(all_groups), 2))]
    for row in selected:
        payload = load_payload(row["input"], raw_root)
        observed_results = []
        for plan in plans:
            session = responder.start_session(payload) if hasattr(responder, "start_session") else None
            observed = schema.empty_state()
            for action in plan:
                atoms = schema.expand(action)
                values = session.respond(atoms) if session else responder.respond(payload, atoms)
                observed = schema.reveal(observed, action, tuple(values))
            observed_results.append(observed)
        if any(result != observed_results[0] for result in observed_results[1:]):
            mismatches.append(row["sample_id"])
    return {"passed": not mismatches, "sample_ids": [r["sample_id"] for r in selected],
            "num_validation_cases": len(selected), "mismatched_cases": mismatches,
            "plans": ["all", "single_forward", "single_reverse", "pairs"],
            "scope": "observed_validation_cases_only_not_universal_guarantee"}


def load_cache(cache_dir):
    cache_dir = Path(cache_dir)
    schema = Schema.from_dict(read_json(cache_dir / "schema.json"))
    manifest = read_json(cache_dir / "manifest.json")
    if manifest.get("format") != "cbmjev-cache-v1" or manifest["schema_hash"] != schema.hash:
        raise ValueError("cache format/schema mismatch")
    if manifest["responses_sha256"] != file_hash(cache_dir / "responses.jsonl"):
        raise ValueError("cache response content differs from frozen hash")
    if manifest.get("response_source") not in ("automatic_model", "synthetic"):
        raise ValueError("main cache must contain automatic responses, not gold concepts")
    if not manifest.get("batch_independent"):
        raise ValueError("flat replay is invalid for batch-dependent responders")
    rows = read_jsonl(cache_dir / "responses.jsonl")
    validate_rows(rows, schema, required_roles=("head_fit", "policy_fit", "validation"))
    return schema, rows, manifest


def train_models(cache_dir, config, out):
    from .learning import fit_models, save_models
    from .baselines import fit_static_order
    from .provenance import make_fit_record, validate_target_exclusion
    schema, rows, manifest = load_cache(cache_dir)
    config = resolve_config(config)
    if config["policy"]["max_groups"] is not None and config["policy"]["max_groups"] > schema.num_groups:
        raise ValueError("max_groups exceeds schema group count")
    inherited = manifest["provenance"]
    protected = sorted({r["group_id"] for r in rows})
    validate_target_exclusion(inherited, artifact_ids=[manifest["responder_artifact_id"]],
                              target_group_ids=protected)
    out = fresh_dir(out)
    head, controller, report = fit_models(rows, schema, learning_config(config))
    order, static_report = fit_static_order(rows, head, schema)
    save_models(out / "models.pt", head, controller, report)
    head_digest = component_digest(head, schema, "head")
    controller_digest = component_digest(controller, schema, "controller",
                                          objective=controller.objective, pairs=controller.pairs)
    head_id = "f:" + head_digest
    head_fit = sorted({r["group_id"] for r in rows if r["split"] == "head_fit"})
    policy_fit = sorted({r["group_id"] for r in rows if r["split"] == "policy_fit"})
    provenance = inherited + [make_fit_record(head_id, supervised_group_ids=head_fit,
                      parent_ids=[manifest["responder_artifact_id"]], metadata={"role": "head_fit"})]
    validate_target_exclusion(provenance, artifact_ids=[head_id], target_group_ids=policy_fit)
    controller_id = "policy:" + controller_digest
    provenance.append(make_fit_record(controller_id, supervised_group_ids=policy_fit,
                                      parent_ids=[head_id], metadata={"role": "policy_fit"}))
    write_json(out / "static_order.json", {"order": list(order), "report": static_report})
    receipt = {"format": "cbmjev-models-v1", "schema_hash": schema.hash,
               "models_sha256": file_hash(out / "models.pt"),
               "cache_sha256": manifest["responses_sha256"],
               "static_order_sha256": file_hash(out / "static_order.json"),
               "responder_checkpoint_sha256": manifest["responder_checkpoint_sha256"],
               "semantic_code_hash": manifest.get("semantic_code_hash", "UNSPECIFIED_HISTORICAL_ARTIFACT"),
               "source_revision": manifest.get("source_revision", "UNSPECIFIED_HISTORICAL_ARTIFACT"),
               "data_evidence": manifest.get("data_evidence", "UNSPECIFIED_HISTORICAL_ARTIFACT"),
               "head_artifact_id": head_id, "controller_artifact_id": controller_id,
               "head_component_sha256": head_digest, "controller_component_sha256": controller_digest,
               "component_hash_scope": "semantic_schema_and_named_component_state_dict_not_whole_bundle",
               "provenance": provenance, "source_code_hash": code_fingerprint(),
               "seed": config["seed"], "test_evaluated": False,
               "training_protocol": "DISJOINT_ROLES_NO_FINAL_REFIT",
               "evidence_status": "TRAINING_COMPLETE_NOT_PAPER_EVIDENCE"}
    write_json(out / "config.json", config)
    write_json(out / "schema.json", schema.to_dict())
    write_json(out / "receipt.json", receipt)
    write_json(out / "training.json", report)
    write_json(out / "environment.json", environment())
    return {"out": str(out), "checkpoint_sha256": receipt["models_sha256"],
            "training_protocol": receipt["training_protocol"], "test_evaluated": False}


def load_model_bundle(models_dir, cache_dir, device=None):
    from .learning import load_models
    schema, rows, cache_manifest = load_cache(cache_dir)
    models_dir = Path(models_dir)
    config = resolve_config(read_json(models_dir / "config.json"))
    receipt = read_json(models_dir / "receipt.json")
    if receipt["models_sha256"] != file_hash(models_dir / "models.pt"):
        raise ValueError("model weights differ from training receipt")
    if receipt["static_order_sha256"] != file_hash(models_dir / "static_order.json"):
        raise ValueError("static order differs from its frozen training receipt")
    if receipt["schema_hash"] != schema.hash or receipt["cache_sha256"] != cache_manifest["responses_sha256"]:
        raise ValueError("models were trained against a different cache/schema")
    if device is not None:
        config["device"] = device
    head, controller, _ = load_models(models_dir / "models.pt", schema, device=config["device"])
    if component_digest(head, schema, "head") != receipt["head_component_sha256"] or (
            component_digest(controller, schema, "controller", objective=controller.objective,
                             pairs=controller.pairs) != receipt["controller_component_sha256"]):
        raise ValueError("component payload differs from its supervision-scoped artifact identity")
    order = tuple(read_json(models_dir / "static_order.json")["order"])
    return schema, rows, cache_manifest, config, receipt, head, controller, order


def system_hash(receipt, schema, config, *, method, mode="offline_replay", nano_hash=None,
                matched_hash=None, brig_hash=None, nano_identity=None, static_mask_hash=None):
    identity = {"models_sha256": receipt["models_sha256"], "schema_hash": schema.hash,
                        "cache_sha256": receipt["cache_sha256"],
                        "static_order_sha256": receipt["static_order_sha256"],
                        "responder_sha256": receipt["responder_checkpoint_sha256"],
                        "semantic_code_hash": receipt.get("semantic_code_hash"),
                        "source_code_hash": code_fingerprint(), "config": config,
                        "method": method, "mode": mode, "nano_hash": nano_hash,
                        "failure_handling": "fail_run_no_silent_imputation_or_case_drop"}
    if method == "nano_static_risk" and (nano_hash is None or nano_identity is None):
        raise ValueError("Nano static risk identity requires head and training receipt hashes")
    if nano_identity is not None:
        identity["nano_identity"] = nano_identity
    if method == "matched_mlp_risk":
        if matched_hash is None:
            raise ValueError("matched MLP system identity requires paired artifact hashes")
        identity["matched_mlp_identity"] = matched_hash
    if method == "brig":
        if brig_hash is None:
            raise ValueError("BRiG system identity requires controller artifact hashes")
        identity["brig_identity"] = brig_hash
    if method == "learned_static_mask":
        if static_mask_hash is None:
            raise ValueError("learned static mask system identity requires artifact hashes")
        identity["static_mask_identity"] = static_mask_hash
    return stable_hash(identity)


def freeze_family(models_dir, cache_dir, out, *, weights=None, methods=None,
                  mode="offline_replay", nano_dir=None):
    from .evaluation import freeze_policy_manifest
    schema, _, _, config, receipt, _, _, _ = load_model_bundle(models_dir, cache_dir)
    weights = [config["policy"]["cost_weight"]] if weights is None else list(weights)
    methods = config["evaluation"]["methods"] if methods is None else list(methods)
    if set(methods) & {"matched_mlp_risk", "brig", "nano_static_risk", "learned_static_mask"}:
        raise ValueError("matched_mlp_risk/brig/nano_static_risk/learned_static_mask are validation-only and cannot freeze a formal family")
    if not weights or any(not math.isfinite(v) or v < 0 for v in weights):
        raise ValueError("finite nonnegative cost weights required")
    if len(set(weights)) != len(weights) or len(set(methods)) != len(methods):
        raise ValueError("frozen methods and cost weights must be unique")
    nano_hash = file_hash(Path(nano_dir) / "nano_head.pt") if nano_dir else None
    policies, settings = [], {}
    for method in methods:
        candidates = weights if method in ("risk", "value", "static_value", "value_singleton", "lookahead", "nano_risk") else weights[:1]
        for weight in candidates:
            instance = resolve_config(config)
            instance["policy"]["cost_weight"] = weight
            policy_id = "{}@lambda={:.8g}".format(method, weight)
            digest = system_hash(receipt, schema, instance, method=method, mode=mode,
                                 nano_hash=nano_hash if method == "nano_risk" else None)
            policies.append({"policy_id": policy_id, "system_hash": digest})
            settings[policy_id] = {"method": method, "config": instance, "mode": mode,
                                   "nano_hash": nano_hash if method == "nano_risk" else None}
    manifest = freeze_policy_manifest(policies)
    out = fresh_dir(out)
    write_json(out / "manifest.json", manifest)
    write_json(out / "settings.json", settings)
    return {"out": str(out), "manifest_hash": manifest["manifest_hash"],
            "num_policies": manifest["num_policies"],
            "notice": "retain manifest hash before calibration; timestamps do not prove chronology"}


def train_nano_controller(cache_dir, models_dir, backbone, out, *, device="cpu",
                          epochs=5, batch_size=16, learning_rate=0.001,
                          max_length=2048, max_examples=50000, matched_mlp=False,
                          cache_nano_features=False, nano_feature_batch_size=None,
                          feature_normalization="none"):
    import torch
    from .learning import iter_risk_training_examples
    from .nanojev import (load_local_nano, fit_nano_risk, save_nano_head,
                          RiskTrainingExample)
    from .matched_risk import matched_event_manifest, fit_matched_mlp_risk
    from .provenance import validate_target_exclusion
    for name, value in (("epochs", epochs), ("batch_size", batch_size),
                        ("max_examples", max_examples), ("max_length", max_length)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")
    if type(matched_mlp) is not bool:
        raise ValueError("matched_mlp must be boolean")
    if feature_normalization not in ("none", "layernorm"):
        raise ValueError("feature_normalization must be none or layernorm")
    if (type(learning_rate) not in (int, float) or not math.isfinite(learning_rate)
            or learning_rate <= 0):
        raise ValueError("learning_rate must be finite and positive")
    source_before = code_fingerprint()
    def input_hashes():
        return {role + "/" + p.name: file_hash(p)
                for role, directory in (("models", models_dir), ("cache", cache_dir))
                for p in sorted(Path(directory).iterdir()) if p.is_file()}
    inputs_before = input_hashes()
    schema, rows, manifest, config, receipt, head, _, _ = load_model_bundle(models_dir, cache_dir, device)
    if max_examples < 1:
        raise ValueError("max_examples must be positive")
    validate_target_exclusion(receipt["provenance"], artifact_ids=[receipt["head_artifact_id"]],
                              target_group_ids=sorted({r["group_id"] for r in rows
                                                       if r["split"] == "policy_fit"}))
    # Deterministic reservoir sampling avoids retaining a giant prompt/target table.
    rng = random.Random(config["seed"])
    examples, seen = [], 0
    for seen, item in enumerate(iter_risk_training_examples(rows, head, schema, learning_config(config)), 1):
        example = RiskTrainingExample(tuple(item["observed"]), tuple(item["action"]),
                                      float(item["error"]), split="policy_fit")
        if len(examples) < max_examples:
            examples.append(example)
        else:
            index = rng.randrange(seen)
            if index < max_examples:
                examples[index] = example
    examples = tuple(examples)
    events = matched_event_manifest(examples, schema)
    torch.manual_seed(config["seed"])
    scorer = load_local_nano(backbone, device=device, max_length=max_length,
                             feature_normalization=feature_normalization)
    parameter_count = sum(p.numel() for p in scorer.parameters())
    trainable_count = sum(p.numel() for p in scorer.parameters() if p.requires_grad)
    out = fresh_dir(out)
    scorer, report = fit_nano_risk(scorer, examples, schema, epochs=epochs,
                                  batch_size=batch_size, learning_rate=learning_rate,
                                  seed=config["seed"], cache_features=cache_nano_features,
                                  feature_batch_size=nano_feature_batch_size)
    save_nano_head(scorer, out / "nano_head.pt", schema, task="risk")
    matched_report = None
    if matched_mlp:
        controller, matched_report = fit_matched_mlp_risk(
            examples, schema, epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, seed=config["seed"],
            hidden=config["learning"]["hidden"], device=device)
        # Tensors only: metadata and architecture live in the hash-bound report.
        torch.save({k: v.detach().cpu() for k, v in controller.network.state_dict().items()},
                   out / "matched_mlp.pt")
        matched_report.update({"checkpoint_sha256": file_hash(out / "matched_mlp.pt"),
                               "parent_models_sha256": receipt["models_sha256"],
                               "cache_sha256": manifest["responses_sha256"],
                               "source_code_hash": source_before})
        write_json(out / "matched_mlp_training.json", matched_report)
    if matched_event_manifest(examples, schema) != events:
        raise ValueError("shared risk events changed during paired training")
    if code_fingerprint() != source_before or input_hashes() != inputs_before:
        raise ValueError("training source or parent/cache artifacts changed during Nano fitting")
    write_json(out / "environment.json", environment())
    # Existing evaluator consumes training.json; publish it only after every check.
    write_json(out / "training.json", {**report, **events, "available_examples": seen,
                "retained_examples": len(examples), "sampling": "seeded_reservoir",
                "parent_models_sha256": receipt["models_sha256"],
                "cache_sha256": manifest["responses_sha256"], "schema_hash": schema.hash,
                "nano_sha256": file_hash(out / "nano_head.pt"),
                "pairs": config["learning"].get("pairs"), "source_code_hash": source_before,
                "input_file_sha256": inputs_before,
                "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                "feature_normalization": feature_normalization,
                "event_exposures": len(examples) * epochs,
                "optimizer_steps": math.ceil(len(examples) / batch_size) * epochs,
                "parameter_count": parameter_count,
                "trainable_parameter_count_during_fit": trainable_count,
                "head_excludes_policy_fit_groups_verified": True,
                "target_generator": "iter_risk_training_examples; frozen parent head; epoch zero; policy_fit only",
                "aggregation": "unweighted mean of minibatch mean event BCE",
                "matched_mlp": matched_mlp,
                "matched_report_sha256": file_hash(out / "matched_mlp_training.json") if matched_mlp else None,
                "matched_checkpoint_sha256": file_hash(out / "matched_mlp.pt") if matched_mlp else None,
                "initialization_comparison": "same seed; different architectures and initialization distributions",
                "comparison_scope": "frozen local LM scalar risk adapter versus MLP, not official NanoJev reproduction",
                "real_backbone_status": "LOCAL_BACKBONE_EXECUTED",
                "calibrated": False, "new_algorithm_claim": False})
    return {"out": str(out), "examples": len(examples), "objective": "independent_error_BCE",
            "backbone_frozen": True, "calibrated": False, "matched_mlp": matched_mlp,
            "event_sha256": events["event_sha256"]}


def load_matched_mlp_controller(nano_dir, models_dir, cache_dir, *, device="cpu"):
    """Read a paired diagnostic controller; does not change the evaluator protocol."""
    import torch
    from .learning import ActionController
    nano_dir = Path(nano_dir)
    schema, _, manifest, _, parent, _, _, _ = load_model_bundle(models_dir, cache_dir, device)
    receipt = read_json(nano_dir / "training.json")
    report = read_json(nano_dir / "matched_mlp_training.json")
    if (receipt.get("matched_mlp") is not True
            or receipt["matched_report_sha256"] != file_hash(nano_dir / "matched_mlp_training.json")
            or receipt["matched_checkpoint_sha256"] != file_hash(nano_dir / "matched_mlp.pt")
            or receipt["nano_sha256"] != file_hash(nano_dir / "nano_head.pt")):
        raise ValueError("paired risk artifacts differ from completion receipt")
    for field, expected in (("schema_hash", schema.hash),
                            ("parent_models_sha256", parent["models_sha256"]),
                            ("cache_sha256", manifest["responses_sha256"]),
                            ("event_sha256", receipt["event_sha256"]),
                            ("source_code_hash", receipt["source_code_hash"])):
        if report.get(field) != expected or receipt.get(field) != expected:
            raise ValueError("paired risk ancestry mismatch: " + field)
    if report["checkpoint_sha256"] != receipt["matched_checkpoint_sha256"] or report["config"]["objective"] != "risk":
        raise ValueError("invalid matched risk checkpoint report")
    controller = ActionController(schema, {**report["config"], "device": device})
    controller.network.load_state_dict(torch.load(nano_dir / "matched_mlp.pt", map_location=device,
                                                  weights_only=True))
    controller.network.eval().requires_grad_(False)
    return controller, report


def train_brig_controller(cache_dir, models_dir, out, **kwargs):
    from .brig_artifacts import train_brig_controller as train
    return train(cache_dir, models_dir, out, **kwargs)


def train_static_mask(cache_dir, models_dir, out, **kwargs):
    from .static_mask_artifacts import train_static_mask as train
    return train(cache_dir, models_dir, out, **kwargs)


def evaluate_models(models_dir, cache_dir, out, *, split="validation", evaluate_test=False,
                    frozen_family=None, certification_run=False, methods=None,
                    cost_weight=None, max_groups=None, device=None, prepared=None, responder_dir=None,
                    raw_root=None, nano_dir=None, warmup=0, brig_dir=None, static_mask_dir=None):
    from .baselines import EmpiricalLookaheadPolicy
    from .evaluation import summarize_traces, paired_group_bootstrap
    from .provenance import validate_target_exclusion
    if split not in ("validation", "calibration", "test"):
        raise ValueError("evaluation split must be validation/calibration/test")
    if split == "test" and not evaluate_test:
        raise ValueError("test evaluation requires --evaluate-test after protocol freeze")
    if split == "calibration" and (not frozen_family or not certification_run):
        raise ValueError("calibration requires a pre-frozen family and --certification-run")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    schema, rows, cache_manifest, config, receipt, head, learned, order = load_model_bundle(
        models_dir, cache_dir, device)
    requested = config["evaluation"]["methods"] if methods is None else methods
    if frozen_family:
        frozen_specs = read_json(Path(frozen_family) / "settings.json")
        requested = [s["method"] for s in frozen_specs.values()]
    if "matched_mlp_risk" in requested and (split != "validation" or frozen_family or certification_run):
        raise ValueError("matched_mlp_risk is validation-only, not a frozen/calibration/test policy")
    if "nano_static_risk" in requested and (split != "validation" or frozen_family or certification_run or evaluate_test):
        raise ValueError("nano_static_risk is validation-only, not a frozen/calibration/test policy")
    if "brig" in requested and (split != "validation" or frozen_family or certification_run or evaluate_test):
        raise ValueError("brig is validation-only, not a frozen/calibration/test policy")
    if "learned_static_mask" in requested and (split != "validation" or frozen_family or certification_run or evaluate_test):
        raise ValueError("learned_static_mask is validation-only, not a frozen/calibration/test policy")
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError("no samples in requested split")
    validate_target_exclusion(receipt["provenance"], artifact_ids=[receipt["controller_artifact_id"]],
                              target_group_ids=sorted({r["group_id"] for r in selected}))
    mode = "live" if responder_dir is not None else "offline_replay"
    evaluation_source_before = code_fingerprint()
    nano_hash = None
    nano_identity = None
    nano = None
    matched, matched_hash = None, None
    brig, brig_hash = None, None
    static_mask, static_mask_hash = None, None
    if "learned_static_mask" in requested:
        if not static_mask_dir:
            raise ValueError("learned_static_mask requires --static-mask")
        from .static_mask_artifacts import load_static_mask, static_mask_identity
        static_mask, _ = load_static_mask(static_mask_dir, models_dir, cache_dir, device=config["device"])
        static_mask_hash = static_mask_identity(static_mask_dir)
        budget = max_groups if max_groups is not None else config["policy"]["max_groups"]
        if budget is not None and static_mask.k > budget:
            raise ValueError("learned static mask K exceeds evaluation budget")
    if "brig" in requested:
        if not brig_dir:
            raise ValueError("brig requires --brig-controller")
        from .brig_artifacts import load_brig_controller, artifact_identity
        brig, _ = load_brig_controller(brig_dir, models_dir, cache_dir, device=config["device"])
        brig_hash = artifact_identity(brig_dir)
        budget = max_groups if max_groups is not None else config["policy"]["max_groups"]
        budget = schema.num_groups if budget is None else budget
        if budget and budget not in brig.models:
            raise ValueError("BRiG requested budget was not trained")
        if config["policy"]["max_cost"] is not None:
            raise ValueError("BRiG supports fixed group budgets only")
    if "matched_mlp_risk" in requested:
        if not nano_dir:
            raise ValueError("matched_mlp_risk requires paired --nano-controller directory")
        matched, _ = load_matched_mlp_controller(nano_dir, models_dir, cache_dir, device=config["device"])
        matched_hash = {name: file_hash(Path(nano_dir) / name) for name in
                        ("matched_mlp.pt", "matched_mlp_training.json", "training.json")}
    if nano_dir:
        from .nanojev import load_nano_head, NanoRiskController
        nano_dir = Path(nano_dir)
        nano_receipt = read_json(nano_dir / "training.json")
        nano_hash = file_hash(nano_dir / "nano_head.pt")
        if nano_receipt["nano_sha256"] != nano_hash or nano_receipt["parent_models_sha256"] != receipt["models_sha256"]:
            raise ValueError("Nano controller checkpoint or parent task head changed")
        if nano_receipt["cache_sha256"] != cache_manifest["responses_sha256"]:
            raise ValueError("Nano controller cache ancestry mismatch")
        if set(requested) & {"nano_risk", "nano_static_risk"}:
            nano_identity = {name: file_hash(nano_dir / name) for name in ("nano_head.pt", "training.json")}
            scorer = load_nano_head(nano_dir / "nano_head.pt", schema, device=config["device"], expected_task="risk")
            nano = NanoRiskController(scorer, schema)
    manifest_hash = None
    if frozen_family:
        if methods is not None or cost_weight is not None or max_groups is not None:
            raise ValueError("cannot override methods/thresholds/budgets of a frozen family")
        frozen_family = Path(frozen_family)
        manifest = read_json(frozen_family / "manifest.json")
        settings = read_json(frozen_family / "settings.json")
        from .evaluation import freeze_policy_manifest
        if freeze_policy_manifest(manifest["policies"], frozen_at=manifest["frozen_at"]) != manifest:
            raise ValueError("invalid frozen family manifest")
        expected = {p["policy_id"]: p["system_hash"] for p in manifest["policies"]}
        if set(settings) != set(expected):
            raise ValueError("frozen candidate settings do not cover exactly the registered family")
        for policy_id, item in settings.items():
            item_config = resolve_config(item["config"])
            if mode != item["mode"] or config["device"] != item_config["device"]:
                raise ValueError("frozen deployment mode/device cannot be changed silently")
            actual_nano = nano_hash if item["method"] == "nano_risk" else None
            if system_hash(receipt, schema, item_config, method=item["method"], mode=mode,
                           nano_hash=actual_nano) != expected[policy_id]:
                raise ValueError("complete frozen system changed: " + policy_id)
        manifest_hash = manifest["manifest_hash"]
    else:
        if cost_weight is not None:
            config["policy"]["cost_weight"] = cost_weight
        if max_groups is not None:
            config["policy"]["max_groups"] = max_groups
        if cost_weight is not None or max_groups is not None:
            config = resolve_config(config)
        choices = config["evaluation"]["methods"] if methods is None else methods
        if not choices or len(set(choices)) != len(choices):
            raise ValueError("evaluation methods must be nonempty and unique")
        settings = {m: {"method": m, "config": config, "mode": mode} for m in choices}
        expected = {m: system_hash(receipt, schema, config, method=m, mode=mode,
                                   nano_hash=nano_hash if m in ("nano_risk", "nano_static_risk") else None,
                                   nano_identity=nano_identity if m == "nano_static_risk" else None,
                                   matched_hash=matched_hash if m == "matched_mlp_risk" else None,
                                   brig_hash=brig_hash if m == "brig" else None,
                                   static_mask_hash=static_mask_hash if m == "learned_static_mask" else None) for m in choices}
        if "learned_static_mask" in settings:
            settings["learned_static_mask"]["static_mask_identity"] = static_mask_hash
            settings["learned_static_mask"]["cost_weight_ignored_fixed_subset"] = True
        if "brig" in settings:
            settings["brig"]["brig_identity"] = brig_hash
            settings["brig"]["cost_weight_ignored_fixed_budget"] = True
        if "matched_mlp_risk" in settings:
            settings["matched_mlp_risk"]["matched_mlp_identity"] = matched_hash
        for m in ("nano_risk", "nano_static_risk"):
            if m in settings:
                settings[m]["nano_identity"] = nano_identity
    responder, raw_records = None, None
    if mode == "live":
        if prepared is None:
            raise ValueError("live evaluation requires prepared canonical records")
        actual_schema, canonical, _ = load_prepared(prepared)
        if actual_schema != schema or file_hash(Path(prepared) / "samples.jsonl") != cache_manifest["prepared_samples_sha256"]:
            raise ValueError("live canonical records differ from the frozen cache source")
        responder, responder_receipt = load_backend(responder_dir, schema, config["device"])
        if responder_receipt["checkpoint_sha256"] != receipt["responder_checkpoint_sha256"]:
            raise ValueError("live responder is not the one used to train the task head/controller")
        raw_records = {r["sample_id"]: r["input"] for r in canonical}
    elif warmup:
        raise ValueError("warmup is only meaningful for live evaluation")
    warm_rows = [r for r in rows if r["split"] == "validation"][:warmup] if mode == "live" else []
    warmup_ids = [r["sample_id"] for r in warm_rows]
    selected_ids = {r["sample_id"] for r in selected}
    out = fresh_dir(out)
    traces, reports = [], {}
    for policy_id, spec in settings.items():
        method, active = spec["method"], spec["config"]
        controller = learned
        if method == "lookahead":
            controller = EmpiricalLookaheadPolicy.fit(
                rows, head, schema, depth=active["evaluation"]["lookahead_depth"],
                include_pairs=active["learning"]["include_pairs"],
                include_all=active["learning"]["include_all"], pairs=learned.pairs)
        elif method in ("nano_risk", "nano_static_risk"):
            if nano is None:
                raise ValueError(method + " requires a fitted --nano-controller directory")
            controller = nano
        elif method == "matched_mlp_risk":
            controller = matched
        elif method == "brig":
            controller = brig
        elif method == "learned_static_mask":
            controller = static_mask
        arguments = {"method": method, "controller": controller,
                     "cost": DeclaredCost(**active["cost"]), **active["policy"],
                     "include_pairs": active["learning"]["include_pairs"],
                     "include_all": active["learning"]["include_all"], "pairs": learned.pairs,
                     "order": order if method in ("static", "static_value", "nano_static_risk") else None,
                     "seed": active["seed"], "device": active["device"]}
        if mode == "live" and warmup:
            # Warmup only on validation inputs, even for a final test run.
            for row in warm_rows:
                env = LiveEnvironment(raw_records[row["sample_id"]], schema, responder,
                                      raw_root, active["device"], expected_digest=row.get("input_digest"),
                                      expected_responses=row["z"])
                run_episode(env, schema, head, **arguments)
        current = []
        for row in selected:
            if mode == "live":
                env = LiveEnvironment(raw_records[row["sample_id"]], schema, responder,
                                      raw_root, active["device"], expected_digest=row.get("input_digest"),
                                      expected_responses=row["z"])
            else:
                env = ReplayEnvironment(row["z"], schema)
            trace = run_episode(env, schema, head, **arguments)
            trace.update(sample_id=row["sample_id"], group_id=row["group_id"], split=split, y=row["y"],
                         policy_id=policy_id, system_hash=expected[policy_id], seed=active["seed"],
                         num_query_groups=schema.num_groups, num_atoms=schema.num_atoms)
            current.append(trace)
        reports[policy_id] = summarize_traces(current, num_classes=schema.num_classes)
        traces.extend(current)
    paired = {}
    all_ids = [p for p, s in settings.items() if s["method"] == "all"]
    if all_ids and split != "calibration":
        reference = [r for r in traces if r["policy_id"] == all_ids[0]]
        if any(r["calls"] != 1 or r["queried_groups"] != list(range(schema.num_groups)) for r in reference):
            raise ValueError("paired all-at-once reference must query all groups in exactly one call")
        for policy_id in settings:
            if policy_id == all_ids[0]:
                continue
            current = [r for r in traces if r["policy_id"] == policy_id]
            paired[policy_id] = paired_group_bootstrap(current, reference, seed=config["seed"],
                                     n_resamples=config["evaluation"]["bootstrap_resamples"])
    if nano_identity is not None and (code_fingerprint() != evaluation_source_before or
            nano_identity != {name: file_hash(Path(nano_dir) / name) for name in nano_identity}):
        raise ValueError("Nano diagnostic source/checkpoint changed during evaluation")
    if matched_hash is not None and (code_fingerprint() != evaluation_source_before or
            matched_hash != {name: file_hash(Path(nano_dir) / name) for name in matched_hash}):
        raise ValueError("paired diagnostic source/checkpoint changed during evaluation")
    if brig_hash is not None and (code_fingerprint() != evaluation_source_before or
                                 artifact_identity(brig_dir) != brig_hash):
        raise ValueError("BRiG source/checkpoint changed during evaluation")
    if static_mask_hash is not None and (code_fingerprint() != evaluation_source_before or
                                        static_mask_identity(static_mask_dir) != static_mask_hash):
        raise ValueError("static-mask source/checkpoint changed during evaluation")
    write_jsonl(out / "traces.jsonl", traces)
    write_json(out / "metrics.json", {"split": split, "mode": mode, "seed": config["seed"],
              "manifest_hash": manifest_hash, "policies": reports,
              "paired_against_all": paired, "warmup_validation_cases_per_policy": len(warm_rows),
              "warmup_requested_cases_per_policy": warmup, "warmup_sample_ids": warmup_ids,
              "warmup_overlap_sample_ids": [sid for sid in warmup_ids if sid in selected_ids],
              "warmup_scope": "fresh episodes, same loaded models; validation overlap explicitly reported",
              "source_code_hash": code_fingerprint(),
              "models_receipt_sha256": file_hash(Path(models_dir) / "receipt.json"),
              "schema_hash": receipt["schema_hash"], "models_sha256": receipt["models_sha256"],
              "cache_sha256": receipt["cache_sha256"],
              "static_order_sha256": receipt["static_order_sha256"],
              "responder_checkpoint_sha256": receipt["responder_checkpoint_sha256"],
              "head_component_sha256": receipt["head_component_sha256"],
              "controller_component_sha256": receipt["controller_component_sha256"],
              "matched_mlp_identity": matched_hash,
              "brig_identity": brig_hash,
              "static_mask_identity": static_mask_hash,
              "nano_controller_sha256": nano_hash,
              "nano_identity": nano_identity,
              "training_source_code_hash": receipt["source_code_hash"],
              "paper_evidence": False,
              "source_revision": receipt.get("source_revision", "UNSPECIFIED_HISTORICAL_ARTIFACT"),
              "data_evidence": receipt.get("data_evidence", "UNSPECIFIED_HISTORICAL_ARTIFACT"),
              "evidence_status": "LIVE_SINGLE_RUN_REQUIRES_REPLICATION" if mode == "live" else "OFFLINE_REPLAY_NOT_LATENCY"})
    write_json(out / "settings.json", settings)
    write_json(out / "environment.json", environment())
    return {"out": str(out), "split": split, "mode": mode, "num_policies": len(settings),
            "samples_per_policy": len(selected), "manifest_hash": manifest_hash}


def export_summary(run_dirs, out):
    """Export per-run facts; no pooling of seed variation with sample-level CIs."""
    import csv
    out = fresh_dir(out)
    table = []
    fields = ["run", "split", "mode", "seed", "policy_id", "accuracy", "macro_f1",
              "num_samples", "num_groups", "group_mean_risk", "mean_queried_groups",
              "mean_queried_atoms", "mean_calls", "mean_declared_cost",
              "live_p50_ms", "live_p95_ms", "data_evidence", "evidence_status"]
    for directory in run_dirs:
        source = Path(directory) / "metrics.json"
        report = read_json(source)
        for policy_id, metrics in report["policies"].items():
            row = {key: metrics[key] for key in fields if key in metrics}
            latency = metrics.get("deployment_latency")
            row.update(run=str(Path(directory)), split=report["split"], mode=report["mode"],
                       seed=report["seed"], policy_id=policy_id,
                       live_p50_ms=latency["p50"] if latency else "",
                       live_p95_ms=latency["p95"] if latency else "",
                       data_evidence=report.get("data_evidence", "UNSPECIFIED_HISTORICAL_RUN"),
                       evidence_status=report["evidence_status"])
            table.append(row)
    with (out / "metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(table)
    write_json(out / "table.json", {"rows": table,
                 "aggregation": "one row per run and policy; no implicit pooling",
                 "source_metrics": {str(Path(d) / "metrics.json"): file_hash(Path(d) / "metrics.json")
                                    for d in run_dirs}})
    return {"out": str(out), "rows": len(table), "csv": str(out / "metrics.csv")}


def synthetic_cebab_source(out, seed=17, train_size=100, dev_size=30, test_size=20):
    """A generated official-format fixture, explicitly NOT real CEBaB data."""
    out = fresh_dir(out)
    rng = random.Random(seed)
    labels = ("Negative", "Positive", "unknown")
    words = ("poor", "excellent", "unmentioned")
    aspects = ("food", "noise", "ambiance", "service")
    for split, count in (("train_exclusive", train_size), ("dev", dev_size), ("test", test_size)):
        records = []
        for index in range(count):
            values = [rng.randrange(3) for _ in aspects]
            sid = "synthetic-{}-{:04d}".format(split, index)
            # Independent random neutral suffix prevents accidental duplicate
            # fixture texts; it never enters the concept-only policy/head.
            text = "; ".join("{} is {}".format(a, words[v]) for a, v in zip(aspects, values))
            text += ". Visit note: {}.".format(rng.getrandbits(64))
            record = {"id": sid, "original_id": sid, "edit_id": "0", "is_original": True,
                      "description": text, "review_majority": str(1 + (values[0] + 2 * values[3]) % 5)}
            record.update({a + "_aspect_majority": labels[v] for a, v in zip(aspects, values)})
            records.append(record)
        write_json(out / (split + ".json"), records)
    return out


def smoke(out, *, seed=17):
    """Exercise actual code paths using synthetic text, never publishable metrics."""
    import torch
    from .data import prepare_dataset
    from .evaluation import certify_policies
    torch.set_num_threads(1)
    out = fresh_dir(out)
    source = synthetic_cebab_source(out / "synthetic_raw", seed=seed)
    prepare_dataset("cebab", source, out / "prepared", seed=17,
                    source_revision="SYNTHETIC_FIXTURE_NOT_CEBAB")
    train_responder(out / "prepared", out / "responder", seed=seed, epochs=3,
                    batch_size=16, learning_rate=0.02)
    cache_responses(out / "prepared", out / "responder", out / "cache")
    config = resolve_config({"seed": seed, "learning": {"hidden": 32, "head_epochs": 3,
                 "policy_epochs": 3, "batch_size": 32, "masks_per_sample": 3},
                 "evaluation": {"methods": ["stop", "all", "fixed", "random", "static", "risk", "lookahead"],
                                "bootstrap_resamples": 50}})
    train_models(out / "cache", config, out / "models")
    evaluate_models(out / "models", out / "cache", out / "validation_replay")
    evaluate_models(out / "models", out / "cache", out / "validation_live",
                    methods=["all", "fixed", "risk"], prepared=out / "prepared",
                    responder_dir=out / "responder", warmup=1)
    frozen = freeze_family(out / "models", out / "cache", out / "frozen",
                           weights=[0.0, 0.03], methods=["all", "risk"])
    evaluate_models(out / "models", out / "cache", out / "calibration",
                    split="calibration", frozen_family=out / "frozen", certification_run=True)
    certificate = certify_policies(read_jsonl(out / "calibration" / "traces.jsonl"),
                         read_json(out / "frozen" / "manifest.json"), alpha=0.1, delta=0.05,
                         expected_manifest_hash=frozen["manifest_hash"])
    write_json(out / "certification.json", certificate)
    # Test metrics are not evaluated. Data preparation/cache construction does
    # inspect test records, but never fits or selects models using their labels.
    export_summary([out / "validation_replay", out / "validation_live"], out / "tables")
    result = {"status": "PASS", "kind": "SYNTHETIC_SOFTWARE_SMOKE_NOT_PAPER_EVIDENCE",
              "out": str(out), "seed": seed, "test_evaluated": False,
              "stages": ["prepare", "semantic_fit", "automatic_cache", "head_fit", "policy_fit",
                         "replay", "live_cpu", "freeze", "calibration", "certification", "csv_export"],
              "certification_status": certificate["status"],
              "not_validated": ["real_dataset_performance", "CUDA", "real_HF_backbone",
                                "published_AFA_reproduction", "clinical_reliability"]}
    write_json(out / "smoke_report.json", result)
    return result

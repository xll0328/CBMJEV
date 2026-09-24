"""Plan-bound nested OOF response tables; never latency measurements.

Integrity checks certify declared target-task ancestry, not unknown pretraining
overlap or the authenticity of a maliciously rewritten training history.
"""
import hashlib
from pathlib import Path

from .contracts import stable_hash
from .crossfit import verify_crossfit_prepared
from .io import file_hash, fresh_dir, read_json, read_jsonl, write_json, write_jsonl
from .pipeline import (load_backend, load_prepared, responder_code_fingerprint,
                       verify_batch_invariance)
from .provenance import validate_provenance_dag, validate_target_exclusion
from .runtime import load_payload
from .semantic_source import verify_responder_source


def _context(prepared, planned, responder_dir, split, *, responder_source_dir=None):
    prepared, planned, responder_dir = map(Path, (prepared, planned, responder_dir))
    verified = verify_crossfit_prepared(prepared, planned)
    plan = read_json(planned / "plan.json")
    schema, records, _ = load_prepared(prepared)
    receipt = read_json(responder_dir / "receipt.json")
    cf = receipt.get("crossfit", {})
    stage, outer, inner = (cf.get(k) for k in ("stage", "outer_fold", "inner_fold"))
    # Held-out data may never enter a prediction ancestor, even when caching
    # train OOF rows rather than the held-out split itself.
    protected = [g for groups in plan["excluded_group_ids_by_outer_split"].values()
                 for g in groups]
    if stage == "final":
        if outer is not None or inner is not None or split not in ("validation", "calibration"):
            raise ValueError("final cache requires explicit validation or calibration split; test is closed")
        fit = plan["eligible_group_ids"]
        targets = plan["excluded_group_ids_by_outer_split"][split]
    else:
        if split is not None or type(outer) is not int or not 0 <= outer < plan["outer_folds"]:
            raise ValueError("invalid outer cache selection")
        fold = plan["folds"][outer]
        protected += fold["target_group_ids"]
        if stage == "outer" and inner is None:
            fit, targets = fold["fit_group_ids"], fold["target_group_ids"]
        elif stage == "inner" and type(inner) is int and 0 <= inner < plan["inner_folds"]:
            fold = fold["inner_folds"][inner]
            fit, targets = fold["fit_group_ids"], fold["target_group_ids"]
        else:
            raise ValueError("invalid inner cache selection")
    selected = sorted((r for r in records if r["group_id"] in set(targets)),
                      key=lambda r: r["sample_id"])
    if not selected or set(targets) != {r["group_id"] for r in selected}:
        raise ValueError("cache requires nonempty complete planned target groups")
    fit_ids = sorted(r["sample_id"] for r in records if r["group_id"] in set(fit))
    source_binding = verify_responder_source(receipt, responder_source_dir=responder_source_dir)
    semantic_code_hash = (source_binding["semantic_code_hash"] if source_binding is not None
                          else responder_code_fingerprint())
    expected_cf = {"plan_hash": verified["plan_hash"],
                   "plan_sha256": file_hash(planned / "plan.json"),
                   "fold_manifest_hash": verified["fold_manifest_hash"],
                   "stage": stage, "outer_fold": outer, "inner_fold": inner,
                   "fit_group_ids": fit, "fit_sample_ids": fit_ids,
                   "prepared_files_sha256": plan["receipt"]["prepared_files_sha256"]}
    if cf != expected_cf:
        raise ValueError("responder crossfit receipt differs from verified plan")
    checkpoint = file_hash(responder_dir / "responder.pt")
    expected = {"supervised_group_ids": fit, "supervised_sample_ids": fit_ids,
                "schema_hash": schema.hash, "checkpoint_sha256": checkpoint,
                "artifact_id": "R:" + checkpoint, "batch_independent": True,
                "semantic_code_hash": semantic_code_hash,
                "prepared_samples_sha256": file_hash(prepared / "samples.jsonl"),
                "membership_sha256": file_hash(prepared / "membership.jsonl")}
    for key, value in expected.items():
        if receipt.get(key) != value or (key == "batch_independent" and receipt[key] is not True):
            raise ValueError("responder receipt mismatch: " + key)
    if read_json(responder_dir / "schema.json") != schema.to_dict():
        raise ValueError("responder schema file mismatch")
    provenance = receipt.get("provenance", [])
    validate_provenance_dag(provenance)
    producer = next((r for r in provenance if r["artifact_id"] == receipt["artifact_id"]), None)
    if producer is None or producer["fit_kind"] != "supervised" or producer["supervised_group_ids"] != fit:
        raise ValueError("producer provenance does not declare exact planned fit groups")
    ancestry = validate_target_exclusion(provenance, artifact_ids=[receipt["artifact_id"]],
                                        target_group_ids=sorted(set(targets + protected)))
    base = {"format": "cbmjev-crossfit-cache-v1", "status": "COMPLETE",
            "evidence_status": "OFFLINE_RESPONSES_NOT_LATENCY_EVIDENCE",
            "crossfit": expected_cf, "split": split, "schema_hash": schema.hash,
            "plan_sha256": file_hash(planned / "plan.json"),
            "fold_manifest_sha256": file_hash(planned / "fold_manifest.jsonl"),
            "prepared_files_sha256": plan["receipt"]["prepared_files_sha256"],
            "responder_checkpoint_sha256": checkpoint,
            "responder_receipt_sha256": file_hash(responder_dir / "receipt.json"),
            "responder_schema_sha256": file_hash(responder_dir / "schema.json"),
            "responder_artifact_id": receipt["artifact_id"], "provenance": provenance,
            "target_group_ids": targets, "target_sample_ids": [r["sample_id"] for r in selected],
            "ancestry_exclusion": ancestry, "batch_independent": True,
            "response_source": "automatic_model"}
    return schema, selected, base


def _observed(row, schema):
    target = row["target"]
    if target["status"] != "OBSERVED":
        return False
    if type(target["value"]) is not int or not 0 <= target["value"] < schema.num_classes:
        raise ValueError("invalid observed target")
    return True


def _row_identity(row, base):
    return {"sample_id": row["sample_id"], "group_id": row["group_id"],
            "split": base["split"] or "train", "y": row["target"]["value"],
            "response_source": "automatic_model",
            "responder_artifact_id": base["responder_artifact_id"],
            "input_record_hash": stable_hash(row["input"])}


def cache_crossfit_responses(prepared, planned, responder_dir, out, *, raw_root=None,
                            device="cpu", split=None):
    """Cache exactly the responder's planned OOF targets (or final held-out split)."""
    schema, selected, base = _context(prepared, planned, responder_dir, split)
    responder, _ = load_backend(responder_dir, schema, device)
    batch = verify_batch_invariance(schema, responder, selected, raw_root=raw_root)
    if batch.get("passed") is not True:
        raise ValueError("batch-dependent backend cannot produce flat OOF cache")
    out = fresh_dir(out)
    responses, excluded = [], []
    for row in selected:
        if not _observed(row, schema):
            excluded.append({"sample_id": row["sample_id"], "group_id": row["group_id"],
                             "reason": "NO_OBSERVED_TASK_TARGET"})
            continue
        payload = load_payload(row["input"], raw_root)
        values = tuple(responder.respond(payload, tuple(range(schema.num_atoms))))
        if any(type(value) is not int for value in values):
            raise ValueError("backend categories must be hard integer values")
        schema.validate_state(values, complete=True)
        responses.append({**_row_identity(row, base), "z": list(values),
                          "input_digest": stable_hash({"text": payload.text,
                              "images": [hashlib.sha256(b).hexdigest() for b in payload.images]})})
    # Detect source changes during inference before committing a completed artifact.
    if _context(prepared, planned, responder_dir, split)[2] != base:
        raise ValueError("sources changed while caching")
    write_json(out / "schema.json", schema.to_dict())
    write_jsonl(out / "responses.jsonl", responses)
    write_jsonl(out / "exclusions.jsonl", excluded)
    manifest = {**base, "batch_invariance_check": batch, "producer_device": device,
                "response_artifact_by_group": {g: base["responder_artifact_id"]
                    for g in sorted({r["group_id"] for r in responses})},
                "num_responses": len(responses), "num_excluded_targets": len(excluded),
                "files_sha256": {name: file_hash(out / name) for name in
                                 ("schema.json", "responses.jsonl", "exclusions.jsonl")}}
    manifest["manifest_hash"] = stable_hash(manifest)
    write_json(out / "manifest.json", manifest)  # Completion marker is always last.
    return {"out": str(out), "num_responses": len(responses),
            "num_excluded_targets": len(excluded), "manifest_hash": manifest["manifest_hash"]}


def load_crossfit_cache(cache_dir, prepared, planned, responder_dir, *,
                        responder_source_dir=None):
    """Revalidate source identity, ancestry, full coverage, and label/category integrity."""
    cache_dir = Path(cache_dir)
    manifest = read_json(cache_dir / "manifest.json")
    unsigned = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if manifest.get("manifest_hash") != stable_hash(unsigned):
        raise ValueError("cache manifest hash mismatch")
    schema, selected, base = _context(prepared, planned, responder_dir, manifest.get("split"),
                                      responder_source_dir=responder_source_dir)
    if any(manifest.get(k) != v for k, v in base.items()):
        raise ValueError("cache source binding differs from verified inputs")
    expected_hashes = {name: file_hash(cache_dir / name) for name in
                       ("schema.json", "responses.jsonl", "exclusions.jsonl")}
    if manifest.get("files_sha256") != expected_hashes:
        raise ValueError("cache file hash mismatch")
    if read_json(cache_dir / "schema.json") != schema.to_dict():
        raise ValueError("cache schema mismatch")
    rows, exclusions = read_jsonl(cache_dir / "responses.jsonl"), read_jsonl(cache_dir / "exclusions.jsonl")
    expected_rows = [r for r in selected if _observed(r, schema)]
    expected_exclusions = [{"sample_id": r["sample_id"], "group_id": r["group_id"],
                            "reason": "NO_OBSERVED_TASK_TARGET"}
                           for r in selected if not _observed(r, schema)]
    if len(rows) != len(expected_rows) or exclusions != expected_exclusions:
        raise ValueError("cache target coverage or exclusions mismatch")
    for row, source in zip(rows, expected_rows):
        if any(row.get(k) != v for k, v in _row_identity(source, base).items()):
            raise ValueError("cached response identity/label/source mismatch")
        z = row.get("z")
        if not isinstance(z, list) or any(type(v) is not int for v in z):
            raise ValueError("cached categories must be hard integer values")
        schema.validate_state(tuple(z), complete=True)
        digest = row.get("input_digest")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("missing input content digest")
    assignments = {g: base["responder_artifact_id"] for g in sorted({r["group_id"] for r in rows})}
    if (manifest.get("response_artifact_by_group") != assignments
            or manifest.get("num_responses") != len(rows)
            or manifest.get("num_excluded_targets") != len(exclusions)):
        raise ValueError("cache source assignments/counts mismatch")
    batch = manifest.get("batch_invariance_check", {})
    expected_check_ids = [r["sample_id"] for r in sorted(selected, key=lambda r: stable_hash(r["sample_id"]))[:8]]
    if (batch.get("passed") is not True or batch.get("mismatched_cases") != []
            or batch.get("sample_ids") != expected_check_ids
            or batch.get("num_validation_cases") != len(expected_check_ids)
            or batch.get("plans") != ["all", "single_forward", "single_reverse", "pairs"]):
        raise ValueError("cache batch invariance evidence missing or inconsistent")
    return schema, rows, manifest


def crossfit_cache_source_binding(cache_dir, responder_dir, responder_source_dir):
    """Return explicit historical-source binding metadata for new consumers."""
    binding = verify_responder_source(read_json(Path(responder_dir) / "receipt.json"),
                                      responder_source_dir=responder_source_dir)
    if binding is None:
        return None
    manifest = read_json(Path(cache_dir) / "manifest.json")
    return {**binding,
            "responder_receipt_sha256": file_hash(Path(responder_dir) / "receipt.json"),
            "cache_manifest_sha256": file_hash(Path(cache_dir) / "manifest.json"),
            "cache_manifest_hash": manifest.get("manifest_hash")}

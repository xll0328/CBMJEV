"""Declared-supervision provenance, with actual group IDs and immutable hashes."""

import hashlib
import json
import random
from collections.abc import Mapping


def canonical_hash(value):
    """SHA256 of deterministic JSON; reject nonfinite/unserializable metadata."""
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Provenance must be finite JSON-serializable data.") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a nonempty string.".format(name))
    return value


def _ids(values, name, allow_empty=True):
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, set)):
        raise ValueError("{} must be a collection of string IDs.".format(name))
    result = [_identifier(item, name) for item in values]
    if len(set(result)) != len(result):
        raise ValueError("{} contains duplicate IDs.".format(name))
    if not allow_empty and not result:
        raise ValueError("{} cannot be empty.".format(name))
    return sorted(result)


def make_fit_record(artifact_id, *, supervised_group_ids, parent_ids=(), fit_kind="supervised", metadata=None):
    """Record completed artifact lineage, not evidence that fitting took place.

    Callers must populate actual target-task supervision groups. External
    pretraining membership is unknown and is not certified by this ledger.
    """
    artifact_id = _identifier(artifact_id, "artifact_id")
    kinds = {"supervised", "derived", "frozen_external", "target_construction"}
    if fit_kind not in kinds:
        raise ValueError("Unknown fit_kind; planned jobs are not fitted artifacts.")
    groups = _ids(supervised_group_ids, "supervised_group_ids")
    parents = _ids(parent_ids, "parent_ids")
    if artifact_id in parents:
        raise ValueError("An artifact cannot be its own parent.")
    if fit_kind == "frozen_external" and (groups or parents):
        raise ValueError("frozen_external has no declared target-task supervision or parents.")
    if fit_kind == "derived" and groups:
        raise ValueError("A derived record with direct labels must use a supervised fit_kind.")
    if fit_kind in {"supervised", "target_construction"} and not groups:
        raise ValueError("A supervised record must list actual supervised groups.")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a JSON object.")
    record = {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "fit_kind": fit_kind,
        "supervised_group_ids": groups,
        "supervised_group_ids_hash": canonical_hash(groups),
        "supervised_group_count": len(groups),
        "parent_ids": parents,
        "metadata": dict(metadata or {}),
        "scope": "declared_target_task_supervision_only",
    }
    record["record_hash"] = canonical_hash(record)
    return record


def _record_map(records):
    if isinstance(records, Mapping) or isinstance(records, (str, bytes)):
        raise ValueError("records must be an iterable of record objects, not an ID mapping.")
    mapping = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Each provenance record must be an object.")
        artifact_id = _identifier(record.get("artifact_id"), "artifact_id")
        if artifact_id in mapping:
            raise ValueError("Duplicate artifact_id: {}".format(artifact_id))
        reconstructed = make_fit_record(
            artifact_id,
            supervised_group_ids=record.get("supervised_group_ids"),
            parent_ids=record.get("parent_ids", ()),
            fit_kind=record.get("fit_kind"),
            metadata=record.get("metadata"),
        )
        if dict(record) != reconstructed:
            raise ValueError("Record hash, group list, count, or schema mismatch: {}".format(artifact_id))
        mapping[artifact_id] = dict(record)
    if not mapping:
        raise ValueError("At least one provenance record is required.")
    return mapping


def _topological_order(mapping):
    visiting, complete, order = set(), set(), []

    def visit(artifact_id):
        if artifact_id in complete:
            return
        if artifact_id in visiting:
            raise ValueError("Provenance cycle through {}".format(artifact_id))
        if artifact_id not in mapping:
            raise ValueError("Unknown provenance parent: {}".format(artifact_id))
        visiting.add(artifact_id)
        for parent in mapping[artifact_id]["parent_ids"]:
            visit(parent)
        visiting.remove(artifact_id)
        complete.add(artifact_id)
        order.append(artifact_id)

    for artifact_id in sorted(mapping):
        visit(artifact_id)
    return order


def validate_provenance_dag(records):
    """Check content hashes, real ID lists, references, and absence of cycles."""
    mapping = _record_map(records)
    return {
        "status": "VALID",
        "num_artifacts": len(mapping),
        "topological_order": _topological_order(mapping),
        "scope": "declared_target_task_supervision_only",
        "external_pretraining_membership_verified": False,
    }


def validate_target_exclusion(records, *, artifact_ids, target_group_ids):
    """Reject target supervision in any declared ancestor, including roots."""
    mapping = _record_map(records)
    _topological_order(mapping)
    roots = _ids(artifact_ids, "artifact_ids", allow_empty=False)
    target_groups = _ids(target_group_ids, "target_group_ids", allow_empty=False)
    visited = set()

    def visit(artifact_id):
        if artifact_id not in mapping:
            raise ValueError("Unknown prediction artifact: {}".format(artifact_id))
        if artifact_id in visited:
            return
        visited.add(artifact_id)
        leaked = set(target_groups).intersection(mapping[artifact_id]["supervised_group_ids"])
        if leaked:
            raise ValueError(
                "Target-group supervision leakage at {}: {}".format(artifact_id, sorted(leaked))
            )
        for parent in mapping[artifact_id]["parent_ids"]:
            visit(parent)

    for artifact_id in roots:
        visit(artifact_id)
    declared_supervision = sorted(
        {group for artifact_id in visited for group in mapping[artifact_id]["supervised_group_ids"]}
    )
    return {
        "status": "PASS",
        "artifact_ids": roots,
        "checked_ancestor_ids": sorted(visited),
        "target_group_ids": target_groups,
        "target_group_ids_hash": canonical_hash(target_groups),
        "ancestor_supervised_group_ids": declared_supervision,
        "ancestor_supervised_group_ids_hash": canonical_hash(declared_supervision),
        "scope": "declared_target_task_supervision_only",
        "external_pretraining_membership_verified": False,
        "group_disjointness_implies_iid": False,
    }


def plan_nested_crossfit(group_ids, *, outer_folds=3, inner_folds=3, seed):
    """Generate a group-disjoint job plan; does not train or assert completed OOF."""
    groups = _ids(group_ids, "group_ids", allow_empty=False)
    for value, name in ((outer_folds, "outer_folds"), (inner_folds, "inner_folds")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError("{} must be an integer >= 2.".format(name))
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an explicit integer.")
    if len(groups) < outer_folds:
        raise ValueError("Not enough groups for nonempty outer folds.")
    shuffled = groups[:]
    random.Random(seed).shuffle(shuffled)
    outer = [sorted(shuffled[index::outer_folds]) for index in range(outer_folds)]
    jobs, outer_target_ids, outer_responder_ids = [], [], []

    def add_job(job_id, kind, supervision, targets=(), parents=(), **extra):
        record = {
            "job_id": job_id,
            "kind": kind,
            "status": "PLANNED",
            "supervised_group_ids": sorted(supervision),
            "supervised_group_ids_hash": canonical_hash(sorted(supervision)),
            "target_group_ids": sorted(targets),
            "target_group_ids_hash": canonical_hash(sorted(targets)),
            "parent_job_ids": list(parents),
        }
        record.update(extra)
        jobs.append(record)

    for outer_index, held in enumerate(outer):
        training = sorted(set(groups).difference(held))
        if len(training) < inner_folds:
            raise ValueError("Not enough outer-training groups for nonempty inner folds.")
        inner_shuffled = training[:]
        random.Random(seed + 1009 * (outer_index + 1)).shuffle(inner_shuffled)
        inner = [sorted(inner_shuffled[index::inner_folds]) for index in range(inner_folds)]
        inner_ids = []
        for inner_index, inner_held in enumerate(inner):
            job_id = "responder.outer{}.inner{}".format(outer_index, inner_index)
            add_job(
                job_id, "fit_responder",
                sorted(set(training).difference(inner_held)), inner_held,
                outer_fold=outer_index, inner_fold=inner_index,
            )
            inner_ids.append(job_id)
        responder_id = "responder.outer{}".format(outer_index)
        add_job(responder_id, "fit_responder", training, held, outer_fold=outer_index)
        outer_responder_ids.append(responder_id)
        head_id = "head.outer{}".format(outer_index)
        add_job(head_id, "fit_head_on_inner_oof", training, held, inner_ids, outer_fold=outer_index)
        target_id = "policy_targets.outer{}".format(outer_index)
        add_job(
            target_id, "construct_policy_targets", held, held, (responder_id, head_id),
            outer_fold=outer_index,
            prediction_ancestor_job_ids=[responder_id, head_id],
            target_labels_used_only_after_prediction=True,
        )
        outer_target_ids.append(target_id)
    add_job("responder.final", "fit_responder", groups)
    add_job("head.final", "fit_head_on_outer_oof", groups, parents=outer_responder_ids)
    add_job("policy.final", "fit_policy_on_oof_targets", groups, parents=outer_target_ids)
    result = {
        "schema_version": 1,
        "status": "PLANNED",
        "has_trained_models": False,
        "has_validated_actual_oof_predictions": False,
        "seed": seed,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "group_ids": groups,
        "group_ids_hash": canonical_hash(groups),
        "jobs": jobs,
        "fit_job_count": sum(job["kind"].startswith("fit_") for job in jobs),
        "responder_fit_job_count": sum(job["kind"] == "fit_responder" for job in jobs),
        "iid_assumption_verified": False,
    }
    result["plan_hash"] = canonical_hash(result)
    return result

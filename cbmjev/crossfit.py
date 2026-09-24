"""Prepared-data anchored nested cross-fitting plans.

The outer fold is immutable prepared data: this module validates and consumes
``samples.jsonl[*].fold_id``.  It never shuffles or silently creates a new
outer partition.  Inner folds are recomputed inside each outer-training set
with a deterministic, stratum-aware hash rule.

This module only emits a plan.  It does not train models or certify that later
predictions are genuinely out of fold.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .contracts import ROLES, stable_hash
from .io import file_hash, fresh_dir, read_json, read_jsonl, write_json, write_jsonl


_FIT_ROLES = frozenset(("responder_fit", "head_fit", "policy_fit"))
_HELDOUT_SPLITS = ("validation", "calibration", "test")
_PROTOCOL = "cbmjev-prepared-nested-crossfit-v1"


def _require_identifier(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a nonempty string".format(field))
    return value


def _require_int(value, field, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(field, minimum))
    return value


def _inner_sort_key(seed, dataset, outer_fold, group_id):
    """Prepared-data protocol hash order; deliberately contains no RNG."""
    purpose = "crossfit-inner-v1-outer{}".format(outer_fold)
    payload = "acam-split-v1|{}|{}|{}|{}".format(seed, purpose, dataset, group_id)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), group_id


def _group_stratum(rows, dataset):
    """Versioned copy of the prepared-data group stratum contract.

    Keeping this rule in the crossfit protocol prevents an unrelated private
    helper refactor in the data adapter from silently changing stored plans.
    """
    for row in rows:
        target = row.get("target")
        if not isinstance(target, dict) or "value" not in target:
            raise ValueError("crossfit stratum requires an explicit target.value")
    if dataset == "cebab":
        for row in rows:
            audit = row.get("audit_metadata")
            if not isinstance(audit, dict) or type(audit.get("is_original")) is not bool:
                raise ValueError("CEBaB crossfit stratum requires boolean is_original")
        originals = [row for row in rows if row["audit_metadata"]["is_original"]]
        if not originals:
            return "missing-original"
        rows = originals
    targets = {row["target"]["value"] for row in rows}
    if len(targets) > 1:
        return "mixed"
    if not targets:
        raise ValueError("cannot stratify an empty group")
    value = next(iter(targets))
    return "missing-target" if value is None else "class:" + str(value)


def _canonical_manifest_sha256(rows):
    """Byte identity of ``io.write_jsonl`` for the ordered manifest rows."""
    digest = hashlib.sha256()
    for row in rows:
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        digest.update(encoded.encode("utf-8"))
    return digest.hexdigest()


def _load_prepared(prepared):
    prepared = Path(prepared).resolve()
    required = {
        "samples.jsonl": prepared / "samples.jsonl",
        "schema.json": prepared / "schema.json",
        "membership.jsonl": prepared / "membership.jsonl",
        "audit.json": prepared / "audit.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("prepared data missing required files: " + ", ".join(missing))
    schema = read_json(required["schema.json"])
    audit = read_json(required["audit.json"])
    samples = read_jsonl(required["samples.jsonl"])
    members = read_jsonl(required["membership.jsonl"])
    dataset = _require_identifier(schema.get("dataset"), "schema.dataset")
    if audit.get("dataset") != dataset:
        raise ValueError("audit dataset does not match schema dataset")
    seed = _require_int(audit.get("seed"), "audit.seed")
    declared_outer_folds = _require_int(audit.get("fold_count"), "audit.fold_count", 2)
    if audit.get("schema_hash") != stable_hash(schema):
        raise ValueError("schema content does not match audit.schema_hash")
    if audit.get("split_hash") != stable_hash(members):
        raise ValueError("membership content does not match audit.split_hash")
    file_hashes = {name: file_hash(path) for name, path in sorted(required.items())}
    return prepared, schema, audit, samples, members, dataset, seed, declared_outer_folds, file_hashes


def _validate_groups(samples, members, dataset, declared_outer_folds):
    if not samples:
        raise ValueError("prepared samples cannot be empty")
    sample_map = {}
    for row in samples:
        sid = _require_identifier(row.get("sample_id"), "sample_id")
        if sid in sample_map:
            raise ValueError("duplicate sample_id in samples.jsonl: " + sid)
        if row.get("dataset") != dataset:
            raise ValueError("sample dataset does not match schema: " + sid)
        _require_identifier(row.get("group_id"), "group_id")
        sample_map[sid] = row

    member_map = {}
    for row in members:
        sid = _require_identifier(row.get("sample_id"), "membership.sample_id")
        if sid in member_map:
            raise ValueError("duplicate sample_id in membership.jsonl: " + sid)
        if row.get("split") not in ROLES:
            raise ValueError("invalid membership role for " + sid)
        if row.get("outer_split") not in ("train",) + _HELDOUT_SPLITS:
            raise ValueError("invalid membership outer_split for " + sid)
        member_map[sid] = row
    if set(sample_map) != set(member_map):
        raise ValueError("samples and membership sample IDs do not align")

    grouped = defaultdict(list)
    group_contract = {}
    for sid, sample in sample_map.items():
        member = member_map[sid]
        if member.get("group_id") != sample["group_id"]:
            raise ValueError("membership group mismatch for " + sid)
        outer_split = member["outer_split"]
        if sample.get("split") != outer_split:
            raise ValueError("sample split does not match membership outer_split for " + sid)
        role = member["split"]
        if outer_split == "train":
            if role not in _FIT_ROLES:
                raise ValueError("train sample has a held-out role: " + sid)
            fold_id = sample.get("fold_id")
            _require_int(fold_id, "train fold_id")
        else:
            if role != outer_split:
                raise ValueError("held-out membership role must equal outer_split: " + sid)
            if sample.get("fold_id") is not None:
                raise ValueError("held-out sample must not have fold_id: " + sid)
            fold_id = None
        contract = (outer_split, role, fold_id)
        group_id = sample["group_id"]
        if group_id in group_contract and group_contract[group_id] != contract:
            previous = group_contract[group_id]
            if previous[:2] != contract[:2]:
                raise ValueError("group crosses prepared split/role: " + group_id)
            raise ValueError("group crosses prepared fold_id: " + group_id)
        group_contract[group_id] = contract
        grouped[group_id].append(sample)

    training = sorted(g for g, value in group_contract.items() if value[0] == "train")
    if not training:
        raise ValueError("prepared data contains no train groups")
    observed_folds = sorted({group_contract[g][2] for g in training})
    expected_folds = list(range(declared_outer_folds))
    if observed_folds != expected_folds:
        raise ValueError("prepared train fold IDs must be contiguous and match audit.fold_count")

    strata = {}
    for group_id, rows in grouped.items():
        if group_contract[group_id][0] == "train":
            strata[group_id] = _group_stratum(rows, dataset)
    return grouped, group_contract, training, strata


def _assign_inner(training_groups, strata, *, dataset, seed, outer_fold, inner_folds):
    by_stratum = defaultdict(list)
    for group_id in training_groups:
        by_stratum[strata[group_id]].append(group_id)
    assignments = {}
    for stratum in sorted(by_stratum):
        ordered = sorted(by_stratum[stratum], key=lambda group_id: _inner_sort_key(
            seed, dataset, outer_fold, group_id))
        for rank, group_id in enumerate(ordered):
            assignments[group_id] = rank % inner_folds
    counts = {fold: sum(value == fold for value in assignments.values())
              for fold in range(inner_folds)}
    if any(count == 0 for count in counts.values()):
        raise ValueError(
            "outer fold {} cannot form {} nonempty stratified inner folds".format(
                outer_fold, inner_folds))
    return assignments


def _build(prepared, inner_folds):
    inner_folds = _require_int(inner_folds, "inner_folds", 2)
    (prepared, schema, audit, samples, members, dataset, seed, outer_folds,
     file_hashes) = _load_prepared(prepared)
    grouped, contracts, training, strata = _validate_groups(
        samples, members, dataset, outer_folds)
    training_set = set(training)

    assignments_by_outer = {}
    outer_plans = []
    for outer_fold in range(outer_folds):
        target = sorted(g for g in training if contracts[g][2] == outer_fold)
        fit = sorted(training_set.difference(target))
        if not target:
            raise ValueError("prepared outer fold {} is empty".format(outer_fold))
        if len(fit) < inner_folds:
            raise ValueError("outer-training set is smaller than inner_folds")
        inner_assignment = _assign_inner(
            fit, strata, dataset=dataset, seed=seed,
            outer_fold=outer_fold, inner_folds=inner_folds)
        assignments_by_outer[outer_fold] = inner_assignment
        inner_plans = []
        for inner_fold in range(inner_folds):
            inner_target = sorted(g for g, fold in inner_assignment.items()
                                  if fold == inner_fold)
            inner_fit = sorted(set(fit).difference(inner_target))
            inner_plans.append({
                "inner_fold_id": inner_fold,
                "fit_group_ids": inner_fit,
                "fit_group_ids_hash": stable_hash(inner_fit),
                "target_group_ids": inner_target,
                "target_group_ids_hash": stable_hash(inner_target),
            })
        outer_plans.append({
            "outer_fold_id": outer_fold,
            "fit_group_ids": fit,
            "fit_group_ids_hash": stable_hash(fit),
            "target_group_ids": target,
            "target_group_ids_hash": stable_hash(target),
            "inner_folds": inner_plans,
        })

    excluded = {split: sorted(g for g, value in contracts.items() if value[0] == split)
                for split in _HELDOUT_SPLITS}
    excluded_all = sorted(g for groups in excluded.values() for g in groups)
    manifest = []
    for group_id in sorted(grouped):
        outer_split, role, outer_fold = contracts[group_id]
        eligible = outer_split == "train"
        inner = ({str(fold): assignments_by_outer[fold][group_id]
                  for fold in range(outer_folds) if group_id in assignments_by_outer[fold]}
                 if eligible else {})
        manifest.append({
            "schema_version": "cbmjev-crossfit-fold-manifest-v1",
            "dataset": dataset,
            "group_id": group_id,
            "sample_ids": sorted(row["sample_id"] for row in grouped[group_id]),
            "sample_count": len(grouped[group_id]),
            "outer_split": outer_split,
            "prepared_role": role,
            "eligible_for_crossfit": eligible,
            "exclusion_reason": None if eligible else "heldout_outer_split:" + outer_split,
            "outer_fold_id": outer_fold,
            "stratum": strata.get(group_id),
            "inner_fold_by_outer_fold": inner,
        })

    manifest_hash = stable_hash(manifest)
    manifest_sha256 = _canonical_manifest_sha256(manifest)
    prepared_hash = stable_hash(file_hashes)
    receipt = {
        "protocol": _PROTOCOL,
        "prepared_files_sha256": file_hashes,
        "prepared_hash": prepared_hash,
        "prepared_files_hash": prepared_hash,
        "samples_sha256": file_hashes["samples.jsonl"],
        "schema_sha256": file_hashes["schema.json"],
        "schema_hash": stable_hash(schema),
        "membership_sha256": file_hashes["membership.jsonl"],
        "membership_hash": stable_hash(members),
        "fold_hash": manifest_hash,
        "fold_manifest_hash": manifest_hash,
        "fold_manifest_sha256": manifest_sha256,
        "group_ids_hash": stable_hash(training),
        "eligible_group_ids_hash": stable_hash(training),
        "eligible_group_count": len(training),
        "excluded_group_ids_hash": stable_hash(excluded_all),
        "excluded_group_count": len(excluded_all),
        "excluded_group_ids_hash_by_split": {
            split: stable_hash(excluded[split]) for split in _HELDOUT_SPLITS},
    }
    plan = {
        "schema_version": "cbmjev-prepared-crossfit-plan-v1",
        "status": "PLANNED_NOT_TRAINED",
        "has_trained_models": False,
        "has_validated_actual_oof_predictions": False,
        "dataset": dataset,
        "prepared_seed": seed,
        "outer_fold_source": "frozen_samples_jsonl_fold_id",
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "inner_assignment_rule": (
            "acam-split-v1 stratum-aware SHA256 order inside each outer-training set; "
            "rank modulo inner_folds"),
        "eligible_outer_splits": ["train"],
        "excluded_outer_splits": list(_HELDOUT_SPLITS),
        "eligible_group_ids": training,
        "excluded_group_ids_by_outer_split": excluded,
        "folds": outer_plans,
        "receipt": receipt,
        "limitations": [
            "This artifact is a deterministic job plan, not evidence of completed training.",
            "Actual OOF predictions and fitted-artifact supervision ancestry require separate audit.",
        ],
    }
    plan["plan_hash"] = stable_hash(plan)
    return manifest, plan


def plan_crossfit_prepared(prepared, out, *, inner_folds=3):
    """Validate prepared folds and write a deterministic nested OOF plan."""
    manifest, plan = _build(prepared, inner_folds)
    out = fresh_dir(out)
    manifest_path = out / "fold_manifest.jsonl"
    plan_path = out / "plan.json"
    write_jsonl(manifest_path, manifest)
    if plan["receipt"]["fold_manifest_sha256"] != file_hash(manifest_path):
        raise AssertionError("canonical fold manifest writer changed unexpectedly")
    write_json(plan_path, plan)
    return {
        "status": plan["status"],
        "fold_manifest": manifest_path,
        "plan": plan_path,
        "plan_hash": plan["plan_hash"],
        "fold_manifest_hash": plan["receipt"]["fold_manifest_hash"],
    }


def verify_crossfit_prepared(prepared, planned):
    """Verify inputs, hashes, membership sets, and exact deterministic rebuild."""
    planned = Path(planned).resolve()
    manifest_path, plan_path = planned / "fold_manifest.jsonl", planned / "plan.json"
    if not manifest_path.is_file() or not plan_path.is_file():
        raise FileNotFoundError("planned directory requires fold_manifest.jsonl and plan.json")
    actual_manifest, actual_plan = read_jsonl(manifest_path), read_json(plan_path)
    claimed_plan_hash = actual_plan.get("plan_hash")
    unsigned = dict(actual_plan)
    unsigned.pop("plan_hash", None)
    if claimed_plan_hash != stable_hash(unsigned):
        raise ValueError("plan_hash mismatch")
    receipt = actual_plan.get("receipt")
    if not isinstance(receipt, dict):
        raise ValueError("plan receipt is missing")
    if receipt.get("fold_manifest_sha256") != file_hash(manifest_path):
        raise ValueError("fold_manifest byte hash mismatch")
    if receipt.get("fold_manifest_hash") != stable_hash(actual_manifest):
        raise ValueError("fold_manifest content hash mismatch")

    expected_manifest, expected_plan = _build(prepared, actual_plan.get("inner_folds"))
    expected_manifest_sha256 = _canonical_manifest_sha256(expected_manifest)
    if actual_manifest != expected_manifest:
        raise ValueError("fold_manifest differs from deterministic prepared-data rebuild")
    if file_hash(manifest_path) != expected_manifest_sha256:
        raise ValueError("fold_manifest bytes are not the canonical deterministic encoding")
    if actual_plan != expected_plan:
        raise ValueError("plan differs from deterministic prepared-data rebuild")
    return {
        "status": "PASS",
        "plan_hash": claimed_plan_hash,
        "fold_manifest_hash": receipt["fold_manifest_hash"],
        "eligible_group_ids_hash": receipt["eligible_group_ids_hash"],
        "excluded_group_ids_hash": receipt["excluded_group_ids_hash"],
    }

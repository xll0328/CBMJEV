"""Role-filtered joins for private, offline verification experiments.

These loaders are not policy inputs. They keep answer/label access in the
offline evaluator and refuse protected populations unless explicitly enabled.
Never publish the returned per-example records.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import Schema
from .io import file_hash, read_json, read_jsonl


PROTECTED_ROLES = frozenset(("test", "confirmation", "locked_confirmation", "locked-confirmation"))
SOURCE_ROLES = frozenset(("responder_fit", "source_fit", "source-fit"))


@dataclass
class VerificationInputs:
    schema: Schema
    rows: dict
    report: dict

    def data(self, role):
        from .verification_experiment import VerificationData
        rows = self.rows[role]
        return VerificationData(
            sample_ids=tuple(row["sample_id"] for row in rows),
            group_ids=tuple(row["group_id"] for row in rows),
            A=np.asarray([row["A"] for row in rows], dtype=np.int64),
            B=np.asarray([row["B"] for row in rows], dtype=np.int64),
            C=np.asarray([row["C"] for row in rows], dtype=object),
            Y=np.asarray([row["Y"] for row in rows], dtype=np.int64),
            categories=self.schema.value_counts,
        )

    def shared_concepts(self, role):
        return np.asarray([row["shared"] for row in self.rows[role]], dtype=np.int64)


def _hard_vector(values, counts, name):
    if not isinstance(values, (list, tuple)) or len(values) != len(counts):
        raise ValueError(name + " has incorrect concept width")
    if any(type(value) is not int or not 0 <= value < size
           for value, size in zip(values, counts)):
        raise ValueError(name + " must use zero-based native semantic categories")
    return tuple(values)


def _predictions(path, roles):
    selected, seen = {}, set()
    for row in read_jsonl(path):
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("invalid or duplicate prediction sample_id")
        seen.add(sid)
        if row.get("split") in roles:
            selected[sid] = row
    return selected


def load_verification_inputs(prepared, a_predictions, b_predictions, *, roles,
                             a_metadata=None, b_metadata=None, legacy_a=False,
                             allow_protected=False):
    """Load only declared task-observed populations, validating each join.

    Old A caches contain task labels; these are ignored, not used as ground
    truth. Actual Y/C always come from the prepared data joined by membership.
    Missing task labels are counted, never mapped to a class. Missing concept
    labels remain None, distinct from a native 'unknown' category.
    """
    roles = tuple(roles)
    if not roles or len(set(roles)) != len(roles):
        raise ValueError("roles must be nonempty and distinct")
    if set(roles) & SOURCE_ROLES:
        raise ValueError("source supervision is not a task-policy evaluation role")
    if set(roles) & PROTECTED_ROLES and not allow_protected:
        raise ValueError("protected roles require a separately frozen confirmation protocol")
    prepared = Path(prepared)
    schema = Schema.from_dict(read_json(prepared / "schema.json"))
    membership = read_jsonl(prepared / "membership.jsonl")
    members, group_roles = {}, {}
    for row in membership:
        sid, gid, role = row.get("sample_id"), row.get("group_id"), row.get("split")
        if not all(isinstance(v, str) and v for v in (sid, gid, role)) or sid in members:
            raise ValueError("invalid or duplicate membership")
        if gid in group_roles and group_roles[gid] != role:
            raise ValueError("a group crosses data roles")
        members[sid], group_roles[gid] = row, role
    if set(roles) - set(group_roles.values()):
        raise ValueError("requested role absent from membership")
    membership_hash = file_hash(prepared / "membership.jsonl")
    a_meta = read_json(a_metadata) if a_metadata else None
    b_meta = read_json(b_metadata) if b_metadata else None
    if legacy_a:
        if a_meta is None or a_meta.get("format") != "cbmjev-cache-v1":
            raise ValueError("legacy A requires its original cache manifest")
        if a_meta.get("schema_hash") != schema.hash or a_meta.get("membership_sha256") != membership_hash:
            raise ValueError("legacy A cache is not bound to this schema/membership")
        if a_meta.get("responses_sha256") != file_hash(a_predictions):
            raise ValueError("legacy A response cache changed since its manifest")
    for meta, predictions_path in ((a_meta, a_predictions), (b_meta, b_predictions)):
        if meta is None:
            continue
        declared_schema = meta.get("binding", {}).get("schema_hash")
        if declared_schema is not None and declared_schema != schema.hash:
            raise ValueError("prediction schema binding mismatch")
        provenance = meta.get("provenance")
        declared_membership = (provenance.get("membership_sha256")
                               if isinstance(provenance, dict) else None)
        if declared_membership is not None and declared_membership != membership_hash:
            raise ValueError("prediction membership binding mismatch")
        declared_predictions = meta.get("predictions_sha256")
        if declared_predictions is not None and declared_predictions != file_hash(predictions_path):
            raise ValueError("prediction bytes differ from their metadata binding")
    a_rows, b_rows = _predictions(a_predictions, roles), _predictions(b_predictions, roles)
    selected = {role: [] for role in roles}
    missing_y = {role: 0 for role in roles}
    seen, eligible = set(), set()
    for row in read_jsonl(prepared / "samples.jsonl"):
        sid = row.get("sample_id")
        if sid in seen or sid not in members:
            raise ValueError("duplicate sample or membership missing")
        seen.add(sid)
        member = members[sid]
        if row.get("group_id") != member["group_id"]:
            raise ValueError("sample/membership group mismatch")
        role = member["split"]
        if role not in roles:
            continue
        if row.get("dataset") != schema.dataset:
            raise ValueError("sample dataset mismatch")
        target = row.get("target", {})
        if target.get("status") != "OBSERVED":
            missing_y[role] += 1
            continue
        y = target.get("value")
        if type(y) is not int or not 0 <= y < schema.num_classes:
            raise ValueError("observed task label outside native categories")
        eligible.add(sid)
        if sid not in a_rows or sid not in b_rows:
            raise ValueError("missing source prediction for an eligible task sample: " + sid)
        ar, br = a_rows[sid], b_rows[sid]
        for predicted in (ar, br):
            if predicted.get("group_id") != member["group_id"] or predicted.get("split") != role:
                raise ValueError("prediction sample/group/role join mismatch")
        annotations = row.get("concepts", [])
        labels = {item.get("concept_id"): item for item in annotations}
        if len(labels) != schema.num_atoms or len(annotations) != schema.num_atoms:
            raise ValueError("concept annotations missing or duplicated")
        gold = []
        for concept in schema.concepts:
            annotation = labels.get(concept.id)
            if annotation is None:
                raise ValueError("concept label IDs differ from schema")
            value = annotation.get("value")
            if annotation.get("annotation_status") == "OBSERVED":
                if type(value) is not int or not 0 <= value < len(concept.values):
                    raise ValueError("invalid observed concept category")
                gold.append(value)
            else:
                gold.append(None)
        selected[role].append({"sample_id": sid, "group_id": member["group_id"],
            "A": _hard_vector(ar.get("z") if legacy_a else ar.get("A"), schema.value_counts, "A"),
            "B": _hard_vector(br.get("typed_values"), schema.value_counts, "typed B"),
            "shared": _hard_vector(br.get("shared_values"), schema.value_counts, "shared strong B"),
            "C": tuple(gold), "Y": y})
    if seen != set(members):
        raise ValueError("prepared samples do not cover membership")
    # Extra predictions are permitted only for genuine, target-missing members
    # of a requested role, never for invented IDs or mismatched roles.
    for prediction_map in (a_rows, b_rows):
        for sid, prediction in prediction_map.items():
            if (sid not in members or members[sid]["split"] not in roles or
                    members[sid]["group_id"] != prediction.get("group_id") or
                    members[sid]["split"] != prediction.get("split")):
                raise ValueError("extra prediction is not a valid requested-role member")
    for role in roles:
        selected[role].sort(key=lambda row: row["sample_id"])
        if not selected[role]:
            raise ValueError("requested task population is empty: " + role)
    return VerificationInputs(schema, selected, {
        "schema_hash": schema.hash, "membership_sha256": membership_hash,
        "roles": {role: {"task_observed_rows": len(selected[role]),
                         "groups": len({r["group_id"] for r in selected[role]}),
                         "excluded_missing_task": missing_y[role]} for role in roles},
        "a_predictions_sha256": file_hash(a_predictions),
        "b_predictions_sha256": file_hash(b_predictions),
        "a_manifest_sha256": file_hash(a_metadata) if a_metadata else None,
        "b_manifest_sha256": file_hash(b_metadata) if b_metadata else None,
        "legacy_a": legacy_a, "protected_override": bool(allow_protected),
        "task_ground_truth": "prepared_target_not_prediction_cache",
        "visibility": "PRIVATE_OFFLINE_LABELS_AND_ANSWERS_NOT_POLICY_INPUTS",
    })

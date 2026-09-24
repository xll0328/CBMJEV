"""Leakage-auditable training primitives for prepared nested cross-fitting.

This module is deliberately stricter than :func:`cbmjev.learning.fit_models`.
It accepts explicit out-of-fold source assignments, binds every fitted model
and target package to content hashes, and rechecks source ancestry at every
boundary.  The legacy disjoint-role trainer remains available for existing
experiments; formal nested-OOF executors should use the three public primitives
defined here.
"""

from collections.abc import Mapping
from functools import lru_cache
from itertools import islice
import hashlib
import math
import random

import torch
from torch.nn import functional as F

from .contracts import candidate_actions, stable_hash
from .crossfit_targets import (DiskRecords, TargetWriter, iter_epoch,
                               package_hash, records_hash)
from .learning import (ActionController, MaskedHead, action_targets,
                       encode_actions, encode_states, mask_answers,
                       normalize_config, sample_group_masks, schema_signature,
                       validate_action, validate_observed)
from .provenance import (make_fit_record, validate_provenance_dag,
                         validate_target_exclusion)


_HEAD_PROTOCOL = "cbmjev-nested-oof-head-v1"
_TARGET_FORMAT = "cbmjev-nested-oof-action-targets-v1"
_CONTROLLER_PROTOCOL = "cbmjev-nested-oof-controller-v1"


def _batches(values, size):
    iterator = iter(values)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


def _tensor_digest(network, schema, component, **metadata):
    tensors = {}
    for name, value in sorted(network.state_dict().items()):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        tensors[name] = {
            "dtype": str(value.dtype), "shape": list(value.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return stable_hash({"component": component, "schema_hash": schema_signature(schema),
                        "tensors": tensors, **metadata})


def _head_digest(head, schema):
    if schema_signature(head.schema) != schema_signature(schema):
        raise ValueError("head schema mismatch")
    weights = head.class_weights
    actual = head._loss_weights
    if ((weights is None) != (actual is None)
            or (weights is not None and not torch.equal(
                actual, torch.tensor(weights, dtype=actual.dtype, device=actual.device)))):
        raise ValueError("head runtime class weights differ from declared weights")
    return _tensor_digest(head.network, schema, "head",
                          config=head.config, class_weights=weights)


def _controller_digest(controller, schema):
    return _tensor_digest(controller.network, schema, "controller",
                          objective=controller.objective,
                          pairs=[list(pair) for pair in controller.pairs])


def _validated_rows(rows, schema):
    """Return immutable row projections without relying on legacy split roles."""
    result, sample_ids, group_by_sample = [], set(), {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("crossfit rows must be objects")
        sample_id, group_id = raw.get("sample_id"), raw.get("group_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValueError("crossfit sample_id must be unique and nonempty")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("crossfit group_id must be nonempty")
        z, y = raw.get("z"), raw.get("y")
        if not isinstance(z, (list, tuple)) or len(z) != schema.num_atoms:
            raise ValueError("crossfit response width differs from schema")
        if any(type(value) is not int or not 0 <= value < count
               for value, count in zip(z, schema.num_categories)):
            raise ValueError("crossfit responses must be complete hard runtime categories")
        if type(y) is not int or not 0 <= y < schema.num_classes:
            raise ValueError("invalid crossfit target")
        sample_ids.add(sample_id)
        group_by_sample[sample_id] = group_id
        result.append({"sample_id": sample_id, "group_id": group_id,
                       "z": tuple(z), "y": y})
    if not result:
        raise ValueError("nonempty crossfit rows required")
    return tuple(result)


def _merge_provenance(*collections):
    merged = {}
    for records in collections:
        if isinstance(records, Mapping) or isinstance(records, (str, bytes)):
            raise ValueError("provenance records must be a sequence")
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("provenance record must be an object")
            artifact_id = record.get("artifact_id")
            if artifact_id in merged and merged[artifact_id] != dict(record):
                raise ValueError("conflicting provenance records for " + str(artifact_id))
            merged[artifact_id] = dict(record)
    records = [merged[key] for key in sorted(merged)]
    validate_provenance_dag(records)
    return records


def _validate_response_assignments(rows, assignments, provenance):
    if not isinstance(assignments, Mapping):
        raise ValueError("response_artifact_by_group must be an object")
    target_groups = sorted({row["group_id"] for row in rows})
    if set(assignments) != set(target_groups):
        raise ValueError("response source assignment must cover exactly the target groups")
    record_ids = {record["artifact_id"] for record in provenance}
    groups_by_artifact = {}
    normalized = {}
    for group_id in target_groups:
        artifact_id = assignments[group_id]
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id not in record_ids:
            raise ValueError("response source assignment references an unknown artifact")
        normalized[group_id] = artifact_id
        groups_by_artifact.setdefault(artifact_id, []).append(group_id)
    # This is the central OOF invariant.  It is checked per producing responder,
    # not against the union, because different inner responders legitimately fit
    # other folds whose predictions they never produce.
    for artifact_id, assigned_groups in sorted(groups_by_artifact.items()):
        validate_target_exclusion(provenance, artifact_ids=[artifact_id],
                                  target_group_ids=assigned_groups)
    return normalized


def _rows_hash(rows):
    return stable_hash([{"sample_id": row["sample_id"], "group_id": row["group_id"],
                         "z": list(row["z"]), "y": row["y"]} for row in rows])


def _class_weights(rows, schema, cfg):
    counts = [0] * schema.num_classes
    for row in rows:
        counts[row["y"]] += 1
    weights = None
    if cfg["class_weighting"] == "inverse_frequency":
        if any(count == 0 for count in counts):
            raise ValueError("inverse_frequency requires every class in crossfit head rows")
        weights = tuple(len(rows) / (schema.num_classes * count) for count in counts)
    return counts, weights


def fit_head_only(rows, schema, config, *, response_artifact_by_group,
                  provenance_records):
    """Fit a masked task head from per-row OOF responses.

    ``response_artifact_by_group`` identifies the responder that produced each
    group's cached concepts.  Every producer and all its declared ancestors are
    required to exclude the groups for which it produced responses.
    """
    records = _validated_rows(rows, schema)
    provenance = _merge_provenance(provenance_records)
    assignments = _validate_response_assignments(records, response_artifact_by_group,
                                                 provenance)
    cfg = normalize_config(config, schema)
    class_counts, class_weights = _class_weights(records, schema, cfg)

    torch.manual_seed(cfg["seed"])
    if cfg["device"].startswith("cuda"):
        torch.cuda.manual_seed_all(cfg["seed"])
    torch.use_deterministic_algorithms(cfg["deterministic"])
    if cfg["device"] == "cpu":
        torch.set_num_threads(cfg["cpu_threads"])
    head = MaskedHead(schema, cfg, class_weights=class_weights)
    optimizer = torch.optim.AdamW(head.network.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    rng = random.Random(cfg["seed"] + 31)
    losses = []
    for _ in range(cfg["head_epochs"]):
        head.network.train()
        order = list(range(len(records)))
        rng.shuffle(order)
        loss_sum = count = 0
        for indices in _batches(order, cfg["batch_size"]):
            masks = sample_group_masks(len(indices), schema.num_groups, rng)
            states = [mask_answers(records[index]["z"], mask, schema)
                      for index, mask in zip(indices, masks)]
            targets = torch.tensor([records[index]["y"] for index in indices],
                                   dtype=torch.long, device=head.device)
            loss = head.cross_entropy(head.network(encode_states(states, schema, head.device)),
                                      targets)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite crossfit head training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            count += len(indices)
        losses.append(loss_sum / count)
    head.network.eval().requires_grad_(False)

    component_sha256 = _head_digest(head, schema)
    artifact_id = "head:" + component_sha256
    fit_groups = sorted({row["group_id"] for row in records})
    fit_samples = sorted(row["sample_id"] for row in records)
    head_record = make_fit_record(
        artifact_id, supervised_group_ids=fit_groups,
        parent_ids=sorted(set(assignments.values())),
        metadata={"protocol": _HEAD_PROTOCOL, "component_sha256": component_sha256,
                  "fit_rows_sha256": _rows_hash(records), "config": cfg,
                  "class_weights": class_weights},
    )
    full_provenance = _merge_provenance(provenance, [head_record])
    report = {
        "training_protocol": _HEAD_PROTOCOL,
        "config": cfg,
        "schema_signature": schema_signature(schema),
        "head_artifact_id": artifact_id,
        "head_component_sha256": component_sha256,
        "fit_sample_ids": fit_samples,
        "fit_group_ids": fit_groups,
        "fit_rows_sha256": _rows_hash(records),
        "response_artifact_by_group": dict(sorted(assignments.items())),
        "response_assignment_sha256": stable_hash(dict(sorted(assignments.items()))),
        "provenance": full_provenance,
        "objective": cfg["objective"], "seed": cfg["seed"], "device": cfg["device"],
        "head_fit_rows": len(records), "head_loss": losses,
        "class_weighting": cfg["class_weighting"],
        "head_class_counts": class_counts,
        "head_class_weights": list(class_weights) if class_weights is not None else None,
        "heldout_labels_used": False,
        "oof_response_source_verified": True,
    }
    report["report_sha256"] = stable_hash(report)
    return head, report


def _target_config(cfg):
    return {
        key: cfg[key] for key in (
            "objective", "seed", "batch_size", "masks_per_sample", "include_pairs",
            "include_all", "actions_per_state", "policy_epochs", "pairs",
            "class_weighting")
    }


def _policy_examples_with_identity(rows, schema, cfg, epoch):
    rng = random.Random(cfg["seed"] + 104729 * (epoch + 1))
    order = list(range(len(rows)))
    rng.shuffle(order)
    for index in order:
        row = rows[index]
        masks = list(sample_group_masks(cfg["masks_per_sample"], schema.num_groups, rng))
        masks[0] = (False,) * schema.num_groups
        if len(masks) > 1:
            masks[-1] = (True,) * schema.num_groups
        for mask in masks:
            before = mask_answers(row["z"], mask, schema)
            actions = list(candidate_actions(
                before, schema, include_pairs=cfg["include_pairs"],
                include_all=cfg["include_all"],
                pairs=tuple(tuple(pair) for pair in cfg["pairs"])))
            cap = cfg["actions_per_state"]
            if cap and len(actions) > cap:
                nonempty = [action for action in actions if action]
                all_remaining = tuple(g for g in range(schema.num_groups) if not mask[g])
                retained = [()]
                if cfg["include_all"] and all_remaining in nonempty:
                    retained.append(all_remaining)
                    nonempty.remove(all_remaining)
                retained.extend(rng.sample(nonempty, cap - len(retained)))
                actions = retained
            for action in actions:
                after = list(before)
                for atom in schema.expand(action):
                    after[atom] = row["z"][atom]
                yield row, before, tuple(action), tuple(after), row["y"]


def _validate_head_binding(head, schema, report):
    if not isinstance(report, Mapping) or report.get("training_protocol") != _HEAD_PROTOCOL:
        raise ValueError("head report is not a nested-OOF head receipt")
    if report.get("schema_signature") != schema_signature(schema):
        raise ValueError("head report schema mismatch")
    candidate = dict(report)
    if candidate.pop("report_sha256", None) != stable_hash(candidate):
        raise ValueError("head report hash mismatch")
    digest = _head_digest(head, schema)
    if report.get("head_component_sha256") != digest or report.get("head_artifact_id") != "head:" + digest:
        raise ValueError("head weights do not match the bound nested-OOF receipt")
    provenance = _merge_provenance(report.get("provenance", ()))
    matching = [record for record in provenance
                if record["artifact_id"] == report["head_artifact_id"]]
    if len(matching) != 1:
        raise ValueError("head artifact is absent from its provenance ledger")
    if report.get("config") != head.config or report.get("head_class_weights") != (
            list(head.class_weights) if head.class_weights is not None else None):
        raise ValueError("head configuration or class weights receipt mismatch")
    expected = make_fit_record(
        report["head_artifact_id"], supervised_group_ids=report["fit_group_ids"],
        parent_ids=sorted(set(report["response_artifact_by_group"].values())),
        metadata={"protocol": _HEAD_PROTOCOL, "component_sha256": digest,
                  "fit_rows_sha256": report["fit_rows_sha256"],
                  "config": head.config, "class_weights": head.class_weights})
    # JSON receipts round-trip tuples as lists.
    if stable_hash(matching[0]) != stable_hash(expected):
        raise ValueError("head provenance does not match its receipt")
    return provenance


def construct_action_targets(rows, head, schema, config, *, head_report,
                             response_artifact_by_group, provenance_records,
                             manifest_path=None):
    """Construct label-derived one-step targets using only OOF model outputs.

    Returned model-facing records intentionally omit labels, post-action hidden
    states, full responses, and sample content.  Source identities and hashes
    remain in the package receipt for audit. With ``manifest_path``, records are
    streamed to epoch JSONL files and the returned package has a reiterable
    DiskRecords instead of a list. Logical hashes and event order are unchanged.
    """
    target_rows = _validated_rows(rows, schema)
    head_provenance = _validate_head_binding(head, schema, head_report)
    provenance = _merge_provenance(head_provenance, provenance_records)
    assignments = _validate_response_assignments(
        target_rows, response_artifact_by_group, provenance)
    target_groups = sorted({row["group_id"] for row in target_rows})
    validate_target_exclusion(
        provenance, artifact_ids=[head_report["head_artifact_id"]],
        target_group_ids=target_groups)

    cfg = normalize_config(config, schema)
    if cfg["class_weighting"] != head.config["class_weighting"]:
        raise ValueError("target class weighting differs from fitted head")
    writer = TargetWriter(manifest_path) if manifest_path is not None else None
    output = writer if writer is not None else []
    counts = []
    try:
        for epoch in range(cfg["policy_epochs"]):
            examples = _policy_examples_with_identity(target_rows, schema, cfg, epoch)
            epoch_count = 0
            for batch in _batches(examples, cfg["batch_size"]):
                raw_examples = [(before, action, after, y)
                                for _, before, action, after, y in batch]
                targets = action_targets(head, raw_examples, cfg["objective"]).cpu().tolist()
                for (row, before, action, _, _), target in zip(batch, targets):
                    if not math.isfinite(target):
                        raise ValueError("nonfinite action target")
                    output.append({
                        "sample_id": row["sample_id"], "group_id": row["group_id"],
                        "epoch": epoch, "observed": list(before),
                        "action": list(action), "target": float(target),
                    })
                    epoch_count += 1
            counts.append(epoch_count)
    finally:
        if writer is not None:
            writer.close()
    if writer is not None:
        output = writer.records()
    records_sha256 = records_hash(output)
    source_core = {
        "schema_signature": schema_signature(schema), "objective": cfg["objective"],
        "head_artifact_id": head_report["head_artifact_id"],
        "head_component_sha256": head_report["head_component_sha256"],
        "target_group_ids": target_groups,
        "target_sample_ids": sorted(row["sample_id"] for row in target_rows),
        "target_rows_sha256": _rows_hash(target_rows),
        "response_artifact_by_group": dict(sorted(assignments.items())),
        "target_config": _target_config(cfg), "records_sha256": records_sha256,
    }
    target_artifact_id = "action-targets:" + stable_hash(source_core)
    target_record = make_fit_record(
        target_artifact_id, supervised_group_ids=target_groups,
        parent_ids=sorted(set(assignments.values()) | {head_report["head_artifact_id"]}),
        fit_kind="target_construction",
        metadata={"format": _TARGET_FORMAT, "records_sha256": records_sha256,
                  "target_rows_sha256": source_core["target_rows_sha256"]},
    )
    full_provenance = _merge_provenance(provenance, [target_record])
    package = {
        "format": _TARGET_FORMAT, **source_core,
        "target_artifact_id": target_artifact_id,
        "record_count": len(output), "records_per_epoch": counts,
        "records": output, "provenance": full_provenance,
        "contains_model_facing_labels": False,
        "contains_hidden_post_action_states": False,
        "oof_source_exclusion_verified": True,
    }
    package["package_sha256"] = package_hash(package)
    if writer is not None:
        writer.finish(package, validate=lambda value: _validate_target_package(value, schema, cfg))
    return package


def _candidate_pool(schema, cfg):
    """Invocation-local bounded memo of actions, never of record validity."""
    @lru_cache(maxsize=2048)
    def candidates(mask):
        # Candidate identities depend solely on group visibility. Use a valid
        # canonical representative, not unvalidated record values, as input.
        state = [-1] * schema.num_atoms
        for group, visible in zip(schema.groups, mask):
            if visible:
                for atom in group.atoms:
                    state[atom] = 0
        return frozenset(candidate_actions(
            tuple(state), schema, include_pairs=cfg["include_pairs"],
            include_all=cfg["include_all"], pairs=cfg["pairs"]))
    return candidates


def _encode_target_batch(records, schema, device):
    """Fast tensorization for already package-bound action-target records."""
    states = [record["observed"] for record in records]
    z = torch.tensor(states, dtype=torch.long, device=device)
    mask = z >= 0
    safe = torch.where(mask, z, torch.zeros_like(z))
    pieces = [F.one_hot(safe[:, atom], count).float() * mask[:, atom:atom + 1]
              for atom, count in enumerate(schema.num_categories)]
    state_x = torch.cat(pieces + [mask.float()], dim=-1)
    action_x = torch.zeros((len(records), schema.num_groups + 1),
                           dtype=torch.float32, device=device)
    row_indices, column_indices = [], []
    for row_index, record in enumerate(records):
        action = record["action"]
        if action:
            for group in action:
                row_indices.append(row_index)
                column_indices.append(group)
        else:
            row_indices.append(row_index)
            column_indices.append(schema.num_groups)
    action_x[torch.tensor(row_indices, dtype=torch.long, device=device),
             torch.tensor(column_indices, dtype=torch.long, device=device)] = 1.0
    target = torch.tensor([record["target"] for record in records],
                          dtype=torch.float32, device=device)
    return torch.cat((state_x, action_x), dim=-1), target


def _validate_target_package(package, schema, cfg):
    _validate_target_package_metadata(package, schema, cfg)
    candidate = dict(package)
    package_sha256 = candidate.pop("package_sha256", None)
    if package_sha256 != package_hash(candidate):
        raise ValueError("action-target package hash mismatch")
    records = package.get("records")
    if package.get("record_count") != len(records) or package.get("records_sha256") != records_hash(records):
        raise ValueError("action-target record count or hash mismatch")
    sample_groups, epoch_counts = {}, [0] * cfg["policy_epochs"]
    sample_set, group_set = set(package["target_sample_ids"]), set(package["target_group_ids"])
    candidates = _candidate_pool(schema, cfg)
    for record in records:
        if set(record) != {"sample_id", "group_id", "epoch", "observed", "action", "target"}:
            raise ValueError("action-target record has unexpected or missing fields")
        sample_id, group_id, epoch = record["sample_id"], record["group_id"], record["epoch"]
        if sample_id not in sample_set or group_id not in group_set:
            raise ValueError("action-target record identity is outside package scope")
        if sample_id in sample_groups and sample_groups[sample_id] != group_id:
            raise ValueError("action-target sample crosses groups")
        sample_groups[sample_id] = group_id
        if type(epoch) is not int or not 0 <= epoch < cfg["policy_epochs"]:
            raise ValueError("invalid action-target epoch")
        state = validate_observed(record["observed"], schema)
        action = validate_action(record["action"], state, schema)
        mask = tuple(state[group.atoms[0]] >= 0 for group in schema.groups)
        if action not in candidates(mask):
            raise ValueError("action-target action is outside configured candidates")
        target = record["target"]
        if (isinstance(target, bool) or not isinstance(target, (int, float))
                or not math.isfinite(target)):
            raise ValueError("invalid action target")
        if cfg["objective"] == "risk" and not 0 <= target <= 1:
            raise ValueError("risk action target must lie in [0, 1]")
        epoch_counts[epoch] += 1
    if set(sample_groups) != set(package["target_sample_ids"]) or set(sample_groups.values()) != set(package["target_group_ids"]):
        raise ValueError("action-target records do not cover declared samples/groups")
    if package.get("records_per_epoch") != epoch_counts or any(count == 0 for count in epoch_counts):
        raise ValueError("action-target epoch counts mismatch")
    return _validate_target_package_lineage(package)


def _validate_target_package_metadata(package, schema, cfg):
    """Cheap package validation for already completed, file-hash-bound outer folds."""
    if not isinstance(package, Mapping) or package.get("format") != _TARGET_FORMAT:
        raise ValueError("invalid nested-OOF action-target package")
    if package.get("schema_signature") != schema_signature(schema):
        raise ValueError("action-target schema mismatch")
    if package.get("objective") != cfg["objective"]:
        raise ValueError("action-target objective mismatch")
    if package.get("target_config") != _target_config(cfg):
        raise ValueError("controller configuration differs from target construction")
    records = package.get("records")
    if not isinstance(records, (list, DiskRecords)) or not records:
        raise ValueError("action-target records must be a nonempty list")
    if package.get("record_count") != len(records):
        raise ValueError("action-target record count mismatch")
    samples = package.get("target_sample_ids")
    groups = package.get("target_group_ids")
    if (not isinstance(samples, list) or not samples or len(set(samples)) != len(samples)
            or not isinstance(groups, list) or not groups or len(set(groups)) != len(groups)):
        raise ValueError("invalid action-target sample/group identity sets")
    if (not isinstance(package.get("records_per_epoch"), list)
            or len(package["records_per_epoch"]) != cfg["policy_epochs"]
            or sum(package["records_per_epoch"]) != package["record_count"]
            or any(type(count) is not int or count < 1 for count in package["records_per_epoch"])):
        raise ValueError("action-target epoch counts mismatch")
    if not isinstance(package.get("records_sha256"), str) or len(package["records_sha256"]) != 64:
        raise ValueError("invalid action-target records hash")


def _validate_target_package_lineage(package):
    provenance = _merge_provenance(package.get("provenance", ()))
    assignments = _validate_response_assignments(
        [{"group_id": group_id} for group_id in package["target_group_ids"]],
        package.get("response_artifact_by_group"), provenance)
    validate_target_exclusion(
        provenance, artifact_ids=[package.get("head_artifact_id")],
        target_group_ids=package["target_group_ids"])
    source_core = {
        key: package[key] for key in (
            "schema_signature", "objective", "head_artifact_id",
            "head_component_sha256", "target_group_ids", "target_sample_ids",
            "target_rows_sha256", "response_artifact_by_group", "target_config",
            "records_sha256")
    }
    expected_target_id = "action-targets:" + stable_hash(source_core)
    if package.get("target_artifact_id") != expected_target_id:
        raise ValueError("action-target artifact identity mismatch")
    target_records = [record for record in provenance
                      if record["artifact_id"] == expected_target_id]
    if len(target_records) != 1:
        raise ValueError("action-target provenance record missing")
    expected_record = make_fit_record(
        expected_target_id, supervised_group_ids=package["target_group_ids"],
        parent_ids=sorted(set(assignments.values()) | {package["head_artifact_id"]}),
        fit_kind="target_construction",
        metadata={"format": _TARGET_FORMAT, "records_sha256": package["records_sha256"],
                  "target_rows_sha256": package["target_rows_sha256"]},
    )
    if target_records[0] != expected_record:
        raise ValueError("action-target provenance does not match package sources")
    return provenance


def fit_controller_from_targets(target_packages, schema, config, *, validate_packages=True,
                                training_batch_size=None):
    """Fit one controller from one or more disjoint outer-fold target packages."""
    cfg = normalize_config(config, schema)
    batch_size = cfg["batch_size"]
    if training_batch_size is not None:
        if type(training_batch_size) is not int or training_batch_size < 1:
            raise ValueError("training_batch_size must be a positive integer")
        batch_size = training_batch_size
    packages = ([target_packages] if isinstance(target_packages, Mapping)
                else list(target_packages))
    if not packages:
        raise ValueError("at least one action-target package is required")
    for package in packages:
        if validate_packages:
            _validate_target_package(package, schema, cfg)
        else:
            _validate_target_package_metadata(package, schema, cfg)
            _validate_target_package_lineage(package)
    packages.sort(key=lambda package: package["target_artifact_id"])
    seen_samples, seen_groups = set(), set()
    for package in packages:
        if seen_samples.intersection(package["target_sample_ids"]):
            raise ValueError("sample overlap across outer-fold target packages")
        if seen_groups.intersection(package["target_group_ids"]):
            raise ValueError("group overlap across outer-fold target packages")
        seen_samples.update(package["target_sample_ids"])
        seen_groups.update(package["target_group_ids"])

    torch.manual_seed(cfg["seed"] + 1009)
    if cfg["device"].startswith("cuda"):
        torch.cuda.manual_seed_all(cfg["seed"] + 1009)
    torch.use_deterministic_algorithms(cfg["deterministic"])
    if cfg["device"] == "cpu":
        torch.set_num_threads(cfg["cpu_threads"])
    controller = ActionController(schema, cfg)
    optimizer = torch.optim.AdamW(controller.network.parameters(),
                                  lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    losses, counts = [], []
    for epoch in range(cfg["policy_epochs"]):
        controller.network.train()
        epoch_records = (record for package in packages for record in iter_epoch(package, epoch))
        loss_sum = count = 0
        for batch in _batches(epoch_records, batch_size):
            x, target = _encode_target_batch(batch, schema, controller.device)
            logits = controller.network(x).flatten()
            loss = (F.binary_cross_entropy_with_logits(logits, target)
                    if cfg["objective"] == "risk" else F.mse_loss(logits, target))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite crossfit controller training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
            count += len(batch)
        losses.append(loss_sum / count)
        counts.append(count)
    controller.network.eval().requires_grad_(False)

    provenance = _merge_provenance(*(package["provenance"] for package in packages))
    component_sha256 = _controller_digest(controller, schema)
    artifact_id = "controller:" + component_sha256
    target_ids = [package["target_artifact_id"] for package in packages]
    controller_record = make_fit_record(
        artifact_id, supervised_group_ids=(), parent_ids=target_ids,
        fit_kind="derived",
        metadata={"protocol": _CONTROLLER_PROTOCOL,
                  "component_sha256": component_sha256,
                  "target_package_sha256": [package["package_sha256"] for package in packages]},
    )
    full_provenance = _merge_provenance(provenance, [controller_record])
    report = {
        "training_protocol": _CONTROLLER_PROTOCOL,
        "config": cfg,
        "schema_signature": schema_signature(schema),
        "controller_artifact_id": artifact_id,
        "controller_component_sha256": component_sha256,
        "target_artifact_ids": target_ids,
        "target_package_sha256": [package["package_sha256"] for package in packages],
        "target_sample_ids": sorted(seen_samples),
        "target_group_ids": sorted(seen_groups),
        "objective": cfg["objective"], "seed": cfg["seed"], "device": cfg["device"],
        "policy_loss": losses, "policy_examples_per_epoch": counts,
        "controller_training_batch_size": batch_size,
        "target_construction_batch_size": cfg["batch_size"],
        "provenance": full_provenance,
        "oof_source_exclusion_verified": True,
        "final_models_used_for_same_sample_targets": False,
    }
    return controller, report

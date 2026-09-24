"""One training-only global prefix, scored with excluding outer-fold heads.

This is an unweighted CE-gain ranking, not a cost-normalized acquisition policy
or an estimate of the final refitted head's risk. No legacy split is rewritten.
"""
import math
from pathlib import Path

from .baselines import _mean_ce
from .contracts import stable_hash
from .crossfit_artifacts import load_crossfit_head
from .crossfit_cache import crossfit_cache_source_binding, load_crossfit_cache
from .crossfit_merge import load_outer_folds
from .crossfit_training import _merge_provenance, _rows_hash
from .io import file_hash, fresh_dir, read_json, write_json
from .learning import mask_answers, schema_signature
from .pipeline import code_fingerprint
from .provenance import make_fit_record, validate_target_exclusion


def _load_inputs(prepared, planned, outer_dirs, device, *, responder_source_dir=None):
    schema, data = load_outer_folds(prepared, planned, outer_dirs,
                                    responder_source_dir=responder_source_dir,
                                    validate_target_records=False)
    folds, bindings, lineage = [], [], []
    for source in data["sources"]:
        directory = Path(source["directory"])
        head, report = load_crossfit_head(directory / "head", schema, device=device)
        _, rows, manifest = load_crossfit_cache(
            directory / "outer/cache", prepared, planned, directory / "outer/responder",
            responder_source_dir=responder_source_dir)
        provenance = _merge_provenance(report["provenance"], manifest["provenance"])
        parents = [report["head_artifact_id"], manifest["responder_artifact_id"]]
        groups = sorted({r["group_id"] for r in rows})
        validate_target_exclusion(provenance, artifact_ids=parents, target_group_ids=groups)
        folds.append((head, rows))
        bindings.append({**source, "head_artifact_id": parents[0],
            "responder_artifact_id": parents[1], "cache_manifest_hash": manifest["manifest_hash"],
            "target_rows_sha256": _rows_hash(rows),
            "target_sample_ids": [r["sample_id"] for r in rows], "target_group_ids": groups,
            "num_rows": len(rows)})
        if responder_source_dir is not None:
            bindings[-1]["response_source_binding"] = crossfit_cache_source_binding(
                directory / "outer/cache", directory / "outer/responder", responder_source_dir)
        lineage.extend(provenance)
    return schema, folds, bindings, _merge_provenance(lineage), data


def _reader_agnostic_bindings(bindings):
    """Compare persisted historical response bindings across reader-only edits.

    ``current_reader_source_code_hash`` documents which local code performed the
    verification.  It changes after harmless loader/evaluator edits and should
    not invalidate a training-only static order when the historical source,
    semantic files, receipts, and cache manifests still match.
    """
    normalized = []
    for binding in bindings:
        item = dict(binding)
        response = item.get("response_source_binding")
        if isinstance(response, dict):
            response = dict(response)
            response.pop("current_reader_source_code_hash", None)
            item["response_source_binding"] = response
        normalized.append(item)
    return normalized


def _greedy_order(folds, schema, batch_size):
    """Pool rows, NOT fold means equally; each row retains its excluding head."""
    count = sum(len(rows) for _, rows in folds)
    if not count:
        raise ValueError("static order requires observed OOF task targets")

    def loss(mask):
        result = math.fsum(len(rows) * _mean_ce(head,
            ((mask_answers(row["z"], mask, schema), row["y"]) for row in rows), batch_size)
            for head, rows in folds) / count
        if not math.isfinite(result):
            raise ValueError("nonfinite crossfit static-order loss")
        return result

    order, stages = [], []
    mask = [False] * schema.num_groups
    current = loss(mask)
    while len(order) < schema.num_groups:
        losses = {}
        for group in range(schema.num_groups):
            if not mask[group]:
                after = list(mask)
                after[group] = True
                losses[group] = loss(after)
        gains = {group: current - value for group, value in losses.items()}
        selected = min(gains, key=lambda group: (-gains[group], group))
        stages.append({"prefix": list(order), "selected": selected,
            "selected_group_id": schema.groups[selected].id,
            "mean_signed_ce_gain": gains[selected],
            "candidate_gains": {str(g): gain for g, gain in gains.items()},
            "candidate_group_ids": {str(g): schema.groups[g].id for g in gains}})
        order.append(selected)
        mask[selected] = True
        current = losses[selected]
    return tuple(order), stages


def fit_crossfit_static_order(prepared, planned, outer_dirs, out, *, device="cpu",
                              batch_size=64, responder_source_dir=None):
    """Persist a validated global OOF prefix; return ``(order, report)``.

    Every outer-heldout row is scored by its own frozen inner-OOF head, not the
    final refitted head. All observed training targets are pooled row-wise. The
    output must be new; ``receipt.json`` is the last-written completion marker.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    out = Path(out)
    if out.exists():
        raise ValueError("static-order output must be new")
    outer_dirs = list(outer_dirs)
    source = code_fingerprint()
    schema, folds, bindings, lineage, data = _load_inputs(
        prepared, planned, outer_dirs, device, responder_source_dir=responder_source_dir)
    order, stages = _greedy_order(folds, schema, batch_size)
    rows = data["rows"]
    core = {"format": "cbmjev-crossfit-static-order-v1",
        "method": "GLOBAL_GREEDY_STATIC_PREFIX_OUTER_OOF", "order": list(order),
        "order_group_ids": [schema.groups[g].id for g in order],
        "schema_signature": schema_signature(schema), "plan_binding": data["plan_binding"],
        "outer_training_identity": data["outer_training_identity"], "sources": bindings,
        "fit_sample_ids": [r["sample_id"] for r in rows],
        "fit_group_ids": sorted({r["group_id"] for r in rows}),
        "fit_rows_sha256": _rows_hash(rows), "rows_used": len(rows),
        "fit_scope": "ALL_OBSERVED_OUTER_OOF_TRAIN_TARGETS",
        "aggregation": "row-weighted pooled loss; each row scored by excluding outer head",
        "loss_definition": "unweighted_CE_independent_of_head_training_weighting",
        "probability_floor_for_log": 1e-12, "stages": stages,
        "batch_size": batch_size, "execution_device": str(device), "source_code_hash": source,
        "evaluation_holdout_labels_used": False, "outer_heldout_training_labels_used": True,
        "final_head_used_for_ranking": False,
        "cost_assumptions": {
            "ranking": "absolute CE gain, not gain per cost; no learned or measured costs",
            "natural_budget": "number of singleton query groups with equal per-group acquisition cost",
            "unequal_costs": "runtime may enforce declared costs; this ranking is not cost-optimal",
            "batching": "singleton prefix only; no pair/shared-compute benefit optimized"},
        "evidence_status": "TRAINING_ARTIFACT_NOT_FINAL_HEAD_RISK_OR_LATENCY_EVIDENCE"}
    artifact_id = "static_order:" + stable_hash(core)
    record = make_fit_record(artifact_id, supervised_group_ids=core["fit_group_ids"],
        parent_ids=sorted({item[key] for item in bindings
                           for key in ("head_artifact_id", "responder_artifact_id")}),
        metadata={"protocol": core["format"], "core_sha256": stable_hash(core),
                  "candidate_group_ids": [g.id for g in schema.groups]})
    report = {**core, "static_order_artifact_id": artifact_id,
              "provenance": _merge_provenance(lineage, [record])}
    report["report_hash"] = stable_hash(report)
    # Reopen persisted sources before publishing; mutable in-memory heads are
    # never accepted as substitutes for a completed outer-fold execution.
    _, _, checked, _, current = _load_inputs(prepared, planned, outer_dirs, device,
                                             responder_source_dir=responder_source_dir)
    if (checked != bindings or current["plan_binding"] != data["plan_binding"]
            or code_fingerprint() != source):
        raise ValueError("static-order sources changed during fitting")
    out = fresh_dir(out)
    write_json(out / "static_order.json", report)
    receipt = {"format": "cbmjev-crossfit-static-artifact-v1", "status": "COMPLETE",
        "static_order_artifact_id": artifact_id,
        "static_order_sha256": file_hash(out / "static_order.json"),
        "report_hash": report["report_hash"], "schema_signature": schema_signature(schema)}
    receipt["receipt_hash"] = stable_hash(receipt)
    write_json(out / "receipt.json", receipt)
    return order, report


def load_crossfit_static_order(directory, prepared, planned, outer_dirs, *, device="cpu",
                               responder_source_dir=None):
    """Validate a persisted training-only order against all original outer fits.

    Check hashes, complete source identity, semantic greedy trace and exact
    lineage. This does not rerun CE scoring or authenticate rewritten history.
    The ranking-source fingerprint is preserved, not relabeled as current code.
    """
    directory = Path(directory)
    receipt = read_json(directory / "receipt.json")
    unsigned = dict(receipt)
    if unsigned.pop("receipt_hash", None) != stable_hash(unsigned):
        raise ValueError("static-order receipt hash mismatch")
    report = read_json(directory / "static_order.json")
    core = dict(report)
    report_hash = core.pop("report_hash", None)
    if report_hash != stable_hash(core):
        raise ValueError("static-order report hash mismatch")
    artifact_id = core.pop("static_order_artifact_id", None)
    provenance = core.pop("provenance", None)
    if artifact_id != "static_order:" + stable_hash(core):
        raise ValueError("static-order artifact identity mismatch")
    schema, _, bindings, lineage, data = _load_inputs(
        prepared, planned, outer_dirs, device, responder_source_dir=responder_source_dir)
    expected_receipt = {"format": "cbmjev-crossfit-static-artifact-v1", "status": "COMPLETE",
        "static_order_artifact_id": artifact_id,
        "static_order_sha256": file_hash(directory / "static_order.json"),
        "report_hash": report_hash, "schema_signature": schema_signature(schema)}
    if unsigned != expected_receipt:
        raise ValueError("static-order receipt/report/schema mismatch")
    rows = data["rows"]
    expected = {"format": "cbmjev-crossfit-static-order-v1",
        "method": "GLOBAL_GREEDY_STATIC_PREFIX_OUTER_OOF",
        "schema_signature": schema_signature(schema), "plan_binding": data["plan_binding"],
        "outer_training_identity": data["outer_training_identity"], "sources": bindings,
        "fit_sample_ids": [r["sample_id"] for r in rows],
        "fit_group_ids": sorted({r["group_id"] for r in rows}),
        "fit_rows_sha256": _rows_hash(rows), "rows_used": len(rows),
        "fit_scope": "ALL_OBSERVED_OUTER_OOF_TRAIN_TARGETS",
        "aggregation": "row-weighted pooled loss; each row scored by excluding outer head",
        "loss_definition": "unweighted_CE_independent_of_head_training_weighting",
        "probability_floor_for_log": 1e-12,
        "evaluation_holdout_labels_used": False, "outer_heldout_training_labels_used": True,
        "final_head_used_for_ranking": False,
        "cost_assumptions": {
            "ranking": "absolute CE gain, not gain per cost; no learned or measured costs",
            "natural_budget": "number of singleton query groups with equal per-group acquisition cost",
            "unequal_costs": "runtime may enforce declared costs; this ranking is not cost-optimal",
            "batching": "singleton prefix only; no pair/shared-compute benefit optimized"},
        "evidence_status": "TRAINING_ARTIFACT_NOT_FINAL_HEAD_RISK_OR_LATENCY_EVIDENCE"}
    actual_core = dict(core)
    if "sources" in actual_core:
        actual_core["sources"] = _reader_agnostic_bindings(actual_core["sources"])
    expected_core = dict(expected)
    expected_core["sources"] = _reader_agnostic_bindings(expected_core["sources"])
    if any(actual_core.get(k) != v for k, v in expected_core.items()):
        raise ValueError("static-order source binding or scoring semantics mismatch")
    if (type(core.get("batch_size")) is not int or core["batch_size"] < 1
            or not isinstance(core.get("execution_device"), str) or not core["execution_device"]):
        raise ValueError("invalid static-order execution metadata")
    source_hash = core.get("source_code_hash")
    if not isinstance(source_hash, str) or len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
        raise ValueError("invalid static-order source fingerprint")
    order = core.get("order")
    if (not isinstance(order, list) or any(type(g) is not int for g in order)
            or sorted(order) != list(range(schema.num_groups))
            or core.get("order_group_ids") != [schema.groups[g].id for g in order]):
        raise ValueError("static order must be a semantic group permutation")
    stages = core.get("stages")
    if not isinstance(stages, list) or len(stages) != len(order):
        raise ValueError("static-order stages do not cover order")
    for index, (selected, stage) in enumerate(zip(order, stages)):
        remaining = set(range(schema.num_groups)) - set(order[:index])
        gains = stage.get("candidate_gains", {})
        if (set(gains) != {str(g) for g in remaining}
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in gains.values())):
            raise ValueError("static-order gains must cover finite remaining candidates")
        expected_stage = {"prefix": order[:index], "selected": selected,
            "selected_group_id": schema.groups[selected].id,
            "mean_signed_ce_gain": gains[str(selected)], "candidate_gains": gains,
            "candidate_group_ids": {str(g): schema.groups[g].id for g in remaining}}
        if stage != expected_stage or selected != min(remaining, key=lambda g: (-gains[str(g)], g)):
            raise ValueError("static-order stages violate greedy ranking/tie-breaking")
    record = make_fit_record(artifact_id, supervised_group_ids=core["fit_group_ids"],
        parent_ids=sorted({item[key] for item in bindings
                           for key in ("head_artifact_id", "responder_artifact_id")}),
        metadata={"protocol": core["format"], "core_sha256": stable_hash(core),
                  "candidate_group_ids": [g.id for g in schema.groups]})
    if provenance != _merge_provenance(lineage, [record]):
        raise ValueError("static-order provenance differs from verified source ancestry")
    return tuple(order), report

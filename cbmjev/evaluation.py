"""Auditable trace summaries, paired group inference, and fixed-family bounds."""

import datetime
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping

from .provenance import canonical_hash


def _finite(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be a finite number.".format(name))
    try:
        converted = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("{} is outside the finite numeric range.".format(name)) from exc
    if not math.isfinite(converted) or (minimum is not None and converted < minimum):
        raise ValueError("{} must be finite and >= {}.".format(name, minimum))
    return converted


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}.".format(name, minimum))
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a nonempty string.".format(name))
    return value


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError("{} must be a lowercase 64-character SHA256.".format(name))
    return value


def _count(value, name):
    if isinstance(value, int) and not isinstance(value, bool):
        return _integer(value, name)
    if not isinstance(value, (list, tuple)):
        raise ValueError("{} must be a count or a list of unique IDs.".format(name))
    typed = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise ValueError("{} IDs must be strings or integers.".format(name))
        if isinstance(item, str):
            _text(item, name)
        elif item < 0:
            raise ValueError("{} integer IDs must be nonnegative.".format(name))
        typed.append((type(item).__name__, item))
    if len(set(typed)) != len(typed):
        raise ValueError("{} contains duplicate acquired IDs.".format(name))
    return len(value)


def _validate_traces(traces, num_classes=None):
    if num_classes is not None:
        _integer(num_classes, "num_classes", 1)
    if isinstance(traces, (str, bytes, Mapping)):
        raise ValueError("traces must be an iterable of trace objects.")
    rows, seen = [], set()
    for source in traces:
        if not isinstance(source, Mapping):
            raise ValueError("Each trace must be an object.")
        row = dict(source)
        canonical_hash(row)  # Reject nonfinite values even in nested steps/metadata.
        for key in ("sample_id", "group_id", "split", "method"):
            _text(row.get(key), key)
        if row["sample_id"] in seen:
            raise ValueError("Duplicate sample_id within a system: {}".format(row["sample_id"]))
        seen.add(row["sample_id"])
        for key in ("y", "prediction"):
            _integer(row.get(key), key)
            if num_classes is not None and row[key] >= num_classes:
                raise ValueError("{} falls outside fixed num_classes.".format(key))
        for key in ("queried_groups", "queried_atoms"):
            _count(row.get(key), key)
        _integer(row.get("calls"), "calls")
        _finite(row.get("declared_cost"), "declared_cost", 0)
        if row.get("mode") not in {"offline_replay", "live"}:
            raise ValueError("mode must be offline_replay or live.")
        if "steps" not in row or not isinstance(row["steps"], list):
            raise ValueError("steps must be a list.")
        if row["mode"] == "live" and "total_wall_ms" not in row:
            raise ValueError("Live traces require measured total_wall_ms.")
        if "total_wall_ms" in row:
            _finite(row["total_wall_ms"], "total_wall_ms", 0)
        rows.append(row)
    if not rows:
        raise ValueError("At least one trace is required.")
    for key in ("method", "split", "mode"):
        if len({row[key] for row in rows}) != 1:
            raise ValueError("A trace collection must have one {}.".format(key))
    return rows


def _quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("A quantile requires observations.")
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _group_vectors(rows):
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    result = {}
    for group_id, members in sorted(by_group.items()):
        result[group_id] = {
            "num_samples": len(members),
            "error": statistics.mean(int(row["prediction"] != row["y"]) for row in members),
            "declared_cost": statistics.mean(row["declared_cost"] for row in members),
            "queried_groups": statistics.mean(_count(row["queried_groups"], "queried_groups") for row in members),
            "queried_atoms": statistics.mean(_count(row["queried_atoms"], "queried_atoms") for row in members),
            "calls": statistics.mean(row["calls"] for row in members),
        }
    return result


def summarize_traces(traces, *, num_classes):
    """Summarize one method/split/mode, retaining sample vs group estimands."""
    rows = _validate_traces(traces, num_classes)
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for row in rows:
        confusion[row["y"]][row["prediction"]] += 1
    class_f1 = []
    for label in range(num_classes):
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in range(num_classes) if other != label)
        fn = sum(confusion[label][other] for other in range(num_classes) if other != label)
        denominator = 2 * tp + fp + fn
        class_f1.append(2 * tp / denominator if denominator else 0.0)
    group_vectors = _group_vectors(rows)
    error = statistics.mean(int(row["prediction"] != row["y"]) for row in rows)
    output = {
        "schema_version": 1,
        "method": rows[0]["method"],
        "split": rows[0]["split"],
        "mode": rows[0]["mode"],
        "num_samples": len(rows),
        "num_groups": len(group_vectors),
        "num_classes": num_classes,
        "accuracy": 1 - error,
        "error": error,
        "macro_f1": statistics.mean(class_f1),
        "macro_f1_zero_division": 0,
        "macro_f1_includes_all_fixed_classes": True,
        "per_class_f1": class_f1,
        "confusion_matrix_true_rows": confusion,
        "group_mean_risk": statistics.mean(item["error"] for item in group_vectors.values()),
        "group_mean_declared_cost": statistics.mean(item["declared_cost"] for item in group_vectors.values()),
        "aggregation": {
            "accuracy_macro_f1": "all_samples",
            "group_mean_risk": "equal_weight_groups_then_mean_sample_0_1_error",
            "cost_and_query_means": "all_samples",
        },
        "group_metrics": group_vectors,
        "independent_groups_verified": False,
        "group_disjointness_implies_iid": False,
        "declared_cost_is_measured_deployment_latency": False,
    }
    for key in ("queried_groups", "queried_atoms"):
        values = [_count(row[key], key) for row in rows]
        output["mean_" + key] = statistics.mean(values)
        output["total_" + key] = sum(values)
    for key in ("calls", "declared_cost"):
        values = [row[key] for row in rows]
        output["mean_" + key] = statistics.mean(values)
        output["total_" + key] = sum(values)
    if rows[0]["mode"] == "live":
        wall = [row["total_wall_ms"] for row in rows]
        output["deployment_latency"] = {
            "evidence": "live_measured_wall_clock",
            "unit": "ms",
            "num_samples": len(wall),
            "p50": _quantile(wall, 0.50),
            "p95": _quantile(wall, 0.95),
            "quantile_method": "linear_order_statistics",
            "hardware_concurrency_control_verified": False,
        }
    else:
        output["deployment_latency"] = None
        output["latency_unavailable_reason"] = "offline_replay_is_not_deployment_latency_evidence"
        output["ignored_replay_wall_measurements"] = sum("total_wall_ms" in row for row in rows)
    canonical_hash(output)  # Finite inputs can still overflow an aggregate sum.
    return output


def _align(rows_a, rows_b):
    index_a = {row["sample_id"]: row for row in rows_a}
    index_b = {row["sample_id"]: row for row in rows_b}
    if set(index_a) != set(index_b):
        raise ValueError("Paired systems must have exactly the same sample_id set.")
    for sample_id in sorted(index_a):
        for field in ("group_id", "y", "split", "mode"):
            if index_a[sample_id][field] != index_b[sample_id][field]:
                raise ValueError("Paired {} mismatch for sample {}.".format(field, sample_id))
    return index_a, index_b


def paired_group_bootstrap(traces_a, traces_b, *, seed, n_resamples=10000, confidence=0.95):
    """Percentile paired bootstrap of equal-weight group error and cost gaps.

    Differences are A minus B: lower error and lower cost are preferable.
    This quantifies group sampling variation, not model-training seed variation.
    """
    _integer(seed, "seed", 0)
    _integer(n_resamples, "n_resamples", 1)
    confidence = _finite(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1.")
    rows_a, rows_b = _validate_traces(traces_a), _validate_traces(traces_b)
    _align(rows_a, rows_b)
    groups_a, groups_b = _group_vectors(rows_a), _group_vectors(rows_b)
    if set(groups_a) != set(groups_b):
        raise ValueError("Paired systems must have exactly the same group set.")
    group_ids = sorted(groups_a)
    differences = {
        metric: [groups_a[group][metric] - groups_b[group][metric] for group in group_ids]
        for metric in ("error", "declared_cost")
    }
    rng = random.Random(seed)
    draws = {"error": [], "declared_cost": []}
    for _ in range(n_resamples):
        indices = [rng.randrange(len(group_ids)) for _ in group_ids]
        for metric in draws:
            draws[metric].append(statistics.mean(differences[metric][index] for index in indices))
    tail = (1 - confidence) / 2
    estimates = {
        metric: {
            "difference_a_minus_b": statistics.mean(differences[metric]),
            "ci_lower": _quantile(draws[metric], tail),
            "ci_upper": _quantile(draws[metric], 1 - tail),
            "confidence_level": confidence,
        }
        for metric in draws
    }
    return {
        "method_a": rows_a[0]["method"],
        "method_b": rows_b[0]["method"],
        "split": rows_a[0]["split"],
        "mode": rows_a[0]["mode"],
        "num_samples": len(rows_a),
        "num_groups": len(group_ids),
        "group_ids": group_ids,
        "group_ids_hash": canonical_hash(group_ids),
        "seed": seed,
        "n_resamples": n_resamples,
        "bootstrap_type": "paired_equal_weight_group_percentile",
        "uncertainty_kind": "group_sampling_only_fixed_trained_models",
        "training_seed_uncertainty_included": False,
        "independent_groups_verified": False,
        "small_group_warning": len(group_ids) < 20,
        "estimates": estimates,
    }


def summarize_seed_metrics(reports, *, metric):
    """Separate training-seed SD from any within-seed group-bootstrap interval."""
    _text(metric, "metric")
    values, seeds = [], []
    for report in reports:
        if not isinstance(report, Mapping):
            raise ValueError("Seed reports must be objects.")
        seed = _integer(report.get("seed"), "seed")
        if seed in seeds:
            raise ValueError("Repeated training seed.")
        seeds.append(seed)
        values.append(_finite(report.get(metric), metric))
    if not values:
        raise ValueError("At least one seed report is required.")
    return {
        "metric": metric,
        "num_training_seeds": len(values),
        "seeds": seeds,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "uncertainty_kind": "training_seed_sample_sd",
        "sample_confidence_interval": None,
        "small_seed_warning": len(values) < 5,
    }


def pareto_frontier(points, *, error_key="error", cost_key="declared_cost"):
    """Return all nondominated points, preserving exact ties and extra fields."""
    checked = []
    for point in points:
        if not isinstance(point, Mapping):
            raise ValueError("Pareto points must be objects.")
        canonical_hash(dict(point))
        error = _finite(point.get(error_key), error_key, 0)
        if error > 1:
            raise ValueError("Error must be a fraction in [0,1].")
        cost = _finite(point.get(cost_key), cost_key, 0)
        checked.append((dict(point), error, cost))
    frontier = []
    for point, error, cost in checked:
        dominated = any(
            other_error <= error and other_cost <= cost
            and (other_error < error or other_cost < cost)
            for _, other_error, other_cost in checked
        )
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda point: (point[cost_key], point[error_key]))


def freeze_policy_manifest(policies, *, frozen_at=None):
    """Create the entire finite candidate family before accessing calibration.

    A timestamp/hash cannot prove chronology; callers must retain this artifact
    before calibration and pass its externally retained hash to certification.
    """
    if isinstance(policies, (str, bytes, Mapping)):
        raise ValueError("policies must be a list of policy_id/system_hash objects.")
    checked, seen = [], set()
    for policy in policies:
        if not isinstance(policy, Mapping) or set(policy) != {"policy_id", "system_hash"}:
            raise ValueError("Each policy must contain exactly policy_id and system_hash.")
        policy_id = _text(policy["policy_id"], "policy_id")
        if policy_id in seen:
            raise ValueError("Duplicate policy_id.")
        seen.add(policy_id)
        checked.append({"policy_id": policy_id, "system_hash": _hash(policy["system_hash"], "system_hash")})
    if not checked:
        raise ValueError("The frozen family cannot be empty.")
    if frozen_at is None:
        frozen_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _text(frozen_at, "frozen_at")
    try:
        instant = datetime.datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("frozen_at must be a timezone-aware ISO timestamp.") from exc
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("frozen_at must specify a timezone.")
    manifest = {
        "schema_version": 1,
        "kind": "frozen_complete_policy_family",
        "frozen_at": frozen_at,
        "policies": sorted(checked, key=lambda policy: policy["policy_id"]),
        "num_policies": len(checked),
        "hash_scope": "complete_system_hashes_including_responder_head_policy_thresholds_failure_handling",
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    return manifest


def _validate_manifest(manifest, expected_manifest_hash):
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be an object.")
    _hash(expected_manifest_hash, "expected_manifest_hash")
    reconstructed = freeze_policy_manifest(
        manifest.get("policies", []), frozen_at=manifest.get("frozen_at")
    )
    if dict(manifest) != reconstructed:
        raise ValueError("Frozen manifest content/hash mismatch.")
    if reconstructed["manifest_hash"] != expected_manifest_hash:
        raise ValueError("Manifest differs from the externally retained frozen hash.")
    return reconstructed


def certify_policies(calibration_traces, manifest, *, alpha, delta, expected_manifest_hash, assumptions=None):
    """T5 Hoeffding+union bound for a pre-frozen complete policy family.

    Every candidate must have all the same calibration cases. The independent
    unit is a group, whose loss is its within-group mean terminal 0-1 error.
    IID/deployment assumptions cannot be inferred from group-disjoint IDs.
    """
    manifest = _validate_manifest(manifest, expected_manifest_hash)
    alpha, delta = _finite(alpha, "alpha"), _finite(delta, "delta")
    if not 0 <= alpha <= 1 or not 0 < delta < 1:
        raise ValueError("Require alpha in [0,1] and delta in (0,1).")
    names = {
        "independent_groups", "deployment_distribution_match",
        "family_frozen_before_calibration", "same_rollout_mechanism",
    }
    assumptions = {} if assumptions is None else assumptions
    if not isinstance(assumptions, Mapping) or set(assumptions).difference(names):
        raise ValueError("Unknown certification assumption flag.")
    if any(not isinstance(value, bool) for value in assumptions.values()):
        raise ValueError("Certification assumptions must be explicit booleans.")
    flags = {name: assumptions.get(name, False) for name in sorted(names)}
    expected = {policy["policy_id"]: policy["system_hash"] for policy in manifest["policies"]}
    by_policy = defaultdict(list)
    for source in calibration_traces:
        if not isinstance(source, Mapping):
            raise ValueError("Calibration traces must be objects.")
        policy_id = source.get("policy_id", source.get("method"))
        _text(policy_id, "policy_id")
        if policy_id not in expected:
            raise ValueError("Calibration includes an unregistered policy: {}".format(policy_id))
        if source.get("system_hash") != expected[policy_id]:
            raise ValueError("Complete system hash mismatch for {}".format(policy_id))
        if source.get("split") != "calibration":
            raise ValueError("Certification only accepts split='calibration', never test/validation.")
        by_policy[policy_id].append(source)
    if set(by_policy) != set(expected):
        raise ValueError("Every frozen candidate must be included; no post-hoc policy omission.")
    validated, group_vectors = {}, {}
    reference = None
    for policy_id in sorted(expected):
        rows = _validate_traces(by_policy[policy_id])
        if reference is not None:
            _align(reference, rows)
        else:
            reference = rows
        validated[policy_id] = rows
        group_vectors[policy_id] = _group_vectors(rows)
    group_ids = sorted(next(iter(group_vectors.values())))
    if any(set(values) != set(group_ids) for values in group_vectors.values()):
        raise ValueError("All frozen policies require exactly the same calibration groups.")
    n, m = len(group_ids), len(expected)
    radius = math.sqrt((math.log(m) - math.log(delta)) / (2 * n))
    results = []
    for policy_id in sorted(expected):
        groups = group_vectors[policy_id]
        empirical_risk = statistics.mean(group["error"] for group in groups.values())
        upper = min(1.0, empirical_risk + radius)
        results.append({
            "policy_id": policy_id,
            "system_hash": expected[policy_id],
            "empirical_group_mean_risk": empirical_risk,
            "upper_risk_bound": upper,
            "numerically_passes_bound": upper <= alpha,
            "group_mean_declared_cost": statistics.mean(group["declared_cost"] for group in groups.values()),
        })
    passing = [row for row in results if row["numerically_passes_bound"]]
    assumptions_attested = all(flags.values())
    if not passing:
        status = "NO_CERTIFICATE"
    elif not assumptions_attested:
        status = "ASSUMPTIONS_UNVERIFIED"
    else:
        status = "CONDITIONAL_CERTIFICATE"
    passing = sorted(passing, key=lambda row: (row["group_mean_declared_cost"], row["policy_id"]))
    certified_ids = [row["policy_id"] for row in passing] if assumptions_attested else []
    return {
        "schema_version": 1,
        "status": status,
        "manifest_hash": manifest["manifest_hash"],
        "num_policies_M": m,
        "num_independent_units_assumed_n": n,
        "num_samples_per_policy": len(reference),
        "group_ids": group_ids,
        "group_ids_hash": canonical_hash(group_ids),
        "alpha": alpha,
        "delta": delta,
        "hoeffding_radius": radius,
        "risk_estimand": "equal_weight_group_mean_terminal_0_1_loss",
        "guarantee_scope": "conditional_on_attested_iid_deployment_and_prefrozen_family_assumptions",
        "not_guaranteed": ["individual_case_risk", "terminal_pattern_risk", "answered_selective_risk", "distribution_shift", "clinical_safety", "optimal_population_cost"],
        "assumption_flags": flags,
        "assumptions_are_attestations_not_programmatic_proofs": True,
        "group_disjointness_implies_iid": False,
        "complete_terminal_loss_bounds_validated": True,
        "policies": results,
        "numerically_eligible_policy_ids": [row["policy_id"] for row in passing],
        "certified_policy_ids": certified_ids,
        "selected_policy_id": certified_ids[0] if certified_ids else None,
        "selection_rule": "lowest_calibration_group_mean_declared_cost_among_certified",
        "fallback_certified": False,
    }

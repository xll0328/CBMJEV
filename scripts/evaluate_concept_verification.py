"""M1 development-only verification evaluation; never opens a confirmation role.

Without an independently measured cost profile this produces lambda=0 accuracy
diagnostics only. Cached source replay is not a deployment latency measurement.
All outputs are private: aggregate summaries, frozen recipes, model states, and
sample-level outcome records without raw text. No significance tests are run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, fresh_dir, read_json, read_jsonl, write_json
from cbmjev.verification import STOP
from cbmjev.verification_experiment import (SPLIT_SEED, FusionLookup, evaluate_actions,
    group_holdout, hypothetical_predictions, prepare_final_refit, prepare_head_training,
    selected_event_metrics)
from cbmjev.verification_inputs import PROTECTED_ROLES, SOURCE_ROLES, load_verification_inputs
from cbmjev.verification_learning import (SELECTOR_CONFIGS, TASK_CONFIGS, fit_gain_selector,
    fit_joint_selector, fit_task_head, refit_task_head)


LAMBDAS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)
POLICIES = ("stop", "fixed", "fixed_stop", "direct_gain", "joint", "same_q_product", "independent_factorized")


def _source_audit(metadata, members, prediction_path, *, required):
    if metadata is None:
        if required:
            raise ValueError("source metadata with audited fit IDs/groups is required for both A and B")
        return {"status": "SOURCE_FIT_PROVENANCE_UNVERIFIED", "fit_group_ids": []}
    groups = set(metadata.get("fit_group_ids", []))
    ids = metadata.get("fit_sample_ids")
    provenance = metadata.get("provenance", [])
    declared_counts = []
    for record in provenance if isinstance(provenance, list) else ():
        groups.update(record.get("supervised_group_ids", []))
        details = record.get("metadata", {})
        if details.get("role") in SOURCE_ROLES and type(details.get("sample_count")) is int:
            declared_counts.append(details["sample_count"])
    if ids is not None:
        if not ids or len(ids) != len(set(ids)) or any(sid not in members for sid in ids):
            raise ValueError("source fit sample IDs are empty, duplicated, or absent from membership")
        if any(members[sid]["split"] not in SOURCE_ROLES for sid in ids):
            raise ValueError("source fit sample IDs include a non-source role")
        actual_groups = {members[sid]["group_id"] for sid in ids}
        if groups and groups != actual_groups:
            raise ValueError("source fit sample IDs and group IDs disagree")
        groups = actual_groups
    group_roles = {row["group_id"]: row["split"] for row in members.values()}
    if not groups or any(group_roles.get(group) not in SOURCE_ROLES for group in groups):
        raise ValueError("source fitting groups must belong exclusively to source-role membership")
    if metadata.get("predictions_sha256") not in (None, file_hash(prediction_path)):
        raise ValueError("source prediction hash differs from its manifest")
    count = len(ids) if ids is not None else declared_counts[0] if len(declared_counts) == 1 else None
    if count is not None and not 0 < count <= sum(row["group_id"] in groups for row in members.values()):
        raise ValueError("declared source sample count exceeds its fitting-group membership")
    return {"status": "SOURCE_IDS_AUDITED" if ids is not None else "SOURCE_GROUPS_AUDITED",
            "fit_sample_count": count, "exact_fit_sample_ids_recorded": ids is not None,
            "fit_sample_ids_sha256": stable_hash(sorted(ids)) if ids is not None else None,
            "fit_group_ids": sorted(groups), "fit_roles": sorted({group_roles[g] for g in groups}),
            "source_artifact_hashes": {key: value for key, value in metadata.items()
                                      if key.endswith("sha256") and isinstance(value, str)}}


def _cost_profile(path, k):
    if path is None:
        return None
    profile = read_json(path)
    if profile.get("format") != "verification-cost-v1" or profile.get("units") != "seconds_per_sample":
        raise ValueError("cost profile must be verification-cost-v1 in seconds_per_sample")
    if profile.get("validation_status") != "VALIDATED":
        raise ValueError("cost profile requires VALIDATED cache/live agreement")
    context = profile.get("context", {})
    if (any(not isinstance(context.get(key), str) or not context[key] for key in ("platform", "isolation_scope"))
            or any(type(context.get(key)) is not int or context[key] < 1 for key in ("batch_size", "n"))):
        raise ValueError("cost profile needs platform, batch_size, n, and explicit isolation_scope")
    normalizer = profile.get("normalization", {})
    if normalizer.get("reference") != "shared_strong_isolated":
        raise ValueError("cost normalization reference must be shared_strong_isolated")
    scalars = [profile.get(key) for key in ("stop_seconds", "all_fused_seconds", "raw_b_seconds", "shared_strong_seconds")]
    scalars += [normalizer.get("seconds")]
    singletons = profile.get("singleton_seconds", [])
    overheads = profile.get("policy_seconds", {})
    if len(singletons) != k or any(key not in overheads for key in POLICIES):
        raise ValueError("cost profile needs each measured singleton and policy overhead")
    values = scalars + list(singletons) + [overheads[key] for key in POLICIES]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in values):
        raise ValueError("measured costs must be finite nonnegative numbers")
    if normalizer["seconds"] <= 0 or not np.isclose(normalizer["seconds"], profile["shared_strong_seconds"]):
        raise ValueError("normalizer must equal the positive isolated shared-strong mean")
    if any(value < profile["stop_seconds"] for value in singletons):
        raise ValueError("singleton totals below STOP require an explicit measurement audit")
    return profile


def _incremental_costs(profile, k):
    return (np.zeros(k) if profile is None else
            (np.asarray(profile["singleton_seconds"]) - profile["stop_seconds"]) / profile["normalization"]["seconds"])


def _row_costs(profile, method, columns):
    if profile is None or method == "hindsight":
        return None
    if method in ("all_fused", "raw_b", "shared_strong"):
        seconds = np.full(len(columns), profile[method + "_seconds"])
    else:
        base = np.asarray([profile["stop_seconds"]] + profile["singleton_seconds"])
        seconds = base[columns] + profile["policy_seconds"][method]
    return seconds / profile["normalization"]["seconds"]


def choose_columns(gains, incremental_costs, cost_weight):
    """A-only gain estimates -> STOP(0) or singleton(1+j); STOP wins ties."""
    gains = np.asarray(gains, dtype=float)
    costs = np.asarray(incremental_costs, dtype=float)
    if gains.ndim != 2 or costs.shape != (gains.shape[1],) or not np.isfinite(gains).all():
        raise ValueError("gains/costs have invalid shapes or values")
    if not np.isfinite(costs).all() or (costs < 0).any() or not np.isfinite(cost_weight) or cost_weight < 0:
        raise ValueError("costs and cost weight must be finite and nonnegative")
    return np.column_stack((np.zeros(len(gains)), gains - cost_weight * costs)).argmax(axis=1)


def _joint_gains(joints, before, hypothetical, *, product=False):
    if len(joints) != len(hypothetical) or not joints:
        raise ValueError("one joint distribution is required per hypothetical action")
    results, common_y = [], None
    for joint, after in zip(joints, hypothetical):
        q = np.asarray(joint, dtype=float)
        if q.ndim != 3 or q.shape[0] != len(before) or after.shape != (len(before), q.shape[2]):
            raise ValueError("joint and hypothetical prediction shapes differ")
        if not np.isfinite(q).all() or (q < 0).any() or not np.allclose(q.sum(axis=(1, 2)), 1, atol=1e-6):
            raise ValueError("joint predictions must be normalized finite probabilities")
        q = q / q.sum(axis=(1, 2), keepdims=True)
        qy = q.sum(axis=2)
        if common_y is not None and not np.allclose(common_y, qy, atol=1e-6):
            raise ValueError("joint selectors must share their task marginal")
        common_y = qy
        if product:
            q = qy[:, :, None] * q.sum(axis=1)[:, None, :]
        rows, responses = np.arange(len(q))[:, None], np.arange(q.shape[2])[None, :]
        # Per-response subtraction makes unchanged predictions exactly zero,
        # avoiding tiny artificial gains from float32 normalization roundoff.
        results.append((q[rows, after, responses] - q[np.arange(len(q)), before, :]).sum(axis=1))
    return np.column_stack(results)


def runtime_gains(initial_a, lookup, task_head, direct, joint, factorized):
    """Deployment-facing boundary: no B, Y, C, text, IDs, or action outcomes."""
    before = task_head.predict_proba(lookup.initial(initial_a)).argmax(axis=1)
    hypothetical = hypothetical_predictions(initial_a, lookup, task_head)
    q = joint.predict_joint(initial_a)
    return {"direct_gain": direct.predict_gains(initial_a),
            "joint": _joint_gains(q, before, hypothetical),
            "same_q_product": _joint_gains(q, before, hypothetical, product=True),
            "independent_factorized": _joint_gains(factorized.predict_joint(initial_a), before, hypothetical)}


def _classification(probabilities, labels):
    predictions = probabilities.argmax(axis=1)
    classes = probabilities.shape[1]
    confusion = np.zeros((classes, classes), dtype=int)
    np.add.at(confusion, (labels, predictions), 1)
    tp = np.diag(confusion)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(classes), where=denominator > 0)
    return {"samples": len(labels), "error": float(np.mean(predictions != labels)),
            "accuracy": float(np.mean(predictions == labels)), "macro_f1": float(f1.mean()),
            "nll": float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-30)).mean()),
            "brier": float(np.square(probabilities - np.eye(classes)[labels]).sum(axis=1).mean()),
            "confusion_matrix": confusion.tolist(), "macro_f1_includes_all_native_classes": True}


def _method_record(data, table, name, columns, probabilities, method, lam, profile, handle, role):
    probabilities = np.asarray(probabilities, dtype=float)
    prediction = probabilities.argmax(axis=1)
    result = _classification(probabilities, data.Y)
    result.update(groups=len(set(data.group_ids)), diagnostic=method in ("raw_b", "hindsight"),
                  independent_task_head=method == "shared_strong")
    costs = _row_costs(profile, method, columns)
    result["mean_normalized_cost"] = None if costs is None else float(costs.mean())
    result["J"] = None if costs is None else result["error"] + lam * float(costs.mean())
    singleton = method not in ("all_fused", "raw_b", "shared_strong")
    if singleton:
        actions = tuple(STOP if c == 0 else int(c - 1) for c in columns)
        result["events"] = selected_event_metrics(data, table, actions)
        result["mean_queries"] = float(np.mean(columns > 0))
    else:
        result["events"] = None
        result["mean_queries"] = None if method == "shared_strong" else len(data.categories)
    for i, sid in enumerate(data.sample_ids):
        j = int(columns[i] - 1) if singleton and columns[i] else None
        old = int(table.concepts[i, 0, j]) if j is not None else None
        new = int(table.concepts[i, columns[i], j]) if j is not None else None
        gold = data.C[i, j] if j is not None else None
        gain = int(table.predictions[i, 0] != data.Y[i]) - int(prediction[i] != data.Y[i])
        event = {"sample_id": sid, "group_id": data.group_ids[i], "role": role, "method": name,
                 "lambda": lam, "y": int(data.Y[i]), "before_prediction": None if method == "shared_strong" else int(table.predictions[i, 0]),
                 "prediction": int(prediction[i]), "probabilities": probabilities[i].tolist(),
                 "action": (STOP if columns[i] == 0 else j) if singleton else method,
                 "task_event": None if method == "shared_strong" else {1: "repair", -1: "damage", 0: "unchanged"}[gain],
                 "concept_before": old, "concept_after": new,
                 "concept_gold": None if gold is None else int(gold),
                 "concept_event": None if j is None else "missing" if gold is None else
                     "repair" if old != gold and new == gold else "damage" if old == gold and new != gold else "unchanged",
                 "normalized_cost": None if costs is None else float(costs[i])}
        handle.write(json.dumps(event, allow_nan=False) + "\n")
    return result


def run_development(*, prepared, a_predictions, b_predictions, b_metadata, output,
                    seed=40, a_metadata=None, legacy_a=False, cost_profile=None, synthetic=False):
    """Run M1 only. Synthetic overrides are a test helper, deliberately absent from CLI."""
    if type(seed) is not int or seed not in (40, 41, 42) or not isinstance(synthetic, bool):
        raise ValueError("seed must be 40/41/42 and synthetic must be boolean")
    prepared = Path(prepared)
    membership = read_jsonl(prepared / "membership.jsonl")
    present = {row["split"] for row in membership}
    roles = ("head_fit", "policy_fit", "policy_tune") if "policy_tune" in present else ("head_fit", "policy_fit", "validation")
    inputs = load_verification_inputs(prepared, a_predictions, b_predictions, roles=roles,
                                     a_metadata=a_metadata, b_metadata=b_metadata, legacy_a=legacy_a)
    if set(inputs.rows) & (PROTECTED_ROLES | SOURCE_ROLES) or inputs.report["protected_override"]:
        raise ValueError("M1 evaluator refuses protected/source evaluation populations")
    members = {row["sample_id"]: row for row in membership}
    a_meta, b_meta = read_json(a_metadata) if a_metadata else None, read_json(b_metadata)
    sources = {"A": _source_audit(a_meta, members, a_predictions, required=True),
               "B": _source_audit(b_meta, members, b_predictions, required=True)}
    if sources["A"]["fit_group_ids"] and sources["A"]["fit_group_ids"] != sources["B"]["fit_group_ids"]:
        raise ValueError("A and B must use the same source fitting groups")
    if (sources["A"].get("fit_sample_ids_sha256") and sources["B"].get("fit_sample_ids_sha256")
            and sources["A"]["fit_sample_ids_sha256"] != sources["B"]["fit_sample_ids_sha256"]):
        raise ValueError("A and B must use the same source fitting samples")
    if (sources["A"].get("fit_sample_count") is not None and sources["B"].get("fit_sample_count") is not None
            and sources["A"]["fit_sample_count"] != sources["B"]["fit_sample_count"]):
        raise ValueError("A and B source fitting sample counts differ")
    for meta in (a_meta, b_meta):
        if meta:
            binding = meta.get("source_training_membership_sha256", (meta.get("fit_provenance") or {}).get("membership_sha256"))
            if binding is not None and binding != inputs.report["membership_sha256"]:
                raise ValueError("source training membership hash mismatch")
    profile = _cost_profile(cost_profile, inputs.schema.num_atoms)
    lambdas, selection_lambda = (LAMBDAS, .02) if profile else ((0.0,), 0.0)
    head_data, policy = inputs.data("head_fit"), inputs.data("policy_fit")
    if "policy_tune" in roles:
        train, tune = policy, inputs.data("policy_tune")
        populations = {"policy_tune_development": (tune, inputs.shared_concepts("policy_tune"))}
    else:
        train_ids, tune_ids = group_holdout(policy.group_ids)
        train, tune = policy.subset(train_ids), policy.subset(tune_ids)
        populations = {"policy_tune_development": (tune, inputs.shared_concepts("policy_fit")[tune_ids]),
                       "validation_seen_development": (inputs.data("validation"), inputs.shared_concepts("validation"))}
    out = fresh_dir(output)
    freeze = {"protocol": "M1_DEVELOPMENT_ONLY", "seed": seed, "split_seed": SPLIT_SEED,
              "inputs": inputs.report, "sources": sources, "schema": inputs.schema.to_dict(),
              "code_sha256": {name: file_hash(Path(__file__).resolve().parents[1] / name) for name in
                              ("scripts/evaluate_concept_verification.py", "cbmjev/verification_learning.py",
                               "cbmjev/verification_experiment.py", "cbmjev/verification_inputs.py", "cbmjev/verification.py")},
              "samples_sha256": file_hash(prepared / "samples.jsonl"),
              "cost_profile_sha256": file_hash(cost_profile) if cost_profile else None,
              "cost_profile": profile,
              "cost_status": "MEASURED_PROFILE" if profile else "COST_NOT_MEASURED",
              "lambdas": list(lambdas), "selection_lambda": selection_lambda,
              "policy_train_ids": list(train.sample_ids), "policy_tune_ids": list(tune.sample_ids),
              "selector_input": "one-hot hard initial A only", "task_input": "one-hot hard fused concepts only",
              "task_configs": [vars(c) for c in TASK_CONFIGS], "task_epochs": 50,
              "selector_configs": [vars(c) for c in SELECTOR_CONFIGS], "selector_epochs": 100,
              "architectures": {"task_hidden": 128, "task_dropout": .1, "selector_hidden": 64,
                                "joint": "shared qY plus Y-specific response weights", "gain": "unconstrained signed MSE"},
              "synthetic": synthetic, "synthetic_override": {"epochs": 2, "configs": [[.001, 0]], "hidden": 8} if synthetic else None,
              "evidence": "development_not_primary_no_confirmatory_claims",
              "baseline_selection": "fixed j and fixed j+STOP each chosen once on policy-tune at selection_lambda"}
    write_json(out / "freeze.json", freeze)
    options = {"epochs": 2, "configs": [(1e-3, 0)], "hidden": 8, "synthetic": True} if synthetic else {}
    common = {"category_counts": inputs.schema.value_counts, "num_classes": inputs.schema.num_classes, "seed": seed}
    prep = prepare_head_training(head_data)
    selected = fit_task_head(prep.train.concepts, prep.train.labels, prep.tune.concepts, prep.tune.labels,
                             train_weights=prep.train.weights, tune_weights=prep.tune.weights, **common, **options)
    fusion, states, final_log = prepare_final_refit(head_data, prep.selection.method, prep.selection.alpha)
    head = refit_task_head(states.concepts, states.labels, selected=selected, seed=seed, weights=states.weights)
    lookup = FusionLookup(fusion)
    shared = inputs.shared_concepts("head_fit")
    selected_shared = fit_task_head(shared[prep.train_indices], head_data.Y[prep.train_indices],
                                    shared[prep.tune_indices], head_data.Y[prep.tune_indices], **common, **options)
    strong = refit_task_head(shared, head_data.Y, selected=selected_shared, seed=seed)
    train_table, tune_table = evaluate_actions(train, lookup, head), evaluate_actions(tune, lookup, head)
    direct = fit_gain_selector(train.A, train_table.gains, tune.A, tune_table.gains,
                               category_counts=inputs.schema.value_counts, seed=seed, **options)
    joint = fit_joint_selector(train.A, train.Y, train.B, tune.A, tune.Y, tune.B, **common, **options)
    factorized = fit_joint_selector(train.A, train.Y, train.B, tune.A, tune.Y, tune.B,
                                    factorized=True, **common, **options)
    models = {"task_head": head, "shared_strong_task_head": strong, "direct_gain": direct,
              "joint": joint, "independent_factorized": factorized}
    for name, model in models.items():
        model.save(out / (name + ".pt"))
    write_json(out / "fusion_lookup.json", {"method": fusion.method, "alpha": fusion.alpha,
        "category_counts": list(lookup.categories), "probabilities": [p.tolist() for p in lookup.probabilities]})
    write_json(out / "training.json", {"head_preparation": prep.log, "head_final_refit": final_log,
        "models": {name: model.report for name, model in models.items()}})
    k, n = len(train.categories), len(tune)
    incremental = _incremental_costs(profile, k)
    train_mean = train_table.gains.mean(axis=0)
    fixed_scores, stopped_scores = [], []
    for j in range(k):
        for use_stop, scores in ((False, fixed_scores), (True, stopped_scores)):
            column = j + 1 if not use_stop or train_mean[j] > selection_lambda * incremental[j] else 0
            columns = np.full(n, column, dtype=int)
            score = float(np.mean(tune_table.predictions[:, column] != tune.Y))
            costs = _row_costs(profile, "fixed_stop" if use_stop else "fixed", columns)
            scores.append(score + (selection_lambda * float(costs.mean()) if costs is not None else 0))
    chosen_fixed, chosen_stop = int(np.argmin(fixed_scores)), int(np.argmin(stopped_scores))
    selection = {"selection_lambda": selection_lambda, "best_fixed_j": chosen_fixed,
                 "best_fixed_stop_j": chosen_stop, "all_fixed_tune_scores": fixed_scores,
                 "all_fixed_stop_tune_scores": stopped_scores, "train_mean_signed_gain": train_mean.tolist()}
    write_json(out / "selection.json", selection)
    write_json(out / "frozen_models.json", {"models_sha256": {name: file_hash(out / (name + ".pt")) for name in models},
        "selection_sha256": file_hash(out / "selection.json"), "freeze_sha256": file_hash(out / "freeze.json"),
        "fusion_lookup_sha256": file_hash(out / "fusion_lookup.json"),
        "evaluation_populations": list(populations)})
    results = {}
    with (out / "row_events.jsonl").open("x", encoding="utf-8") as handle:
        for role, (data, shared_values) in populations.items():
            gains = runtime_gains(data.A, lookup, head, direct, joint, factorized)
            table = evaluate_actions(data, lookup, head)
            raw_b, shared_p = head.predict_proba(data.B), strong.predict_proba(shared_values)
            results[role] = {}
            for lam in lambdas:
                choices = {"stop": ("stop", np.zeros(len(data), dtype=int)),
                           **{"fixed_" + str(j): ("fixed", np.full(len(data), j + 1, dtype=int)) for j in range(k)},
                           "best_fixed": ("fixed", np.full(len(data), chosen_fixed + 1, dtype=int)),
                           "fixed_stop": ("fixed_stop", np.full(len(data), chosen_stop + 1 if
                               train_mean[chosen_stop] > lam * incremental[chosen_stop] else 0, dtype=int)),
                           **{name: (name, choose_columns(g, incremental, lam)) for name, g in gains.items()},
                           "hindsight": ("hindsight", choose_columns(table.gains, incremental, lam))}
                current = {}
                for name, (method, columns) in choices.items():
                    current[name] = _method_record(data, table, name, columns,
                        table.probabilities[np.arange(len(data)), columns], method, lam, profile, handle, role)
                for name, probabilities in (("all_fused", table.probabilities[:, -1]), ("raw_b", raw_b), ("shared_strong", shared_p)):
                    current[name] = _method_record(data, table, name, np.full(len(data), k + 1),
                                                   probabilities, name, lam, profile, handle, role)
                results[role][str(lam)] = current
    report = {"protocol": "M1_DEVELOPMENT_ONLY", "synthetic": synthetic,
              "evidence_status": "development_not_primary_noJclaims" if profile is None else "development_cost_profile_conditional",
              "cost_status": freeze["cost_status"], "freeze_sha256": file_hash(out / "freeze.json"),
              "source_provenance": {key: value["status"] for key, value in sources.items()},
              "selection": selection, "results": results, "artifact_visibility": "PRIVATE_NO_RAW_TEXT",
              "confirmation_evaluated": False, "p_values": None,
              "interpretation": "policy-tune was used for selection; historical validation is seen development; shared_strong has its own task head"}
    write_json(out / "report.json", report)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "a-predictions", "b-predictions", "a-metadata", "b-metadata", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--legacy-a", action="store_true", help="require the hash-bound historical A cache manifest")
    parser.add_argument("--seed", type=int, choices=(40, 41, 42), default=40)
    parser.add_argument("--cost-profile", help="measured verification-cost-v1 JSON; omission restricts diagnostics to lambda=0")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = run_development(**vars(args))
    print(json.dumps({"output": args.output, "evidence_status": report["evidence_status"],
                      "cost_status": report["cost_status"]}, allow_nan=False))


if __name__ == "__main__":
    main()

"""Synthetic statistical/provenance checks; not research-result evidence."""

import copy
import math
import unittest

from cbmjev.evaluation import (
    certify_policies, freeze_policy_manifest, paired_group_bootstrap,
    pareto_frontier, summarize_seed_metrics, summarize_traces,
)
from cbmjev.provenance import (
    canonical_hash, make_fit_record, plan_nested_crossfit,
    validate_provenance_dag, validate_target_exclusion,
)


def trace(sample, group, *, method="a", y=0, prediction=0, cost=1.0, mode="offline_replay", split="test"):
    row = {
        "sample_id": sample, "group_id": group, "split": split, "method": method,
        "y": y, "prediction": prediction, "queried_groups": ["group0"],
        "queried_atoms": ["atom0", "atom1"], "calls": 1, "declared_cost": cost,
        "mode": mode, "steps": [],
    }
    if mode == "live":
        row["total_wall_ms"] = 10.0
    return row


ASSUMPTIONS = {
    "independent_groups": True, "deployment_distribution_match": True,
    "family_frozen_before_calibration": True, "same_rollout_mechanism": True,
}


def calibration_fixture(n=10, m=2, rows_per_group=1):
    policies = [
        {"policy_id": "policy{}".format(i), "system_hash": canonical_hash({"system": i})}
        for i in range(m)
    ]
    manifest = freeze_policy_manifest(policies, frozen_at="2026-09-22T08:00:00+08:00")
    rows = []
    for policy_index, policy in enumerate(policies):
        for group in range(n):
            for member in range(rows_per_group):
                row = trace(
                    "s{}-{}".format(group, member), "g{}".format(group),
                    method=policy["policy_id"], split="calibration", cost=1.0 + policy_index,
                )
                row.update(policy)
                rows.append(row)
    return rows, manifest


def certify(rows, manifest, **kwargs):
    options = {
        "alpha": 1.0, "delta": 0.05,
        "expected_manifest_hash": manifest["manifest_hash"], "assumptions": ASSUMPTIONS,
    }
    options.update(kwargs)
    return certify_policies(rows, manifest, **options)


class SummaryTests(unittest.TestCase):
    def test_fixed_class_macro_f1_and_group_estimand(self):
        rows = [
            trace("a", "large", y=0, prediction=0, cost=1),
            trace("b", "large", y=1, prediction=0, cost=3),
            trace("c", "small", y=1, prediction=1, cost=2),
        ]
        result = summarize_traces(rows, num_classes=3)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 4 / 9)
        self.assertAlmostEqual(result["group_mean_risk"], 0.25)
        self.assertAlmostEqual(result["error"], 1 / 3)
        self.assertEqual(result["num_classes"], 3)
        self.assertEqual(result["total_queried_groups"], 3)
        self.assertEqual(result["total_queried_atoms"], 6)
        self.assertEqual(result["mean_declared_cost"], 2)
        self.assertEqual(result["confusion_matrix_true_rows"], [[1, 0, 0], [1, 1, 0], [0, 0, 0]])

    def test_replay_never_reports_deployment_latency(self):
        row = trace("s", "g")
        row["total_wall_ms"] = 0.01
        result = summarize_traces([row], num_classes=2)
        self.assertIsNone(result["deployment_latency"])
        self.assertEqual(result["ignored_replay_wall_measurements"], 1)
        self.assertNotIn("p50", result)

    def test_live_latency_quantiles(self):
        rows = [trace(str(i), str(i), mode="live") for i in range(3)]
        for row, latency in zip(rows, (10, 30, 20)):
            row["total_wall_ms"] = latency
        result = summarize_traces(rows, num_classes=2)
        self.assertEqual(result["deployment_latency"]["p50"], 20)
        self.assertAlmostEqual(result["deployment_latency"]["p95"], 29)

    def test_counts_accept_integers(self):
        row = trace("s", "g")
        row["queried_groups"], row["queried_atoms"] = 0, 0
        self.assertEqual(summarize_traces([row], num_classes=2)["total_queried_atoms"], 0)

    def test_live_requires_measured_wall_time(self):
        row = trace("s", "g", mode="live")
        del row["total_wall_ms"]
        with self.assertRaises(ValueError):
            summarize_traces([row], num_classes=2)

    def test_reject_duplicate_sample(self):
        with self.assertRaises(ValueError):
            summarize_traces([trace("s", "g"), trace("s", "g")], num_classes=2)

    def test_aggregate_overflow_rejected(self):
        with self.assertRaises(ValueError):
            summarize_traces(
                [trace("a", "a", cost=1e308), trace("b", "b", cost=1e308)], num_classes=2
            )

    def test_reject_nonfinite_and_invalid_fields(self):
        for field, value in (
            ("declared_cost", float("nan")), ("calls", -1), ("y", 2),
            ("prediction", True), ("total_wall_ms", float("inf")),
            ("queried_groups", ["q", "q"]), ("steps", [{"value": float("nan")}]),
        ):
            with self.subTest(field=field, value=value):
                row = trace("s", "g")
                row[field] = value
                with self.assertRaises(ValueError):
                    summarize_traces([row], num_classes=2)

    def test_reject_mixed_system_split_or_mode(self):
        for field, value in (("method", "b"), ("split", "validation"), ("mode", "live")):
            with self.subTest(field=field):
                rows = [trace("a", "a"), trace("b", "b")]
                rows[1][field] = value
                rows[1]["total_wall_ms"] = 1.0
                with self.assertRaises(ValueError):
                    summarize_traces(rows, num_classes=2)


class BootstrapAndParetoTests(unittest.TestCase):
    def test_constant_paired_effect(self):
        a = [trace(str(i), "g{}".format(i // 2), prediction=0, cost=1) for i in range(6)]
        b = [trace(str(i), "g{}".format(i // 2), method="b", prediction=1, cost=3) for i in range(6)]
        result = paired_group_bootstrap(a, b, seed=17, n_resamples=100)
        self.assertEqual(result["num_groups"], 3)
        self.assertEqual(result["estimates"]["error"]["difference_a_minus_b"], -1)
        self.assertEqual(result["estimates"]["error"]["ci_lower"], -1)
        self.assertEqual(result["estimates"]["declared_cost"]["ci_upper"], -2)
        self.assertFalse(result["training_seed_uncertainty_included"])

    def test_bootstrap_reproducible_and_case_order_invariant(self):
        a = [trace(str(i), "g{}".format(i // 2), prediction=i % 2) for i in range(8)]
        b = [trace(str(i), "g{}".format(i // 2), method="b", prediction=0) for i in range(8)]
        first = paired_group_bootstrap(a, b, seed=19, n_resamples=30)
        second = paired_group_bootstrap(list(reversed(a)), b, seed=19, n_resamples=30)
        self.assertEqual(first, second)

    def test_group_weights_not_row_weights(self):
        a = [trace("a", "large", prediction=1), trace("b", "large", prediction=1), trace("c", "small")]
        b = [trace("a", "large", method="b"), trace("b", "large", method="b"), trace("c", "small", method="b")]
        result = paired_group_bootstrap(a, b, seed=1, n_resamples=20)
        self.assertEqual(result["estimates"]["error"]["difference_a_minus_b"], 0.5)

    def test_pair_alignment_is_strict(self):
        for field, value in (("sample_id", "other"), ("group_id", "other"), ("y", 1), ("split", "validation"), ("mode", "live")):
            with self.subTest(field=field):
                a, b = [trace("a", "g")], [trace("a", "g", method="b")]
                b[0][field] = value
                b[0]["total_wall_ms"] = 1
                with self.assertRaises(ValueError):
                    paired_group_bootstrap(a, b, seed=1, n_resamples=2)

    def test_seed_sd_not_bootstrap_ci(self):
        result = summarize_seed_metrics(
            [{"seed": 17, "accuracy": 0.8}, {"seed": 18, "accuracy": 0.9}, {"seed": 19, "accuracy": 1.0}],
            metric="accuracy",
        )
        self.assertAlmostEqual(result["sample_sd"], 0.1)
        self.assertIsNone(result["sample_confidence_interval"])
        single = summarize_seed_metrics([{"seed": 17, "accuracy": 0.8}], metric="accuracy")
        self.assertIsNone(single["sample_sd"])
        with self.assertRaises(ValueError):
            summarize_seed_metrics([{"seed": 17, "accuracy": 0.8}] * 2, metric="accuracy")

    def test_pareto_lower_error_and_cost_with_exact_ties(self):
        points = [
            {"id": "cheap", "error": 0.3, "declared_cost": 1},
            {"id": "accurate", "error": 0.1, "declared_cost": 3},
            {"id": "dominated", "error": 0.4, "declared_cost": 2},
            {"id": "same", "error": 0.1, "declared_cost": 3},
        ]
        self.assertEqual([row["id"] for row in pareto_frontier(points)], ["cheap", "accurate", "same"])
        with self.assertRaises(ValueError):
            pareto_frontier([{"error": float("nan"), "declared_cost": 1}])


class CertificationTests(unittest.TestCase):
    def test_formula_and_group_unit(self):
        rows, manifest = calibration_fixture(n=10, m=2, rows_per_group=3)
        result = certify(rows, manifest)
        self.assertAlmostEqual(result["hoeffding_radius"], math.sqrt(math.log(2 / 0.05) / 20))
        self.assertEqual(result["num_independent_units_assumed_n"], 10)
        self.assertEqual(result["num_samples_per_policy"], 30)
        self.assertFalse(result["group_disjointness_implies_iid"])
        self.assertEqual(result["selected_policy_id"], "policy0")
        self.assertEqual(result["status"], "CONDITIONAL_CERTIFICATE")

    def test_small_medical_sample_has_no_certificate(self):
        rows, manifest = calibration_fixture(n=203, m=20)
        result = certify(rows, manifest, alpha=0.1)
        self.assertEqual(result["status"], "NO_CERTIFICATE")
        self.assertAlmostEqual(result["hoeffding_radius"], 0.12147963549569715)
        self.assertEqual(result["certified_policy_ids"], [])
        self.assertIsNone(result["selected_policy_id"])

    def test_extreme_valid_delta_stays_finite(self):
        rows, manifest = calibration_fixture()
        result = certify(rows, manifest, delta=1e-320)
        self.assertTrue(math.isfinite(result["hoeffding_radius"]))

    def test_unattested_assumptions_do_not_grant_certificate(self):
        rows, manifest = calibration_fixture()
        result = certify(rows, manifest, assumptions={})
        self.assertEqual(result["status"], "ASSUMPTIONS_UNVERIFIED")
        self.assertTrue(result["numerically_eligible_policy_ids"])
        self.assertEqual(result["certified_policy_ids"], [])

    def test_omitted_registered_policy_rejected(self):
        rows, manifest = calibration_fixture()
        with self.assertRaisesRegex(ValueError, "omission"):
            certify([row for row in rows if row["policy_id"] == "policy0"], manifest)

    def test_manifest_tampering_and_replacement_rejected(self):
        rows, manifest = calibration_fixture()
        altered = copy.deepcopy(manifest)
        altered["policies"].pop()
        with self.assertRaises(ValueError):
            certify(rows, altered)
        replacement = freeze_policy_manifest(manifest["policies"][:1], frozen_at=manifest["frozen_at"])
        with self.assertRaises(ValueError):
            certify(rows, replacement, expected_manifest_hash=manifest["manifest_hash"])

    def test_policy_system_hash_or_unregistered_candidate_rejected(self):
        rows, manifest = calibration_fixture()
        for key, value in (("system_hash", "a" * 64), ("policy_id", "posthoc")):
            altered = copy.deepcopy(rows)
            altered[0][key] = value
            with self.assertRaises(ValueError):
                certify(altered, manifest)

    def test_noncalibration_and_nonfinite_rejected(self):
        rows, manifest = calibration_fixture()
        for key, value in (("split", "test"), ("declared_cost", float("nan")), ("prediction", float("inf"))):
            altered = copy.deepcopy(rows)
            altered[0][key] = value
            with self.assertRaises(ValueError):
                certify(altered, manifest)

    def test_policy_case_group_and_target_alignment(self):
        rows, manifest = calibration_fixture()
        for key, value in (("sample_id", "new"), ("group_id", "new"), ("y", 1)):
            altered = copy.deepcopy(rows)
            altered[-1][key] = value
            with self.assertRaises(ValueError):
                certify(altered, manifest)

    def test_nonbinary_assumption_flag_rejected(self):
        rows, manifest = calibration_fixture()
        with self.assertRaises(ValueError):
            certify(rows, manifest, assumptions={"independent_groups": "yes"})


class ProvenanceTests(unittest.TestCase):
    def test_valid_dag_stores_actual_ids(self):
        responder = make_fit_record("r", supervised_group_ids=["train2", "train1"])
        head = make_fit_record("f", supervised_group_ids=["train3"], parent_ids=["r"])
        records = [head, responder]
        self.assertEqual(validate_provenance_dag(records)["topological_order"], ["r", "f"])
        result = validate_target_exclusion(records, artifact_ids=["f"], target_group_ids=["held"])
        self.assertEqual(result["ancestor_supervised_group_ids"], ["train1", "train2", "train3"])
        self.assertEqual(responder["supervised_group_ids"], ["train1", "train2"])
        self.assertFalse(result["external_pretraining_membership_verified"])

    def test_direct_and_ancestor_label_leakage_rejected(self):
        responder = make_fit_record("r", supervised_group_ids=["held"])
        head = make_fit_record("f", supervised_group_ids=["train"], parent_ids=["r"])
        for artifact_id in ("r", "f"):
            with self.assertRaisesRegex(ValueError, "leakage"):
                validate_target_exclusion([responder, head], artifact_ids=[artifact_id], target_group_ids=["held"])

    def test_hash_tampering_rejected(self):
        record = make_fit_record("r", supervised_group_ids=["train"])
        record["supervised_group_ids"] = ["held"]
        with self.assertRaises(ValueError):
            validate_provenance_dag([record])

    def test_unknown_ancestor_and_cycle_rejected(self):
        a = make_fit_record("a", supervised_group_ids=["train"], parent_ids=["b"])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            validate_provenance_dag([a])
        b = make_fit_record("b", supervised_group_ids=["train"], parent_ids=["a"])
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_provenance_dag([a, b])

    def test_duplicate_group_or_empty_supervision_rejected(self):
        for groups in ([], ["a", "a"]):
            with self.assertRaises(ValueError):
                make_fit_record("r", supervised_group_ids=groups)
        with self.assertRaises(ValueError):
            canonical_hash({"bad": float("nan")})

    def test_external_pretraining_not_claimed_known(self):
        external = make_fit_record("base", supervised_group_ids=[], fit_kind="frozen_external")
        result = validate_target_exclusion([external], artifact_ids=["base"], target_group_ids=["held"])
        self.assertFalse(result["external_pretraining_membership_verified"])

    def test_nested_plan_not_completed_training(self):
        groups = ["g{}".format(index) for index in range(12)]
        plan = plan_nested_crossfit(groups, outer_folds=3, inner_folds=3, seed=17)
        self.assertEqual(plan["status"], "PLANNED")
        self.assertFalse(plan["has_trained_models"])
        self.assertFalse(plan["has_validated_actual_oof_predictions"])
        self.assertEqual(plan["responder_fit_job_count"], 13)
        self.assertEqual(plan["fit_job_count"], 18)
        self.assertEqual(plan, plan_nested_crossfit(list(reversed(groups)), seed=17))
        jobs = {job["job_id"]: job for job in plan["jobs"]}

        def ancestry_supervision(job_id):
            job = jobs[job_id]
            values = set(job["supervised_group_ids"])
            for parent in job["parent_job_ids"]:
                values.update(ancestry_supervision(parent))
            return values

        for job in plan["jobs"]:
            if job["kind"] == "construct_policy_targets":
                targets = set(job["target_group_ids"])
                for ancestor in job["prediction_ancestor_job_ids"]:
                    self.assertFalse(targets.intersection(ancestry_supervision(ancestor)))
                self.assertTrue(job["target_labels_used_only_after_prediction"])

    def test_nested_plan_rejects_too_few_groups(self):
        with self.assertRaises(ValueError):
            plan_nested_crossfit(["a", "b", "c"], seed=17)


if __name__ == "__main__":
    unittest.main()

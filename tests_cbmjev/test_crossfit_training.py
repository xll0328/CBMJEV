"""Synthetic checks for executable nested-OOF training primitives."""

import copy
import json
import random
import unittest

import torch

from cbmjev.contracts import Concept, QueryGroup, Schema, candidate_actions, stable_hash
from cbmjev.crossfit_training import (construct_action_targets,
                                      fit_controller_from_targets,
                                      fit_head_only)
from cbmjev.provenance import make_fit_record, validate_provenance_dag


def schema_fixture():
    return Schema(
        "crossfit-training-fixture", 2,
        tuple(Concept("c%d" % index, "concept %d" % index, ("no", "yes"))
              for index in range(3)),
        tuple(QueryGroup("g%d" % index, (index,)) for index in range(3)),
    )


def rows_fixture(prefix, count, seed):
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        z = [rng.randrange(2) for _ in range(3)]
        rows.append({"sample_id": "%s-s%d" % (prefix, index),
                     "group_id": "%s-g%d" % (prefix, index),
                     "z": z, "y": index % 2})
    return rows


CONFIG = {
    "seed": 37, "device": "cpu", "hidden": 12,
    "head_epochs": 2, "policy_epochs": 2, "batch_size": 8,
    "masks_per_sample": 2, "learning_rate": 0.01,
    "include_pairs": True, "include_all": True,
    "max_pair_actions": 2, "actions_per_state": 4,
    "class_weighting": "inverse_frequency",
}


def oof_sources(rows, prefix):
    """One synthetic producer per group, fitted on all other groups."""
    groups = sorted(row["group_id"] for row in rows)
    records, assignments = [], {}
    for group_id in groups:
        artifact_id = "%s:%s" % (prefix, group_id)
        records.append(make_fit_record(
            artifact_id,
            supervised_group_ids=[other for other in groups if other != group_id],
            metadata={"synthetic_fixture": True},
        ))
        assignments[group_id] = artifact_id
    return records, assignments


class NestedOOFTrainingPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = schema_fixture()
        cls.head_rows = rows_fixture("head", 6, 5)
        cls.target_rows = rows_fixture("target", 4, 11)
        head_sources, head_assignments = oof_sources(cls.head_rows, "inner-responder")
        cls.head, cls.head_report = fit_head_only(
            cls.head_rows, cls.schema, CONFIG,
            response_artifact_by_group=head_assignments,
            provenance_records=head_sources,
        )
        target_groups = sorted(row["group_id"] for row in cls.target_rows)
        cls.outer_responder_id = "outer-responder:0"
        cls.outer_record = make_fit_record(
            cls.outer_responder_id,
            supervised_group_ids=sorted(row["group_id"] for row in cls.head_rows),
            metadata={"synthetic_fixture": True},
        )
        cls.target_assignments = {group_id: cls.outer_responder_id
                                  for group_id in target_groups}
        cls.package = construct_action_targets(
            cls.target_rows, cls.head, cls.schema, CONFIG,
            head_report=cls.head_report,
            response_artifact_by_group=cls.target_assignments,
            provenance_records=[cls.outer_record],
        )

    def test_end_to_end_primitives_are_finite_frozen_and_source_bound(self):
        self.assertEqual(self.head_report["training_protocol"],
                         "cbmjev-nested-oof-head-v1")
        self.assertTrue(self.head_report["oof_response_source_verified"])
        self.assertEqual(self.head_report["head_class_weights"], [1.0, 1.0])
        self.assertTrue(all(not parameter.requires_grad
                            for parameter in self.head.network.parameters()))
        self.assertTrue(self.package["oof_source_exclusion_verified"])
        self.assertTrue(self.package["records"])
        for record in self.package["records"]:
            self.assertEqual(set(record), {"sample_id", "group_id", "epoch",
                                           "observed", "action", "target"})
            self.assertNotIn("y", record)
            self.assertNotIn("z", record)
            self.assertNotIn("after", record)

        controller, report = fit_controller_from_targets(
            self.package, self.schema, CONFIG)
        self.assertEqual(report["training_protocol"],
                         "cbmjev-nested-oof-controller-v1")
        self.assertTrue(report["oof_source_exclusion_verified"])
        self.assertFalse(report["final_models_used_for_same_sample_targets"])
        self.assertEqual(len(report["policy_loss"]), CONFIG["policy_epochs"])
        self.assertTrue(all(torch.isfinite(torch.tensor(report["policy_loss"]))))
        self.assertTrue(all(not parameter.requires_grad
                            for parameter in controller.network.parameters()))
        scores = controller.predict(self.schema.empty_state(),
                                    candidate_actions(self.schema.empty_state(), self.schema,
                                                      pairs=controller.pairs))
        self.assertTrue(scores)
        self.assertTrue(all(0 <= score <= 1 for score in scores))
        self.assertEqual(validate_provenance_dag(report["provenance"])["status"],
                         "VALID")

    def test_per_producer_assignment_is_checked_not_only_union(self):
        sources, assignments = oof_sources(self.head_rows, "assigned")
        swapped = dict(assignments)
        first, second = sorted(swapped)[:2]
        swapped[first], swapped[second] = swapped[second], swapped[first]
        # The swapped producer was fitted on the group it is now claimed to
        # predict, even though the union of source IDs is unchanged.
        with self.assertRaisesRegex(ValueError, "Target-group supervision leakage"):
            fit_head_only(self.head_rows, self.schema, CONFIG,
                          response_artifact_by_group=swapped,
                          provenance_records=sources)

    def test_final_responder_cannot_construct_targets_for_its_training_groups(self):
        target_groups = sorted(row["group_id"] for row in self.target_rows)
        final_record = make_fit_record(
            "responder:final",
            supervised_group_ids=target_groups + sorted(
                row["group_id"] for row in self.head_rows),
        )
        with self.assertRaisesRegex(ValueError, "Target-group supervision leakage"):
            construct_action_targets(
                self.target_rows, self.head, self.schema, CONFIG,
                head_report=self.head_report,
                response_artifact_by_group={group: "responder:final"
                                            for group in target_groups},
                provenance_records=[final_record],
            )

    def test_final_head_cannot_make_targets_for_its_own_training_samples(self):
        final_head, final_report = fit_head_only(
            self.target_rows, self.schema, CONFIG,
            response_artifact_by_group=self.target_assignments,
            provenance_records=[self.outer_record],
        )
        with self.assertRaisesRegex(ValueError, "Target-group supervision leakage"):
            construct_action_targets(
                self.target_rows, final_head, self.schema, CONFIG,
                head_report=final_report,
                response_artifact_by_group=self.target_assignments,
                provenance_records=[self.outer_record],
            )

    def test_mutated_head_and_target_package_fail_closed(self):
        head = copy.deepcopy(self.head)
        next(head.network.parameters()).data.add_(1)
        with self.assertRaisesRegex(ValueError, "head weights do not match"):
            construct_action_targets(
                self.target_rows, head, self.schema, CONFIG,
                head_report=self.head_report,
                response_artifact_by_group=self.target_assignments,
                provenance_records=[self.outer_record],
            )

        package = copy.deepcopy(self.package)
        package["records"][0]["target"] = 1 - package["records"][0]["target"]
        with self.assertRaisesRegex(ValueError, "package hash mismatch"):
            fit_controller_from_targets(package, self.schema, CONFIG)

    def test_config_and_outer_fold_overlap_fail_closed(self):
        changed = {**CONFIG, "masks_per_sample": CONFIG["masks_per_sample"] + 1}
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            fit_controller_from_targets(self.package, self.schema, changed)
        with self.assertRaisesRegex(ValueError, "sample overlap"):
            fit_controller_from_targets([self.package, self.package], self.schema, CONFIG)

    def test_disjoint_outer_packages_merge_into_one_controller(self):
        left_rows, right_rows = self.target_rows[:2], self.target_rows[2:]
        left = construct_action_targets(
            left_rows, self.head, self.schema, CONFIG,
            head_report=self.head_report,
            response_artifact_by_group={row["group_id"]: self.outer_responder_id
                                        for row in left_rows},
            provenance_records=[self.outer_record],
        )
        right = construct_action_targets(
            right_rows, self.head, self.schema, CONFIG,
            head_report=self.head_report,
            response_artifact_by_group={row["group_id"]: self.outer_responder_id
                                        for row in right_rows},
            provenance_records=[self.outer_record],
        )
        controller, report = fit_controller_from_targets(
            [right, left], self.schema, CONFIG)
        self.assertEqual(set(report["target_group_ids"]),
                         {row["group_id"] for row in self.target_rows})
        self.assertEqual(len(report["target_artifact_ids"]), 2)
        self.assertTrue(all(not parameter.requires_grad
                            for parameter in controller.network.parameters()))

    def test_head_fit_is_deterministic_and_preserves_weighting_contract(self):
        sources, assignments = oof_sources(self.head_rows, "repeat")
        first, first_report = fit_head_only(
            self.head_rows, self.schema, CONFIG,
            response_artifact_by_group=assignments,
            provenance_records=sources)
        second, second_report = fit_head_only(
            self.head_rows, self.schema, CONFIG,
            response_artifact_by_group=assignments,
            provenance_records=sources)
        self.assertEqual(first_report, second_report)
        self.assertEqual(first.class_weights, (1.0, 1.0))
        for left, right in zip(first.network.parameters(), second.network.parameters()):
            self.assertTrue(torch.equal(left, right))

    def test_runtime_statuses_are_accepted_but_unqueried_cache_is_rejected(self):
        rows = copy.deepcopy(self.target_rows)
        rows[0]["z"] = [2, 3, 2]  # UNCERTAIN / NOT_APPLICABLE after acquisition.
        package = construct_action_targets(
            rows, self.head, self.schema, CONFIG, head_report=self.head_report,
            response_artifact_by_group=self.target_assignments,
            provenance_records=[self.outer_record])
        self.assertTrue(any(3 in record["observed"] for record in package["records"]))
        fit_controller_from_targets(package, self.schema, CONFIG)
        for invalid in (-1, 4, True):
            rows[0]["z"][0] = invalid
            with self.assertRaisesRegex(ValueError, "complete hard runtime"):
                construct_action_targets(
                    rows, self.head, self.schema, CONFIG, head_report=self.head_report,
                    response_artifact_by_group=self.target_assignments,
                    provenance_records=[self.outer_record])

    def test_head_loss_weights_and_config_are_bound_and_receipts_roundtrip(self):
        for field in ("class_weights", "_loss_weights", "config"):
            head = copy.deepcopy(self.head)
            if field == "class_weights":
                head.class_weights = (2.0, 1.0)
            elif field == "_loss_weights":
                head._loss_weights[0] = 2
            else:
                head.config["dropout"] = 0.1
            with self.assertRaisesRegex(ValueError, "weights"):
                construct_action_targets(
                    self.target_rows, head, self.schema, CONFIG,
                    head_report=self.head_report,
                    response_artifact_by_group=self.target_assignments,
                    provenance_records=[self.outer_record])
        package = construct_action_targets(
            self.target_rows, self.head, self.schema, CONFIG,
            head_report=json.loads(json.dumps(self.head_report)),
            response_artifact_by_group=self.target_assignments,
            provenance_records=[self.outer_record])
        fit_controller_from_targets(json.loads(json.dumps(package)), self.schema, CONFIG)

    def test_ancestor_supervision_leakage_is_rejected(self):
        ancestor = make_fit_record(
            "leaky-ancestor", supervised_group_ids=[self.target_rows[0]["group_id"]])
        responder = make_fit_record(
            "derived-responder", supervised_group_ids=[],
            parent_ids=["leaky-ancestor"], fit_kind="derived")
        with self.assertRaisesRegex(ValueError, "leaky-ancestor"):
            construct_action_targets(
                self.target_rows, self.head, self.schema, CONFIG,
                head_report=self.head_report,
                response_artifact_by_group={group: "derived-responder"
                                            for group in self.target_assignments},
                provenance_records=[ancestor, responder])

    def test_head_responder_ancestor_cannot_have_seen_outer_targets(self):
        sources, assignments = oof_sources(self.head_rows, "inner-leak")
        target_group = self.target_rows[0]["group_id"]
        contaminated = make_fit_record(
            sources[0]["artifact_id"],
            supervised_group_ids=sources[0]["supervised_group_ids"] + [target_group])
        sources[0] = contaminated
        head, receipt = fit_head_only(
            self.head_rows, self.schema, CONFIG,
            response_artifact_by_group=assignments, provenance_records=sources)
        with self.assertRaisesRegex(ValueError, "Target-group supervision leakage"):
            construct_action_targets(
                self.target_rows, head, self.schema, CONFIG, head_report=receipt,
                response_artifact_by_group=self.target_assignments,
                provenance_records=[self.outer_record])

    def test_rehashed_objective_mismatch_and_weighting_change_are_rejected(self):
        package = copy.deepcopy(self.package)
        package["objective"] = "value"
        package.pop("package_sha256")
        package["package_sha256"] = stable_hash(package)
        with self.assertRaisesRegex(ValueError, "objective mismatch"):
            fit_controller_from_targets(package, self.schema, CONFIG)
        with self.assertRaisesRegex(ValueError, "class weighting differs"):
            construct_action_targets(
                self.target_rows, self.head, self.schema,
                {**CONFIG, "class_weighting": "none"}, head_report=self.head_report,
                response_artifact_by_group=self.target_assignments,
                provenance_records=[self.outer_record])

    def test_weighted_value_targets_match_frozen_head_loss_difference(self):
        from cbmjev.learning import encode_states
        imbalanced = copy.deepcopy(self.head_rows)
        for index, row in enumerate(imbalanced):
            row["y"] = int(index == 0)
        sources, assignments = oof_sources(imbalanced, "weighted-value")
        head, receipt = fit_head_only(
            imbalanced, self.schema, CONFIG,
            response_artifact_by_group=assignments, provenance_records=sources)
        self.assertEqual(head.class_weights, (0.6, 3.0))
        package = construct_action_targets(
            self.target_rows, head, self.schema, {**CONFIG, "objective": "value"},
            head_report=receipt,
            response_artifact_by_group=self.target_assignments,
            provenance_records=[self.outer_record])
        rows = {row["sample_id"]: row for row in self.target_rows}
        for record in package["records"]:
            row = rows[record["sample_id"]]
            after = list(record["observed"])
            for atom in self.schema.expand(record["action"]):
                after[atom] = row["z"][atom]
            states = encode_states([record["observed"], after], self.schema)
            losses = head.cross_entropy(
                head.network(states), torch.tensor([row["y"], row["y"]]),
                reduction="none")
            self.assertAlmostEqual(record["target"], float(losses[0] - losses[1]), places=6)
        fit_controller_from_targets(package, self.schema, {**CONFIG, "objective": "value"})


if __name__ == "__main__":
    unittest.main()

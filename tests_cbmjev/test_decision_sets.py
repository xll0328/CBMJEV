import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cbmjev.contracts import stable_hash
from cbmjev.crossfit_targets import open_target_package
from cbmjev.crossfit_training import construct_action_targets
from cbmjev.decision_sets import iter_decision_sets
from cbmjev.provenance import make_fit_record
from tests_cbmjev import test_crossfit_training as fixtures


def rebind(package):
    """Rebuild valid transport/provenance bindings to test structural checks."""
    old_id = package["target_artifact_id"]
    package["record_count"] = len(package["records"])
    package["records_per_epoch"] = [sum(r["epoch"] == e for r in package["records"])
                                   for e in range(package["target_config"]["policy_epochs"])]
    package["records_sha256"] = stable_hash(package["records"])
    keys = ("schema_signature", "objective", "head_artifact_id", "head_component_sha256",
            "target_group_ids", "target_sample_ids", "target_rows_sha256",
            "response_artifact_by_group", "target_config", "records_sha256")
    new_id = "action-targets:" + stable_hash({k: package[k] for k in keys})
    package["target_artifact_id"] = new_id
    record = make_fit_record(new_id, supervised_group_ids=package["target_group_ids"],
        parent_ids=sorted(set(package["response_artifact_by_group"].values()) |
                          {package["head_artifact_id"]}), fit_kind="target_construction",
        metadata={"format": package["format"], "records_sha256": package["records_sha256"],
                  "target_rows_sha256": package["target_rows_sha256"]})
    package["provenance"] = [record if r["artifact_id"] == old_id else r for r in package["provenance"]]
    package.pop("package_sha256", None)
    package["package_sha256"] = stable_hash(package)
    return package


class DecisionSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NestedOOFTrainingPrimitiveTests.setUpClass()
        cls.f = fixtures.NestedOOFTrainingPrimitiveTests
        cls.cfg = {**fixtures.CONFIG, "actions_per_state": 64}

    def make_package(self, cfg=None, path=None):
        f = self.f
        return construct_action_targets(f.target_rows, f.head, f.schema, cfg or self.cfg,
            head_report=f.head_report, response_artifact_by_group=f.target_assignments,
            provenance_records=[f.outer_record], manifest_path=path)

    def test_actual_targets_and_shards_match_no_forward_metadata(self):
        package = self.make_package()
        decisions = list(iter_decision_sets(package, self.f.schema, self.cfg))
        self.assertEqual(len(decisions), 4 * 2 * 2)
        self.assertEqual([risk for d in decisions for risk in d.supervision.risk_targets],
                         [r["target"] for r in package["records"]])
        for i, decision in enumerate(decisions):
            self.assertEqual(set(asdict(decision.model_inputs)), {"observed", "actions"})
            self.assertEqual(set(asdict(decision.supervision)), {"risk_targets"})
            self.assertEqual(decision.metadata.occurrence, i)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "targets.json"
            self.make_package(path=path)
            self.assertEqual(decisions, list(iter_decision_sets(open_target_package(path),
                                                               self.f.schema, self.cfg)))

    def test_repeated_full_visible_masks_remain_distinct_occurrences(self):
        cfg = {**self.cfg, "masks_per_sample": 4}
        with patch("cbmjev.crossfit_training.sample_group_masks",
                   side_effect=lambda count, groups, rng: [(True,) * groups] * count):
            package = self.make_package(cfg)
        decisions = list(iter_decision_sets(package, self.f.schema, cfg))
        self.assertEqual(len(decisions), 4 * 2 * 4)
        self.assertEqual(decisions[1].model_inputs, decisions[2].model_inputs)
        self.assertEqual(decisions[2].model_inputs, decisions[3].model_inputs)
        self.assertNotEqual(decisions[1].metadata.occurrence, decisions[2].metadata.occurrence)

    def test_triggered_cap_and_value_objective_rejected(self):
        cfg = {**self.cfg, "actions_per_state": 2}
        package = self.make_package(cfg)
        with self.assertRaisesRegex(ValueError, "cap removed"):
            iter_decision_sets(package, self.f.schema, cfg)
        with self.assertRaisesRegex(ValueError, "requires original risk"):
            iter_decision_sets(package, self.f.schema, {**cfg, "objective": "value"})

    def test_structural_corruption_rejected_before_iterator_return(self):
        original = self.make_package()
        variants = []
        deleted = copy.deepcopy(original)
        del deleted["records"][1]
        variants.append((deleted, "candidate count"))
        duplicate = copy.deepcopy(original)
        duplicate["records"].insert(2, copy.deepcopy(duplicate["records"][1]))
        variants.append((duplicate, "unique actions|bounded candidate"))
        reordered = copy.deepcopy(original)
        reordered["records"][0], reordered["records"][1] = reordered["records"][1], reordered["records"][0]
        variants.append((reordered, "start with STOP"))
        reordered_candidates = copy.deepcopy(original)
        reordered_candidates["records"][1], reordered_candidates["records"][2] = (
            reordered_candidates["records"][2], reordered_candidates["records"][1])
        variants.append((reordered_candidates, "candidate order"))
        late = copy.deepcopy(original)
        # Damage the final non-STOP set, so preflight must scan past valid sets.
        index = max(i for i, r in enumerate(late["records"]) if r["action"])
        del late["records"][index]
        variants.append((late, "candidate count"))
        for package, message in variants:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                iter_decision_sets(rebind(package), self.f.schema, self.cfg)

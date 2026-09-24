from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from cbmjev.choice_artifacts import (_reservoir, load_choice_pair, train_choice_pair,
                                     verify_choice_binding_unchanged)
from cbmjev.cli import parser, dispatch
from cbmjev.contracts import stable_hash
from cbmjev.crossfit_training import construct_action_targets
from cbmjev.io import file_hash, read_json, write_json, write_jsonl
from tests_cbmjev.test_decision_sets import rebind
from tests_cbmjev import test_crossfit_training as fixtures


class ChoiceArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        fixtures.NestedOOFTrainingPrimitiveTests.setUpClass()
        cls.f = fixtures.NestedOOFTrainingPrimitiveTests
        cls.cfg = {**fixtures.CONFIG, "actions_per_state": 64}

    def fixture(self, root, cfg=None, role="policy_fit"):
        cfg = self.cfg if cfg is None else cfg
        root = Path(root)
        prepared = root / "prepared"
        prepared.mkdir()
        write_json(prepared / "schema.json", self.f.schema.to_dict())
        rows = self.f.head_rows + self.f.target_rows
        write_jsonl(prepared / "samples.jsonl", rows)
        write_jsonl(prepared / "membership.jsonl", [
            {"sample_id": row["sample_id"], "group_id": row["group_id"],
             "split": "head_fit" if row in self.f.head_rows else role} for row in rows])
        config = root / "target_config.json"
        write_json(config, cfg)
        paths = []
        for index, rows in enumerate((self.f.target_rows[:2], self.f.target_rows[2:])):
            path = root / ("targets_%d.json" % index)
            construct_action_targets(rows, self.f.head, self.f.schema, cfg,
                head_report=self.f.head_report,
                response_artifact_by_group={row["group_id"]: self.f.outer_responder_id for row in rows},
                provenance_records=[self.f.outer_record], manifest_path=path)
            paths.append(path)
        return prepared, paths, config

    def test_actual_sharded_targets_fit_reload_same_init_and_cli(self):
        with tempfile.TemporaryDirectory() as root:
            prepared, targets, config = self.fixture(root)
            out = Path(root) / "trained"
            args = parser().parse_args(["train-choice-pair", "--prepared", str(prepared),
                "--targets", *map(str, targets), "--config", str(config), "--out", str(out),
                "--epochs", "1", "--max-questions", "5", "--batch-size", "2"])
            receipt = dispatch(args)
            self.assertEqual(receipt["status"], "COMPLETE")
            scalar, attention, settings, report = load_choice_pair(out, prepared=prepared,
                targets=targets, config=config)
            self.assertTrue(report["fit"]["initial_active_logits_equal"])
            self.assertEqual(report["available_occurrences"], 16)
            self.assertEqual(report["selected_occurrences"], 5)
            self.assertEqual(len(report["package_bindings"]), 2)
            self.assertFalse(settings["paper_evidence"])
            self.assertIn("not-MLP", settings["backend"])
            self.assertFalse(any(p.requires_grad for p in attention.parameters()))
            from cbmjev.choice_runtime import StructuredChoiceController
            from cbmjev.runtime import ReplayEnvironment, run_episode
            for head in (scalar, attention):
                controller = StructuredChoiceController(self.f.schema, head)
                trace = run_episode(ReplayEnvironment(tuple(self.f.target_rows[0]["z"]), self.f.schema),
                    self.f.schema, self.f.head, method="structured_choice", controller=controller,
                    max_groups=1, cost_weight=settings["cost_weight"])
                self.assertLessEqual(len(trace["queried_groups"]), 1)
                self.assertNotIn("y", trace)
                for step in trace["steps"]:
                    self.assertNotIn("y", step)
            second = Path(root) / "trained_again"
            train_choice_pair(prepared, targets, config, second, epochs=1, max_questions=5, batch_size=2)
            other_scalar, other_attention, _, other_report = load_choice_pair(second,
                prepared=prepared, targets=targets, config=config)
            self.assertEqual(report["selected_derived_ids"], other_report["selected_derived_ids"])
            for left, right in ((scalar, other_scalar), (attention, other_attention)):
                for name, value in left.state_dict().items():
                    self.assertTrue(torch.equal(value, right.state_dict()[name]))
            with self.assertRaisesRegex(ValueError, "not test"):
                load_choice_pair(out, prepared=prepared, targets=targets, config=config, split="test")
            with (out / "scalar.pt").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_choice_pair(out, prepared=prepared, targets=targets, config=config)
            weights = scalar.state_dict()
            weights["scalar.bias"] = weights["scalar.bias"] + 1
            torch.save(weights, out / "scalar.pt")
            rebound = read_json(out / "receipt.json")
            rebound["files"]["scalar.pt"] = file_hash(out / "scalar.pt")
            rebound.pop("receipt_sha256")
            rebound["receipt_sha256"] = stable_hash(rebound)
            (out / "receipt.json").unlink()
            write_json(out / "receipt.json", rebound)
            with self.assertRaisesRegex(ValueError, "tensor hash"):
                load_choice_pair(out, prepared=prepared, targets=targets, config=config)

    def test_reservoir_deterministic_occurrences_not_unique_states(self):
        stream = [(i, "identical-state") for i in range(30)]
        selected, total = _reservoir(iter(stream), 7, 60)
        self.assertEqual(total, 30)
        self.assertEqual(len(selected), 7)
        self.assertEqual(selected, _reservoir(iter(stream), 7, 60)[0])
        self.assertNotEqual(selected, _reservoir(iter(stream), 7, 61)[0])
        self.assertEqual(_reservoir(iter(stream), 100, 60)[0], tuple(stream))

    def test_sealed_reload_reuses_validated_stream_but_rechecks_bytes(self):
        from cbmjev.crossfit_targets import DiskRecords
        with tempfile.TemporaryDirectory() as root:
            prepared, targets, config = self.fixture(root)
            out = Path(root) / "trained"
            train_choice_pair(prepared, targets, config, out, epochs=1,
                              max_questions=5, batch_size=2)
            with patch("cbmjev.choice_artifacts.iter_decision_sets",
                       side_effect=AssertionError("training decisions revalidated")):
                with patch.object(DiskRecords, "__iter__",
                                  side_effect=AssertionError("training records reloaded")):
                    *heads, settings, report = load_choice_pair(out, prepared=prepared,
                        targets=targets, config=config,
                        reuse_sealed_training_validation=True)
            self.assertEqual(len(heads), 2)
            self.assertEqual(report["selected_occurrences"], 5)
            self.assertEqual(settings["feature_backend"], "structured")
            verify_choice_binding_unchanged(out, prepared=prepared,
                targets=targets, config=config)
            with targets[0].open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                load_choice_pair(out, prepared=prepared, targets=targets,
                    config=config, reuse_sealed_training_validation=True)
            with self.assertRaisesRegex(ValueError, "binding changed"):
                verify_choice_binding_unchanged(out, prepared=prepared,
                    targets=targets, config=config)

    def test_end_binding_rechecks_frozen_backbone_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            prepared, targets, config = self.fixture(root)
            out = Path(root) / "trained"
            train_choice_pair(prepared, targets, config, out, epochs=1,
                              max_questions=5, batch_size=2)
            settings = read_json(out / "config.json")
            settings["feature_backend"] = "frozen_language"
            settings["language_encoding"] = {"backbone_path": str(Path(root) / "backbone")}
            (out / "config.json").unlink()
            write_json(out / "config.json", settings)
            receipt = read_json(out / "receipt.json")
            receipt["bindings"]["backbone"] = {"config.json": "original"}
            receipt["files"]["config.json"] = file_hash(out / "config.json")
            receipt.pop("receipt_sha256")
            receipt["receipt_sha256"] = stable_hash(receipt)
            (out / "receipt.json").unlink()
            write_json(out / "receipt.json", receipt)
            with patch("cbmjev.choice_artifacts.backbone_manifest",
                       return_value={"config.json": "original"}):
                verify_choice_binding_unchanged(out, prepared=prepared,
                    targets=targets, config=config)
            with patch("cbmjev.choice_artifacts.backbone_manifest",
                       return_value={"config.json": "changed"}):
                with self.assertRaisesRegex(ValueError, "binding changed"):
                    verify_choice_binding_unchanged(out, prepared=prepared,
                        targets=targets, config=config)

    def test_training_validates_each_decision_package_once_before_sampling(self):
        from cbmjev.crossfit_targets import DiskRecords
        from cbmjev.decision_sets import iter_decision_sets
        with tempfile.TemporaryDirectory() as root:
            prepared, targets, config = self.fixture(root)
            original_iter = DiskRecords.__iter__
            passes = []

            def count_pass(records):
                passes.append(records.root)
                return original_iter(records)

            with patch("cbmjev.choice_artifacts.iter_decision_sets",
                       wraps=iter_decision_sets) as checked:
                with patch.object(DiskRecords, "__iter__", count_pass):
                    receipt = train_choice_pair(prepared, targets, config,
                        Path(root) / "single_validation", epochs=1,
                        max_questions=5, batch_size=2)
            self.assertEqual(receipt["status"], "COMPLETE")
            self.assertEqual(checked.call_count, len(targets))
            # Per package: one role scan; package/record/field validation;
            # occurrence coverage; and the consumed decision-set stream.
            self.assertEqual(len(passes), 6 * len(targets))

    def test_uniform_budget_sampling_is_bounded_reproducible_and_keeps_occurrences(self):
        from cbmjev.choice_training import fit_choice_head_pair
        with tempfile.TemporaryDirectory() as root:
            prepared, targets, config = self.fixture(root)
            captured = []

            def capture(examples, *args, **kwargs):
                captured.append(examples)
                return fit_choice_head_pair(examples, *args, **kwargs)

            with patch("cbmjev.choice_artifacts.fit_choice_head_pair", side_effect=capture):
                for name in ("first", "second"):
                    train_choice_pair(prepared, targets, config, Path(root) / name,
                        epochs=1, max_questions=100, budget_mode="uniform_remaining")
            self.assertEqual(captured[0], captured[1])
            self.assertEqual(len(captured[0]), 16)
            self.assertEqual(len({(e.source.source_artifact_id, e.source.occurrence)
                                  for e in captured[0]}), 16)
            budgets = set()
            for example in captured[0]:
                inputs = example.model_inputs
                available = self.f.schema.num_groups - sum(self.f.schema.group_mask(inputs.observed))
                self.assertLessEqual(inputs.remaining_groups, available)
                self.assertGreaterEqual(inputs.remaining_groups, 0)
                budgets.add(inputs.remaining_groups)
                if not inputs.remaining_groups:
                    self.assertEqual(inputs.actions, ((),))
            self.assertGreater(len(budgets), 1)
            _, _, settings, report = load_choice_pair(Path(root) / "first",
                prepared=prepared, targets=targets, config=config)
            self.assertIsNone(settings["remaining_groups"])
            self.assertEqual(sum(report["selected_budget_counts"].values()), 16)
            self.assertEqual(set(map(int, report["selected_budget_counts"])), budgets)
            with patch("cbmjev.choice_artifacts.fit_choice_head_pair") as fit:
                for kwargs in ({"budget_mode": "bad"},
                               {"budget_mode": "uniform_remaining", "remaining_groups": 1}):
                    with self.assertRaises(ValueError):
                        train_choice_pair(prepared, targets, config, Path(root) / "invalid", **kwargs)
                fit.assert_not_called()
            self.assertFalse((Path(root) / "invalid").exists())

    def test_invalid_cap_and_heldout_roles_fail_before_fit_or_output(self):
        for options in ({"cfg": {**self.cfg, "actions_per_state": 2}}, {"role": "validation"}):
            with tempfile.TemporaryDirectory() as root:
                prepared, targets, config = self.fixture(root, **options)
                out = Path(root) / "must_not_exist"
                with patch("cbmjev.choice_artifacts.fit_choice_head_pair") as fit:
                    with self.assertRaises(ValueError):
                        train_choice_pair(prepared, targets, config, out)
                    fit.assert_not_called()
                self.assertFalse(out.exists())

    def test_missing_package_and_late_structural_error_preflight(self):
        with tempfile.TemporaryDirectory() as root:
            prepared, targets, config = self.fixture(root)
            out = Path(root) / "must_not_exist"
            with self.assertRaises(FileNotFoundError):
                train_choice_pair(prepared, [*targets, Path(root) / "missing"], config, out)
            from cbmjev.crossfit_targets import open_target_package
            package = open_target_package(targets[-1])
            package["records"] = list(package["records"])
            package["records"].pop()  # invalid late occurrence, transport hashes rebound
            rebind(package)
            invalid = Path(root) / "late_invalid.json"
            write_json(invalid, package)
            with patch("cbmjev.choice_artifacts.fit_choice_head_pair") as fit:
                with self.assertRaises(ValueError):
                    train_choice_pair(prepared, [targets[0], invalid], config, out, max_questions=1)
                fit.assert_not_called()
            self.assertFalse(out.exists())

    def test_duplicate_group_packages_and_input_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            prepared, targets, config = self.fixture(root)
            from cbmjev.crossfit_targets import open_target_package
            package = open_target_package(targets[0])
            package["records"] = list(package["records"])
            duplicate = Path(root) / "duplicate.json"
            write_json(duplicate, package)
            with self.assertRaisesRegex(ValueError, "disjoint"):
                train_choice_pair(prepared, [targets[0], duplicate], config, Path(root) / "invalid")
            out = Path(root) / "trained"
            train_choice_pair(prepared, targets, config, out, epochs=1, max_questions=2)
            with config.open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                load_choice_pair(out, prepared=prepared, targets=targets, config=config)

    def test_mutation_during_training_leaves_no_completion_or_output(self):
        from cbmjev.choice_training import fit_choice_head_pair
        with tempfile.TemporaryDirectory() as root:
            prepared, targets, config = self.fixture(root)
            out = Path(root) / "must_not_exist"

            def mutate(*args, **kwargs):
                result = fit_choice_head_pair(*args, **kwargs)
                with config.open("a") as handle:
                    handle.write("\n")
                return result

            with patch("cbmjev.choice_artifacts.fit_choice_head_pair", side_effect=mutate):
                with self.assertRaisesRegex(ValueError, "changed during training"):
                    train_choice_pair(prepared, targets, config, out, epochs=1, max_questions=2)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()

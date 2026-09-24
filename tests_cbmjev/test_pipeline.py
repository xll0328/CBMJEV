"""End-to-end CPU/software integrity tests, never evidence of dataset performance.

One shared tiny trained fixture; mutation tests copy artifacts before tampering.
No network, downloaded data, pretrained weights, or GPU is required.
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from cbmjev import pipeline
from cbmjev.cli import main, parser
from cbmjev.contracts import DeclaredCost, ModelInput
from cbmjev.io import read_json, read_jsonl
from cbmjev.runtime import LiveEnvironment, ReplayEnvironment, run_episode


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "pipeline integration needs locally installed torch")
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory(prefix="cbmjev-pipeline-test-")
        cls.root = Path(cls.shared.name)
        cls.addClassCleanup(cls.shared.cleanup)
        # The official smoke executes actual fit/cache/replay/live/freeze/certify
        # paths on generated official-format CEBaB, explicitly labeled synthetic.
        cls.artifacts = cls.root / "smoke"
        cls.smoke_report = pipeline.smoke(cls.artifacts, seed=17)
        cls.prepared = cls.artifacts / "prepared"
        cls.models = cls.artifacts / "models"
        cls.cache = cls.artifacts / "cache"
        cls.responder = cls.artifacts / "responder"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cbmjev-pipeline-case-")
        self.work = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def copy(self, source, name):
        target = self.work / name
        shutil.copytree(source, target)
        return target

    def freeze_now(self, name="frozen", **kwargs):
        out = self.work / name
        result = pipeline.freeze_family(self.models, self.cache, out,
                                        methods=["all", "risk"], weights=[0.0, 0.03], **kwargs)
        return out, result

    def test_smoke_completes_actual_stages_without_test_or_paper_claims(self):
        report = self.smoke_report
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["kind"], "SYNTHETIC_SOFTWARE_SMOKE_NOT_PAPER_EVIDENCE")
        self.assertFalse(report["test_evaluated"])
        self.assertIn("semantic_fit", report["stages"])
        self.assertIn("calibration", report["stages"])
        self.assertIn("csv_export", report["stages"])
        self.assertFalse((self.artifacts / "test").exists())
        self.assertTrue((self.artifacts / "tables/metrics.csv").is_file())
        self.assertNotEqual(read_json(self.artifacts / "certification.json")["status"],
                            "CONDITIONAL_CERTIFICATE")
        self.assertEqual(read_json(self.models / "receipt.json")["training_protocol"],
                         "DISJOINT_ROLES_NO_FINAL_REFIT")
        schema, rows, membership = pipeline.load_prepared(self.prepared)
        self.assertEqual(schema.num_atoms, 4)
        self.assertEqual(len(rows), 150)
        groups = {}
        for entry in membership.values():
            groups.setdefault(entry["split"], set()).add(entry["group_id"])
        self.assertEqual(len(groups), 6)
        names = list(groups)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                self.assertFalse(groups[left] & groups[right])

    def test_hf_responder_pipeline_records_initialization_and_ignores_task_y(self):
        import torch
        from tests_cbmjev.test_responders import fake_hf_library

        poisoned = self.copy(self.prepared, "poisoned-hf-prepared")
        _, rows, membership = pipeline.load_prepared(poisoned)
        for row in rows:
            row["target"] = {"value": "FORBIDDEN_TASK_Y", "status": "FORBIDDEN_TASK_STATUS"}
            if membership[row["sample_id"]]["split"] != "responder_fit":
                row["concepts"] = "FORBIDDEN_HELDOUT_GOLD"
                row["input"] = {"FORBIDDEN_HELDOUT_INPUT": True}
        (poisoned / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        library = fake_hf_library()
        original_out, poisoned_out = self.work / "hf-original", self.work / "hf-poisoned"
        with patch.dict("sys.modules", {"transformers": library}):
            for prepared, out in ((self.prepared, original_out), (poisoned, poisoned_out)):
                pipeline.train_responder(prepared, out, kind="hf_text", backbone="org/test-fake-encoder",
                                         revision="fixture-tag", epochs=1, batch_size=16, seed=41)
            schema, _, roles = pipeline.load_prepared(self.prepared)
            restored, receipt = pipeline.load_backend(original_out, schema)
            self.assertEqual(restored.config()["kind"], "hf_text")
            self.assertEqual(receipt["initialization"]["requested_revision"], "fixture-tag")
            self.assertEqual(receipt["initialization"]["resolved_revision"], "a" * 40)
            self.assertEqual(receipt["source_revision"], read_json(self.prepared / "audit.json")["source_revision"])
            self.assertEqual(receipt["shared_compute_mode"], "SHARED_HF_TEXT_ALL_HEADS")
            self.assertEqual(set(receipt["supervised_group_ids"]),
                             {r["group_id"] for r in roles.values() if r["split"] == "responder_fit"})
            pipeline.cache_responses(self.prepared, original_out, self.work / "hf-cache")
        report = read_json(original_out / "training.json")
        self.assertEqual(report["learning_rate"], 3e-5)
        self.assertFalse(report["task_label_gradient"])
        self.assertFalse(report["freeze_backbone"])
        first = torch.load(original_out / "responder.pt", map_location="cpu", weights_only=True)
        second = torch.load(poisoned_out / "responder.pt", map_location="cpu", weights_only=True)
        self.assertEqual(first["config"], second["config"])
        for key, value in first["state_dict"].items():
            self.assertTrue(torch.equal(value, second["state_dict"][key]), key)

    def test_hf_cli_and_explicit_source_fail_closed(self):
        args = parser().parse_args(["train-responder", "--kind", "hf_text", "--prepared", str(self.prepared),
                                   "--out", str(self.work / "unused"), "--backbone", "org/named-encoder",
                                   "--allow-download", "--revision", "release", "--freeze-backbone", "--max-length", "256"])
        self.assertEqual(args.kind, "hf_text")
        self.assertTrue(args.allow_download)
        self.assertTrue(args.freeze_backbone)
        self.assertEqual(args.max_length, 256)
        with self.assertRaisesRegex(ValueError, "explicit --backbone"):
            pipeline.train_responder(self.prepared, self.work / "missing-source", kind="hf_text")
        with self.assertRaisesRegex(ValueError, "require kind=hf_text"):
            pipeline.train_responder(self.prepared, self.work / "wrong-kind", allow_download=True)
        self.assertFalse((self.work / "missing-source").exists())
        self.assertFalse((self.work / "wrong-kind").exists())

    def test_unfrozen_evaluation_max_groups_override_and_frozen_rejection(self):
        arguments = ["evaluate", "--models", str(self.models), "--cache", str(self.cache),
                     "--out", str(self.work / "budget-one"), "--max-groups", "1"]
        self.assertEqual(parser().parse_args(arguments).max_groups, 1)
        original_config = (self.models / "config.json").read_bytes()
        pipeline.evaluate_models(self.models, self.cache, self.work / "budget-one", methods=["fixed"], max_groups=1)
        traces = read_jsonl(self.work / "budget-one/traces.jsonl")
        self.assertTrue(traces)
        self.assertTrue(all(len(row["queried_groups"]) == 1 for row in traces))
        self.assertEqual((self.models / "config.json").read_bytes(), original_config)
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            pipeline.evaluate_models(self.models, self.cache, self.work / "negative-budget", methods=["fixed"], max_groups=-1)
        frozen, _ = self.freeze_now("budget-frozen")
        with self.assertRaisesRegex(ValueError, "cannot override.*budgets"):
            pipeline.evaluate_models(self.models, self.cache, self.work / "frozen-override", frozen_family=frozen, max_groups=1)
        with self.assertRaisesRegex(ValueError, "all means one full initial batch"):
            pipeline.evaluate_models(self.models, self.cache, self.work / "all-underbudget", methods=["all"], max_groups=1)

    def test_default_evaluation_is_validation_and_test_is_guarded(self):
        result = pipeline.evaluate_models(self.models, self.cache, self.work / "validation",
                                           methods=["all", "risk"])
        self.assertEqual(result["split"], "validation")
        self.assertEqual(result["mode"], "offline_replay")
        traces = read_jsonl(self.work / "validation/traces.jsonl")
        self.assertTrue(all(row["split"] == "validation" for row in traces))
        self.assertTrue(all("total_wall_ms" not in row for row in traces))
        metrics = read_json(self.work / "validation/metrics.json")
        receipt = read_json(self.models / "receipt.json")
        for key in ("schema_hash", "models_sha256", "cache_sha256", "static_order_sha256",
                    "responder_checkpoint_sha256", "head_component_sha256",
                    "controller_component_sha256"):
            self.assertEqual(metrics[key], receipt[key])
        self.assertEqual(metrics["training_source_code_hash"], receipt["source_code_hash"])
        self.assertEqual(metrics["models_receipt_sha256"],
                         pipeline.file_hash(self.models / "receipt.json"))
        with self.assertRaisesRegex(ValueError, "evaluate-test"):
            pipeline.evaluate_models(self.models, self.cache, self.work / "test", split="test")
        self.assertFalse((self.work / "test").exists())

    def test_cli_default_and_error_exit_do_not_touch_test(self):
        arguments = ["evaluate", "--models", str(self.models), "--cache", str(self.cache),
                     "--out", str(self.work / "cli-test")]
        self.assertEqual(parser().parse_args(arguments).split, "validation")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(arguments + ["--split", "test"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(stderr.getvalue())["status"], "ERROR")
        self.assertFalse((self.work / "cli-test").exists())

    def test_calibration_needs_both_freeze_and_certification_flag(self):
        frozen, receipt = self.freeze_now()
        cases = ({}, {"frozen_family": frozen}, {"certification_run": True})
        for i, kwargs in enumerate(cases):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "calibration requires"):
                pipeline.evaluate_models(self.models, self.cache, self.work / ("bad" + str(i)),
                                         split="calibration", **kwargs)
        # Freeze and evaluation remain adjacent: concurrent development may change
        # source hashes, so no test reuses the smoke's earlier frozen manifest.
        result = pipeline.evaluate_models(self.models, self.cache, self.work / "calibration",
                 split="calibration", frozen_family=frozen, certification_run=True)
        self.assertEqual(result["manifest_hash"], receipt["manifest_hash"])
        self.assertEqual(result["num_policies"], 3)
        traces = read_jsonl(self.work / "calibration/traces.jsonl")
        self.assertTrue(all(row["split"] == "calibration" for row in traces))
        self.assertEqual(read_json(self.work / "calibration/metrics.json")["paired_against_all"], {})

    def test_frozen_family_disallows_method_or_threshold_overrides(self):
        frozen, _ = self.freeze_now()
        for i, override in enumerate(({"methods": ["all"]}, {"cost_weight": 0.4})):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "cannot override"):
                pipeline.evaluate_models(self.models, self.cache, self.work / str(i),
                    split="calibration", frozen_family=frozen, certification_run=True, **override)

    def test_tampered_frozen_settings_are_rejected(self):
        frozen, _ = self.freeze_now()
        path = frozen / "settings.json"
        settings = read_json(path)
        first = next(iter(settings))
        settings[first]["config"]["policy"]["cost_weight"] += 0.123
        path.write_text(json.dumps(settings), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "complete frozen system changed"):
            pipeline.evaluate_models(self.models, self.cache, self.work / "rejected",
                split="calibration", frozen_family=frozen, certification_run=True)

    def test_frozen_family_rejects_implicit_config_device_change(self):
        frozen, _ = self.freeze_now()
        copied = self.copy(self.models, "changed-device-models")
        path = copied / "config.json"
        config = read_json(path)
        self.assertEqual(config["device"], "cpu")
        config["device"] = "cpu:0"
        path.write_text(json.dumps(config), encoding="utf-8")
        # Both resolve to CPU, but the complete frozen deployment configuration
        # is exact. No explicit device override is passed to evaluate_models.
        with self.assertRaisesRegex(ValueError, "deployment mode/device"):
            pipeline.evaluate_models(copied, self.cache, self.work / "changed-device-run",
                split="calibration", frozen_family=frozen, certification_run=True)

    def test_tampered_frozen_manifest_and_changed_mode_are_rejected(self):
        frozen, _ = self.freeze_now()
        path = frozen / "manifest.json"
        manifest = read_json(path)
        manifest["policies"][0]["system_hash"] = "0" * 64
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid frozen family manifest"):
            pipeline.evaluate_models(self.models, self.cache, self.work / "tampered",
                split="calibration", frozen_family=frozen, certification_run=True)
        clean, _ = self.freeze_now(name="clean")
        with self.assertRaisesRegex(ValueError, "deployment mode/device"):
            pipeline.evaluate_models(self.models, self.cache, self.work / "live-mode",
                split="calibration", frozen_family=clean, certification_run=True,
                prepared=self.prepared, responder_dir=self.responder)

    def test_cache_hash_tampering_is_rejected(self):
        copied = self.copy(self.cache, "cache")
        path = copied / "responses.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "content differs from frozen hash"):
            pipeline.load_cache(copied)

    def test_checkpoint_and_static_order_hash_tampering_is_rejected(self):
        for filename, expected in (("models.pt", "model weights differ"),
                                   ("static_order.json", "static order differs")):
            with self.subTest(filename=filename):
                copied = self.copy(self.models, filename.replace(".", "_"))
                path = copied / filename
                path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, expected):
                    pipeline.load_model_bundle(copied, self.cache)
        copied = self.copy(self.responder, "responder")
        path = copied / "responder.pt"
        path.write_bytes(path.read_bytes() + b" ")
        schema, _, _ = pipeline.load_prepared(self.prepared)
        with self.assertRaisesRegex(ValueError, "checkpoint does not match"):
            pipeline.load_backend(copied, schema)

    def test_live_prepared_records_hash_tampering_is_rejected(self):
        copied = self.copy(self.prepared, "prepared")
        path = copied / "samples.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "canonical records differ"):
            pipeline.evaluate_models(self.models, self.cache, self.work / "live",
                methods=["all"], prepared=copied, responder_dir=self.responder)

    def test_same_trained_backend_live_and_replay_predictions_agree(self):
        replay = {row["sample_id"]: row for row in
                  read_jsonl(self.artifacts / "validation_replay/traces.jsonl") if row["policy_id"] == "all"}
        live = {row["sample_id"]: row for row in
                read_jsonl(self.artifacts / "validation_live/traces.jsonl") if row["policy_id"] == "all"}
        self.assertEqual(set(replay), set(live))
        self.assertTrue(replay)
        for sid in replay:
            for key in ("prediction", "probabilities", "final_state", "queried_groups", "queried_atoms"):
                self.assertEqual(replay[sid][key], live[sid][key], (sid, key))
            self.assertIn("total_wall_ms", live[sid])
            self.assertNotIn("total_wall_ms", replay[sid])
            self.assertEqual(live[sid]["backend_stats"][0]["atoms_computed"], 4)

    def test_live_warmup_reports_actual_validation_ids_and_overlap(self):
        metrics = read_json(self.artifacts / "validation_live/metrics.json")
        warmup_ids = metrics["warmup_sample_ids"]
        overlap_ids = metrics["warmup_overlap_sample_ids"]
        self.assertIsInstance(warmup_ids, list)
        self.assertIsInstance(overlap_ids, list)
        self.assertEqual(metrics["warmup_validation_cases_per_policy"], 1)
        self.assertEqual(len(warmup_ids), 1)
        self.assertEqual(overlap_ids, warmup_ids)
        _, cache_rows, _ = pipeline.load_cache(self.cache)
        expected = [row["sample_id"] for row in cache_rows if row["split"] == "validation"][:1]
        self.assertEqual(warmup_ids, expected)
        traces = read_jsonl(self.artifacts / "validation_live/traces.jsonl")
        for policy_id in metrics["policies"]:
            evaluated = {row["sample_id"] for row in traces if row["policy_id"] == policy_id}
            self.assertTrue(set(warmup_ids).issubset(evaluated))

    def test_same_fake_live_and_replay_actions_and_predictions_agree(self):
        schema, _, _ = pipeline.load_prepared(self.prepared)
        answers = (0, 1, 2, 1)

        class FakeResponder:
            def __init__(self):
                self.calls = []

            def respond(self, payload, atoms):
                if not isinstance(payload, ModelInput) or payload.text != "content only":
                    raise AssertionError("responder received non-content metadata")
                self.calls.append(tuple(atoms))
                return tuple(answers[a] for a in atoms)

        class FakeHead:
            def probabilities(self, state):
                winner = sum(v for v in state if v >= 0) % 5
                return tuple(float(i == winner) for i in range(5))

        fake = FakeResponder()
        live_env = LiveEnvironment({"modality": "text", "text": "content only", "image_paths": []},
                                   schema, fake)
        replay_env = ReplayEnvironment(answers, schema)
        kwargs = dict(method="fixed", max_groups=3, cost=DeclaredCost(call=0.4, per_group=0.2))
        live = run_episode(live_env, schema, FakeHead(), **kwargs)
        replay = run_episode(replay_env, schema, FakeHead(), **kwargs)
        for key in ("prediction", "final_state", "probabilities", "steps", "calls", "declared_cost"):
            self.assertEqual(live[key], replay[key], key)
        self.assertEqual(fake.calls, [(0,), (1,), (2,)])
        self.assertEqual(live["final_state"][-1], -1)

    def test_shared_cheap_first_query_accounts_for_all_heads(self):
        schema, rows, _ = pipeline.load_prepared(self.prepared)
        responder, _ = pipeline.load_backend(self.responder, schema)
        env = LiveEnvironment(rows[0]["input"], schema, responder)
        env.query((0,))
        first = env.backend_stats[0]
        self.assertEqual(first["encoder_forwards"], 1)
        self.assertEqual(first["atoms_computed"], schema.num_atoms)
        self.assertFalse(first["cache_hit"])
        self.assertEqual(first["cost_mode"], "SHARED_CHEAP_ALL_HEADS")
        self.assertEqual(sum(value != -1 for value in env.state()), 1)
        env.query((1,))
        self.assertEqual(env.backend_stats[1]["encoder_forwards"], 0)
        self.assertEqual(env.backend_stats[1]["atoms_computed"], 0)
        self.assertTrue(env.backend_stats[1]["cache_hit"])
        self.assertEqual(env.calls, 2)

    def test_head_identity_is_independent_of_policy_training_epochs(self):
        baseline = read_json(self.models / "receipt.json")
        config = read_json(self.models / "config.json")
        # Head settings, seed, data, and responder ancestry remain identical.
        # Only the downstream controller optimization duration changes.
        config["learning"]["policy_epochs"] += 1
        changed = self.work / "longer-policy-fit"
        pipeline.train_models(self.cache, config, changed)
        receipt = read_json(changed / "receipt.json")
        self.assertEqual(receipt["head_component_sha256"], baseline["head_component_sha256"])
        self.assertEqual(receipt["head_artifact_id"], baseline["head_artifact_id"])
        self.assertNotEqual(receipt["controller_component_sha256"], baseline["controller_component_sha256"])
        self.assertNotEqual(receipt["controller_artifact_id"], baseline["controller_artifact_id"])
        # Loading rechecks component digests rather than merely trusting receipts.
        pipeline.load_model_bundle(changed, self.cache)

    def test_batch_dependent_backend_fails_flat_cache_guard(self):
        schema, rows, membership = pipeline.load_prepared(self.prepared)
        validation = [row for row in rows if membership[row["sample_id"]]["split"] == "validation"]

        class BatchDependentResponder:
            def respond(self, payload, atom_ids):
                if not isinstance(payload, ModelInput):
                    raise AssertionError("fake backend requires sanitized content")
                return tuple(int(len(atom_ids) > 1) for _ in atom_ids)

        fake = BatchDependentResponder()
        report = pipeline.verify_batch_invariance(schema, fake, validation, limit=2)
        self.assertFalse(report["passed"])
        receipt = read_json(self.responder / "receipt.json")
        # Even an optimistic legacy receipt cannot override observed dependence.
        receipt["batch_independent"] = True
        target = self.work / "invalid-flat-cache"
        with patch.object(pipeline, "load_backend", return_value=(fake, receipt)):
            with self.assertRaisesRegex(ValueError, "batch-dependent answers detected"):
                pipeline.cache_responses(self.prepared, self.responder, target)
        self.assertFalse(target.exists())

    def test_all_budget_failure_never_emits_misnamed_paired_reference(self):
        for i, changed_policy in enumerate(({"max_groups": 1}, {"max_cost": 0.5})):
            with self.subTest(policy=changed_policy):
                copied = self.copy(self.models, "budgeted-models-" + str(i))
                path = copied / "config.json"
                config = read_json(path)
                config["policy"].update(changed_policy)
                path.write_text(json.dumps(config), encoding="utf-8")
                target = self.work / ("invalid-all-run-" + str(i))
                with self.assertRaisesRegex(ValueError, "all means|full all-at-once is infeasible"):
                    pipeline.evaluate_models(copied, self.cache, target, methods=["all", "risk"])
                # An empty output directory may exist; no mislabeled baseline,
                # partial traces, or paired result may be exported as evidence.
                self.assertFalse((target / "metrics.json").exists())
                self.assertFalse((target / "traces.jsonl").exists())

    def test_existing_outputs_are_not_overwritten(self):
        occupied = self.work / "occupied"
        occupied.mkdir()
        sentinel = occupied / "user-owned.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "existing artifacts"):
            pipeline.evaluate_models(self.models, self.cache, occupied, methods=["all"])
        with self.assertRaisesRegex(ValueError, "existing artifacts"):
            pipeline.freeze_family(self.models, self.cache, occupied, methods=["all"])
        with self.assertRaisesRegex(ValueError, "existing artifacts"):
            pipeline.smoke(occupied)
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertEqual(list(occupied.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()

"""Runner contract tests with synthetic files/fake processes, not ML evidence."""
import contextlib
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cbmjev.cli import parser as cbmjev_parser
from cbmjev.contracts import Concept, QueryGroup, Schema
from scripts import run_vision


UUID = "GPU-00000000-0000-0000-0000-000000000005"


class VisionRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cbmjev-vision-runner-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prepared, self.raw = self.root / "prepared", self.root / "raw"
        self.prepared.mkdir()
        self.raw.mkdir()
        self.out = self.root / "new-run"
        self.config = self.root / "config.json"
        self.config.write_text('{}\n', encoding="utf-8")
        self.schema = Schema("isic2018_task2_binary", 2,
                             (Concept("a", "visible a", ("absent", "present")),
                              Concept("b", "visible b", ("absent", "present"))),
                             (QueryGroup("a", (0,)), QueryGroup("b", (1,))))
        self.write_json(self.prepared / "schema.json", self.schema.to_dict())
        self.write_json(self.prepared / "audit.json", {
            "source_revision": "SYNTHETIC_RUNNER_FIXTURE_NOT_IMAGE_DATA", "status": "PREPARED_NOT_ACCEPTED"})
        self.rows, self.membership = [], []
        for index, role in enumerate(run_vision.ROLES):
            sid = "synthetic-" + role
            image_name = sid + ".jpg"
            (self.raw / image_name).write_bytes(b"synthetic non-image bytes; preflight does not decode")
            self.rows.append({"sample_id": sid, "group_id": sid,
                              "input": {"modality": "image", "text": None, "image_paths": [image_name]}})
            self.membership.append({"sample_id": sid, "group_id": sid, "split": role})
        self.save_rows()
        self.save_membership()

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def save_rows(self):
        (self.prepared / "samples.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")

    def save_membership(self):
        (self.prepared / "membership.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.membership), encoding="utf-8")

    def arguments(self, *extra):
        return ["--prepared", str(self.prepared), "--raw-root", str(self.raw), "--out", str(self.out),
                "--config", str(self.config), "--random-init", "--python", sys.executable, *extra]

    def plan(self, *extra):
        with patch.dict("os.environ", {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            return run_vision.build_plan(run_vision.parser().parse_args(self.arguments(*extra)))

    def fake_run(self, plan, fail_stage=None, missing_stage=None, interrupt_stage=None, mutate_stage=None):
        calls = []

        def launch(command, **kwargs):
            stage = plan["stages"][len(calls)]
            calls.append((stage["name"], command, kwargs["env"]))
            self.assertTrue(kwargs["start_new_session"])
            kwargs["stdout"].write(b"FAKE_PROCESS_ENGINEERING_TEST_NOT_TRAINING\n")
            if stage["name"] != missing_stage:
                for value in stage["expected_artifacts"]:
                    path = Path(value)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"FAKE_PROCESS_ARTIFACT_NOT_MODEL_RESULT\n")
            if stage["name"] == mutate_stage:
                self.config.write_text('{"seed": 19}\n', encoding="utf-8")

            class Child:
                pid = -999999

                def poll(self):
                    return None if stage["name"] == interrupt_stage else 0

                def wait(self):
                    if stage["name"] == interrupt_stage:
                        raise run_vision.RunInterrupted(signal.SIGTERM)
                    return 17 if stage["name"] == fail_stage else 0

            return Child()

        with patch.object(run_vision.subprocess, "Popen", side_effect=launch), \
                patch.object(run_vision, "resolve_gpu", return_value=UUID), \
                patch.object(run_vision, "terminate_child") as terminate, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = run_vision.execute(plan)
        return code, calls, terminate

    def test_binary_dry_run_shell_is_nonmutating_and_does_not_query_gpu(self):
        completed = subprocess.run(["bash", str(run_vision.PROJECT / "scripts/run_vision.sh"),
                                    *self.arguments("--device", "cuda:0", "--gpu", "99999", "--dry-run")],
                                   check=True, capture_output=True, text=True)
        plan = json.loads(completed.stdout)
        self.assertEqual(plan["data"]["dataset"], "isic2018_task2_binary")
        self.assertEqual(plan["data"]["adapter_status"], "PREPARED_NOT_ACCEPTED")
        self.assertEqual(plan["initialization"], "EXPLICIT_RANDOM")
        self.assertEqual(len(plan["stages"]), 7)
        self.assertFalse(plan["test_task_evaluated"])
        self.assertFalse(self.out.exists())
        self.assertIn("test automatic responses", plan["cache_scope"])
        for stage in plan["stages"]:
            parsed = cbmjev_parser().parse_args(stage["arguments"])
            if parsed.command in {"evaluate", "audit-responses"}:
                self.assertEqual(parsed.split, "validation")

    def test_supported_visual_datasets_and_live_commands(self):
        for dataset in ("cub", "isic2018_task2", "isic2018_task2_binary", "derm7pt"):
            data = self.schema.to_dict()
            data["dataset"] = dataset
            self.write_json(self.prepared / "schema.json", data)
            plan = self.plan("--live")
            self.assertEqual(len(plan["stages"]), 8)
            live = next(stage for stage in plan["stages"] if stage["name"] == "validation_live")
            self.assertEqual(cbmjev_parser().parse_args(live["arguments"]).warmup, 3)

    def test_existing_output_and_broken_symlink_are_never_reused(self):
        self.out.mkdir()
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            self.plan()
        link = self.root / "broken-link"
        link.symlink_to(self.root / "missing-target")
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            self.plan("--out", str(link))

    def test_checkpoint_must_be_explicit_and_exist(self):
        args = self.arguments()
        args.remove("--random-init")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            run_vision.parser().parse_args(args)
        with self.assertRaisesRegex(ValueError, "existing nonempty"):
            run_vision.build_plan(run_vision.parser().parse_args(args + ["--backbone", str(self.root / "missing.pt")]))
        checkpoint = self.root / "local.pt"
        checkpoint.write_bytes(b"FAKE_LOCAL_CHECKPOINT_PREFLIGHT_ONLY")
        plan = run_vision.build_plan(run_vision.parser().parse_args(args + ["--backbone", str(checkpoint)]))
        self.assertEqual(plan["initialization"], "LOCAL_CHECKPOINT")
        self.assertIn(str(checkpoint.resolve()), plan["tracked_files"])

    def test_bad_gpu_selection_or_memory_fails_before_creating_output(self):
        for extra in (("--device", "cuda:0"), ("--gpu", "5"),
                      ("--device", "cuda:0", "--gpu", "5,6"),
                      ("--gpu-memory-gib", "nan"), ("--gpu-reserve-gib", "0")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.plan(*extra)
        self.assertFalse(self.out.exists())

    def test_roles_path_and_modality_are_checked_before_launch(self):
        self.membership[1]["group_id"] = self.membership[0]["group_id"]
        self.save_membership()
        with self.assertRaisesRegex(ValueError, "group crossing"):
            self.plan()
        self.membership[1]["group_id"] = self.membership[1]["sample_id"]
        self.save_membership()
        self.rows[0]["input"]["image_paths"] = ["../escape.jpg"]
        self.save_rows()
        with self.assertRaisesRegex(ValueError, "relative"):
            self.plan()
        self.rows[0]["input"] = {"modality": "text", "text": "not an image"}
        self.save_rows()
        with self.assertRaisesRegex(ValueError, "image-only"):
            self.plan()

    def test_strict_all_budget_is_rejected_in_preflight(self):
        self.write_json(self.config, {"policy": {"max_groups": 1}})
        with self.assertRaisesRegex(ValueError, "full query-group budget"):
            self.plan()

    def test_successful_fake_chain_records_complete_stages_and_cpu_audit_isolation(self):
        plan = self.plan("--device", "cuda:0", "--gpu", "5", "--live", "--gpu-memory-gib", "6")
        code, calls, _ = self.fake_run(plan)
        self.assertEqual(code, 0)
        complete = run_vision.read_json(self.out / "COMPLETED.json")
        self.assertEqual(complete["completed_stages"], [stage["name"] for stage in plan["stages"]])
        self.assertEqual(complete["status"], "COMPLETED")
        self.assertEqual(run_vision.read_json(self.out / "plan.json")["resolved_gpu_uuid"], UUID)
        self.assertFalse((self.out / "FAILED.json").exists())
        self.assertEqual(len(list((self.out / "logs").glob("*.log"))), 8)
        for name, command, env in calls:
            if name in {"semantic_audit", "summarize"}:
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
                self.assertNotIn(str(run_vision.PROJECT / "tools/run_gpu_limited.py"), command)
            else:
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], UUID)
                self.assertIn(str(run_vision.PROJECT / "tools/run_gpu_limited.py"), command)
                self.assertEqual(command[command.index("--memory-gib") + 1], "6.0")
        events = list(run_vision.iter_jsonl(self.out / "events.jsonl"))
        self.assertEqual(events[-1]["event"], "CHAIN_COMPLETED")

    def test_failed_stage_stops_and_keeps_logs_without_completion(self):
        code, calls, _ = self.fake_run(self.plan(), fail_stage="cache")
        self.assertEqual(code, 17)
        self.assertEqual([row[0] for row in calls], ["doctor", "responder", "cache"])
        self.assertEqual(run_vision.read_json(self.out / "FAILED.json")["stage"], "cache")
        self.assertFalse((self.out / "COMPLETED.json").exists())
        self.assertTrue((self.out / "logs/03_cache.log").read_bytes())

    def test_zero_exit_without_expected_artifact_is_not_success(self):
        code, calls, _ = self.fake_run(self.plan(), missing_stage="responder")
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 2)
        self.assertIn("missing/empty", run_vision.read_json(self.out / "FAILED.json")["error"])

    def test_interruption_records_terminal_status_and_terminates_only_child(self):
        code, calls, terminate = self.fake_run(self.plan(), interrupt_stage="cache")
        self.assertEqual(code, 143)
        terminate.assert_called_once()
        self.assertIsNotNone(terminate.call_args.args[0])
        self.assertEqual(len(calls), 3)
        self.assertTrue((self.out / "INTERRUPTED.json").is_file())
        self.assertFalse((self.out / "COMPLETED.json").exists())

    def test_changed_config_midchain_fails_closed(self):
        code, calls, _ = self.fake_run(self.plan(), mutate_stage="doctor")
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)
        self.assertIn("changed during chain", run_vision.read_json(self.out / "FAILED.json")["error"])


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest


PROJECT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = PROJECT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SprintAutomationScriptsTest(unittest.TestCase):
    def test_canonical_fixed_monitor_fails_closed_before_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            launcher = scripts / "monitor_cub_value_seed.sh"
            shutil.copy2(PROJECT / "scripts/monitor_cub_value_seed.sh", launcher)
            syntax = subprocess.run(["bash", "-n", str(launcher)], capture_output=True, text=True)
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
            disallowed = subprocess.run(["bash", str(launcher), "60", "5", "canonical_fixed_only"],
                                        capture_output=True, text=True)
            self.assertEqual(disallowed.returncode, 3)
            self.assertIn("only physical GPU 0/1", disallowed.stderr)
            missing = subprocess.run(["bash", str(launcher), "60", "0", "canonical_fixed_only"],
                                     capture_output=True, text=True)
            self.assertEqual(missing.returncode, 2)
            self.assertIn("missing source", missing.stderr)
            self.assertFalse((root / "runs").exists())

    def test_progress_report_counts_target_epochs(self):
        progress = load_script("report_cub_value_progress.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs/cub_outer0_value_seed60_v1"
            target = run / "action_targets_records"
            target.mkdir(parents=True)
            (run / "config.json").write_text(
                json.dumps({"learning": {"policy_epochs": 4}}), encoding="utf-8"
            )
            now = time.time()
            for epoch, age in [(0, 120), (1, 60)]:
                path = target / f"epoch_{epoch:03d}.jsonl"
                path.write_text("{}\n", encoding="utf-8")
                os.utime(path, (now - age, now - age))
            (root / "logs").mkdir()
            (root / "logs/cub_outer0_value_seed60_v1.pid").write_text(
                str(os.getpid()), encoding="utf-8"
            )

            report = progress.build_report(root, [60])
            row = report["folds"][0]
            self.assertEqual(row["expected_policy_epochs"], 4)
            self.assertEqual(row["latest_epoch"], 1)
            self.assertEqual(row["completed_epoch_files"], 2)
            self.assertAlmostEqual(row["progress_percent"], 50.0)
            self.assertGreater(row["eta_seconds_after_target_phase"], 0)

    def test_paper_update_keeps_evidence_state_draft(self):
        updater = load_script("prepare_cub_value_paper_update.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aggregate = root / "results/main/cub_value_budget_grid_60_61_62"
            paper = root / "paper/cvpr2027"
            generated = paper / "generated"
            aggregate.mkdir(parents=True)
            generated.mkdir(parents=True)
            summary = {
                "seeds": [60, 61, 62],
                "num_seeds": 3,
                "num_rows": 4,
                "policies": {
                    "value_b4": {
                        "method": "value",
                        "budget_groups": 4,
                        "accuracy_mean": 0.40,
                        "macro_f1_mean": 0.31,
                        "num_seeds": 3,
                    },
                    "fixed_b4": {
                        "method": "fixed",
                        "budget_groups": 4,
                        "accuracy_mean": 0.38,
                        "macro_f1_mean": 0.30,
                        "num_seeds": 3,
                    },
                },
            }
            (aggregate / "frontier_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (aggregate / "frontier_long.csv").write_text(
                "seed,method,budget_groups,accuracy,macro_f1\n", encoding="utf-8"
            )
            for metric_name in ("accuracy", "macro_f1"):
                (aggregate / f"claim_analysis_{metric_name}.json").write_text(
                    json.dumps(
                        {
                            "status": "weak_or_budget_local_dynamic_gain",
                            "comparisons": [{"delta_vs_best_static_or_fixed": 0.02}],
                        }
                    ),
                    encoding="utf-8",
                )
                (aggregate / f"paired_seed_deltas_{metric_name}.json").write_text(
                    json.dumps(
                        {
                            "status": "paired_weak_or_mixed_gain",
                            "budget_reports": [{"mean_delta": 0.02, "num_seed_pairs": 3}],
                        }
                    ),
                    encoding="utf-8",
                )
            (generated / "cub_value_policy_manifest.json").write_text(
                json.dumps({"schema_version": "test"}), encoding="utf-8"
            )
            (generated / "cub_value_policy_table.tex").write_text("% table\n", encoding="utf-8")
            state_path = paper / "evidence_state.json"
            state_path.write_text(
                json.dumps({"status": "DRAFT_NOT_SUBMISSION_READY", "development_results": []}),
                encoding="utf-8",
            )

            report = updater.build_recommendation(root, aggregate, paper, "cub_value_policy")
            self.assertEqual(report["summary"]["accuracy_status"], "weak_or_budget_local_dynamic_gain")
            updated = updater.update_evidence_state(state_path, report["evidence_entry"])
            self.assertEqual(updated["status"], "DRAFT_NOT_SUBMISSION_READY")
            self.assertFalse(updated["empirical_results_promoted"])
            self.assertFalse(updated["three_seed_matched_baselines_audited"])
            self.assertEqual(len(updated["development_results"]), 1)

    def test_paper_stage_refreshes_delivery_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            figures = scripts / "figures"
            aggregate = root / "results/main/cub_value_budget_grid_60_61_62"
            paper = root / "paper/cvpr2027"
            aggregate.mkdir(parents=True)
            figures.mkdir(parents=True)
            paper.mkdir(parents=True)
            for rel in [
                "monitor_cub_value_paper_stage.sh",
                "prepare_cub_value_paper_update.py",
                "report_delivery_gates.py",
            ]:
                shutil.copy2(PROJECT / "scripts" / rel, scripts / rel)
            shutil.copy2(
                PROJECT / "scripts/figures/stage_cub_value_paper_assets.py",
                figures / "stage_cub_value_paper_assets.py",
            )
            summary = {
                "seeds": [60, 61, 62],
                "num_seeds": 3,
                "num_rows": 2,
                "policies": {
                    "value_b4": {
                        "method": "value",
                        "budget_groups": 4,
                        "accuracy_mean": 0.41,
                        "macro_f1_mean": 0.33,
                        "num_seeds": 3,
                    },
                    "fixed_b4": {
                        "method": "fixed",
                        "budget_groups": 4,
                        "accuracy_mean": 0.39,
                        "macro_f1_mean": 0.31,
                        "num_seeds": 3,
                    },
                },
            }
            (aggregate / "frontier_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (aggregate / "frontier_long.csv").write_text(
                "seed,method,budget_groups,accuracy,macro_f1\n", encoding="utf-8"
            )
            (aggregate / "frontier_table.tex").write_text("% table\n", encoding="utf-8")
            (aggregate / "frontier_accuracy.pdf").write_bytes(b"%PDF-1.4\n% test accuracy\n")
            (aggregate / "frontier_macro_f1.pdf").write_bytes(b"%PDF-1.4\n% test macro\n")
            for metric_name in ("accuracy", "macro_f1"):
                (aggregate / f"claim_analysis_{metric_name}.json").write_text(
                    json.dumps(
                        {
                            "status": "weak_or_budget_local_dynamic_gain",
                            "comparisons": [{"delta_vs_best_static_or_fixed": 0.02}],
                        }
                    ),
                    encoding="utf-8",
                )
                (aggregate / f"claim_analysis_{metric_name}.md").write_text(
                    "# claim\n", encoding="utf-8"
                )
                (aggregate / f"paired_seed_deltas_{metric_name}.json").write_text(
                    json.dumps(
                        {
                            "status": "paired_weak_or_mixed_gain",
                            "budget_reports": [{"mean_delta": 0.02, "num_seed_pairs": 3}],
                        }
                    ),
                    encoding="utf-8",
                )
                (aggregate / f"paired_seed_deltas_{metric_name}.md").write_text(
                    "# paired\n", encoding="utf-8"
                )
            (paper / "evidence_state.json").write_text(
                json.dumps({"status": "DRAFT_NOT_SUBMISSION_READY", "development_results": []}),
                encoding="utf-8",
            )

            subprocess.run(
                ["bash", "scripts/monitor_cub_value_paper_stage.sh", "60_61_62", "cub_value_policy"],
                cwd=root,
                env={**os.environ, "PYTHON_BIN": "python3"},
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertTrue((paper / "generated/cub_value_policy_manifest.json").is_file())
            self.assertTrue((root / "results/main/cub_value_policy_paper_update_60_61_62.json").is_file())
            gates = root / "results/main/delivery_gates_20260923.json"
            self.assertTrue(gates.is_file())
            report = json.loads(gates.read_text(encoding="utf-8"))
            value_gate = [
                gate for gate in report["gates"] if gate["gate_id"] == "cub_value_policy_60_61_62"
            ][0]
            self.assertTrue(value_gate["passed"])

    def test_paired_seed_delta_analysis_uses_matched_seed_budget_pairs(self):
        paired = load_script("analyze_cub_value_seed_deltas.py")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frontier_long.csv"
            path.write_text(
                "\n".join([
                    "source,seed,policy_id,method,budget_groups,accuracy,macro_f1,mean_queried_groups,mean_calls,mean_declared_cost,num_samples",
                    "a,1,value_K4,value,4,0.60,0.50,4,4,4,10",
                    "a,1,fixed_K4,fixed,4,0.55,0.45,4,4,4,10",
                    "a,1,static_K4,static,4,0.58,0.46,4,4,4,10",
                    "a,2,value_K4,value,4,0.52,0.42,4,4,4,10",
                    "a,2,fixed_K4,fixed,4,0.54,0.44,4,4,4,10",
                    "a,2,static_K4,static,4,0.50,0.40,4,4,4,10",
                    "a,1,value_K8,value,8,0.70,0.60,8,8,8,10",
                    "a,1,fixed_K8,fixed,8,0.63,0.55,8,8,8,10",
                    "a,2,value_K8,value,8,0.72,0.62,8,8,8,10",
                    "a,2,fixed_K8,fixed,8,0.66,0.56,8,8,8,10",
                ]) + "\n",
                encoding="utf-8",
            )

            report = paired.analyze(paired.load_rows(path), "accuracy", min_delta=0.005,
                                    baseline_methods=("static", "fixed"))
            by_budget = {row["budget_groups"]: row for row in report["budget_reports"]}
            self.assertAlmostEqual(by_budget[4]["mean_delta"], (0.02 - 0.02) / 2)
            self.assertEqual(by_budget[4]["positive_seed_pairs"], 1)
            self.assertEqual(by_budget[4]["negative_seed_pairs"], 1)
            self.assertAlmostEqual(by_budget[8]["mean_delta"], (0.07 + 0.06) / 2)
            self.assertEqual(by_budget[8]["negative_seed_pairs"], 0)
            self.assertEqual(report["status"], "paired_positive_candidate")


if __name__ == "__main__":
    unittest.main()

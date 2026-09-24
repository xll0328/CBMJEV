"""Artifact-only paper exporter tests; no models, datasets, or network required."""
import copy
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import unittest

from tools.build_paper_results import (AUDIT_FORMAT, SWEEP_FORMAT, SWEEP_METRICS,
                                      build, latex_escape, load_audit, load_summary)


class PaperResultBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding="utf-8")
        return path

    def sweep(self, seeds=(31,), dataset=None):
        rows = []
        for i, seed in enumerate(seeds):
            rows.append({"run": f"run{seed}", "seed": seed, "mode": "offline_replay",
                         "policy_id": "value", "method": "value", "cost_weight": 0.03,
                         "max_groups": 4, "accuracy": 0.6 + i * 0.1, "macro_f1": 0.5,
                         "group_mean_risk": 0.4 - i * 0.1, "mean_queried_groups": 2.0,
                         "mean_calls": 1.5, "mean_declared_cost": 2.0,
                         "num_samples": 100, "num_groups": 20,
                         "evidence_status": "OFFLINE_REPLAY_NOT_LATENCY"})
        aggregate = {"method": "value", "cost_weight": 0.03, "max_groups": 4,
                     "num_seeds": len(seeds), "seeds": list(seeds)}
        for metric in SWEEP_METRICS:
            values = [r[metric] for r in rows]
            aggregate[metric + "_mean"] = statistics.mean(values)
            aggregate[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        doc = {"format": SWEEP_FORMAT, "paper_claim": False, "points": rows, "aggregate": [aggregate]}
        if dataset:
            doc.update(dataset=dataset, split="validation")
        return doc

    def summary_spec(self, doc=None, name="summary.json", label="cebab/hash_v1"):
        return f"{label}={self.write(self.root / name, doc or self.sweep())}"

    def audit_run(self, name="run40", seed=40, accuracy=0.8, dataset="cebab"):
        directory = self.root / name
        schema = {"schema_version": "cbmjev-schema-v1", "dataset": dataset,
                  "num_classes": 2, "concepts": [{"id": "food", "description": "food sentiment",
                                                  "values": ["negative", "positive"]}],
                  "groups": [{"id": "food", "atoms": [0]}]}
        schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True, ensure_ascii=False,
                                               separators=(",", ":")).encode()).hexdigest()
        hashes = {"prepared_samples_sha256": "a" * 64, "membership_sha256": "b" * 64,
                  "responder_checkpoint_sha256": hashlib.sha256(str(seed).encode()).hexdigest()}
        audit = {"format": AUDIT_FORMAT, "status": "COMPLETED", "dataset": dataset,
                 "split": "validation", "test_evaluated": False,
                 "response_source": "automatic_model",
                 "evidence_status": "DESCRIPTIVE_SEMANTIC_AUDIT_NOT_PAPER_OR_LATENCY_EVIDENCE",
                 "schema_hash": schema_hash, "source_hashes": hashes,
                 "selection": {"audit_population": "cached_task_labelled_subset_of_requested_split",
                               "is_whole_dataset_concept_rate": False,
                               "audited_sample_ids_hash": "c" * 64, "audited_cached_split_cases": 100},
                 "overall": {"macro_concept_accuracy": accuracy, "macro_concept_f1": accuracy - 0.1,
                             "macro_averaging": "equal_weight_concepts_with_positive_observed_gold_support",
                             "runtime_nonanswers_are_errors_on_observed_gold": True,
                             "num_concepts_with_observed_gold": 1},
                 "concepts": [{"concept_id": "food", "cached_observed_gold_count": 100}]}
        training = {"split": "responder_fit", "seed": seed, "paper_evidence": False,
                    "task_label_gradient": False}
        receipt = {"kind": "hf_text", "seed": seed, "schema_hash": schema_hash,
                   "prepared_samples_sha256": hashes["prepared_samples_sha256"],
                   "membership_sha256": hashes["membership_sha256"],
                   "checkpoint_sha256": hashes["responder_checkpoint_sha256"],
                   "source_revision": "public-revision-test-fixture",
                   "data_evidence": "USER_SUPPLIED_PUBLIC_DATA_NOT_INDEPENDENTLY_AUTHENTICATED",
                   "initialization": {"revision": "pinned"}}
        for suffix, obj in (("semantic_audit/audit.json", audit), ("responder/training.json", training),
                            ("responder/receipt.json", receipt), ("responder/schema.json", schema)):
            self.write(directory / suffix, obj)
        return directory, audit, training, receipt

    def test_single_seed_export_preserves_facts_and_no_claim(self):
        spec = self.summary_spec()
        out = self.root / "out"
        manifest = build([spec], [], out)
        evidence = json.loads((out / "claim_evidence.json").read_text())
        self.assertFalse(manifest["paper_claim"])
        self.assertFalse(evidence["formal_results_modified"])
        self.assertEqual(evidence["rows"][0]["accuracy"], 0.6)
        self.assertIsNone(evidence["aggregate"][0]["metrics"]["accuracy"]["sd"])
        self.assertEqual(evidence["claims"]["C1"]["claim_supported"], "no")
        self.assertEqual(manifest["sources"][0]["split_basis"], "versioned_format_contract")
        self.assertEqual(manifest["sources"][0]["dataset_basis"], "caller_asserted_label")
        self.assertIn("SINGLE-SEED DEVELOPMENT-ONLY", (out / "exploratory_tables.tex").read_text())
        for name, digest in manifest["outputs"].items():
            self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), digest)

    def test_multi_seed_multi_dataset_stay_separate(self):
        specs = [self.summary_spec(self.sweep((1,)), "s1.json"),
                 self.summary_spec(self.sweep((2,)), "s2.json"),
                 self.summary_spec(self.sweep((1, 2, 3), "cub"), "cub.json", "cub/vision_v1")]
        build(specs, [], self.root / "out")
        rows = json.loads((self.root / "out/claim_evidence.json").read_text())["aggregate"]
        self.assertEqual([r["num_seeds"] for r in rows], [2, 3])
        self.assertEqual([r["dataset"] for r in rows], ["cebab", "cub"])
        self.assertTrue(all(r["development_only"] for r in rows))

    def test_reject_invalid_summary_metadata_without_output(self):
        changes = [("split", "test"), ("split", "calibration"), ("paper_claim", True),
                   ("paper_claim", 0), ("paper_claim", None), ("format", "unknown"),
                   ("dataset", "cub"), ("num_seeds", 3), ("seeds", [31, 32])]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                doc = self.sweep()
                doc[key] = value
                out = self.root / "invalid"
                with self.assertRaises(ValueError):
                    build([self.summary_spec(doc)], [], out)
                self.assertFalse(out.exists())

    def test_reject_bad_point_values(self):
        for key, value in (("accuracy", float("nan")), ("accuracy", float("inf")),
                           ("macro_f1", 1.1), ("accuracy", True), ("seed", True),
                           ("seed", 1.5), ("num_samples", 0), ("num_groups", 101),
                           ("cost_weight", -1), ("mean_queried_groups", 5),
                           ("split", "test"), ("paper_claim", True)):
            with self.subTest(key=key, value=value):
                doc = self.sweep()
                doc["points"][0][key] = value
                with self.assertRaises(ValueError):
                    load_summary(self.summary_spec(doc))

    def test_reject_forged_aggregate_and_seed_count(self):
        for key, value in (("accuracy_mean", .99), ("num_seeds", 3),
                           ("seeds", [31, 31]), ("accuracy_sd", 0)):
            with self.subTest(key=key):
                doc = self.sweep()
                doc["aggregate"][0][key] = value
                with self.assertRaises(ValueError):
                    load_summary(self.summary_spec(doc))

    def test_missing_aggregate_metric_is_not_imputed(self):
        doc = self.sweep()
        del doc["aggregate"][0]["accuracy_sd"]
        with self.assertRaisesRegex(ValueError, "missing"):
            load_summary(self.summary_spec(doc))

    def test_duplicate_source_seed_rejected(self):
        spec = self.summary_spec()
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            build([spec, spec], [], self.root / "out")

    def test_different_population_rejected(self):
        doc = self.sweep((32,))
        doc["points"][0]["num_samples"] = 99
        with self.assertRaisesRegex(ValueError, "incomparable population"):
            build([self.summary_spec(), self.summary_spec(doc, "other.json")], [], self.root / "out")

    def test_different_backend_metadata_rejected(self):
        first, second = self.sweep((31,)), self.sweep((32,))
        first["backend"], second["backend"] = "hash", "hf"
        with self.assertRaisesRegex(ValueError, "incomparable"):
            build([self.summary_spec(first), self.summary_spec(second, "other.json")], [], self.root / "out")

    def test_point_level_protocol_conflicts_not_ignored(self):
        for field in ("schema_hash", "source_revision", "backend"):
            doc = self.sweep()
            doc[field], doc["points"][0][field] = "one", "two"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "point/top-level"):
                load_summary(self.summary_spec(doc))
        first, second = self.sweep((31,)), self.sweep((32,))
        first["points"][0]["backend"], second["points"][0]["backend"] = "hash", "hf"
        with self.assertRaisesRegex(ValueError, "incomparable"):
            build([self.summary_spec(first), self.summary_spec(second, "other.json")], [], self.root / "out")

    def test_synthetic_summary_rejected_if_disclosed(self):
        doc = self.sweep()
        doc["source_revision"] = "SYNTHETIC_SMOKE"
        with self.assertRaisesRegex(ValueError, "synthetic summary"):
            load_summary(self.summary_spec(doc))

    def test_duplicate_json_key_rejected(self):
        path = self.root / "bad.json"
        path.write_text('{"paper_claim":false,"paper_claim":true}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            load_summary(f"cebab/cohort={path}")

    def test_audit_multi_seed_exact_metrics(self):
        first, *_ = self.audit_run(accuracy=.8)
        second, *_ = self.audit_run("run41", 41, .9)
        build([], [f"hf_v1={first}", f"hf_v1={second}"], self.root / "out")
        aggregate = json.loads((self.root / "out/claim_evidence.json").read_text())["aggregate"][0]
        self.assertEqual(aggregate["seed_unit"], "responder_training_seed")
        self.assertEqual(aggregate["seeds"], [40, 41])
        self.assertAlmostEqual(aggregate["metrics"]["macro_concept_accuracy"]["mean"], .85)
        self.assertAlmostEqual(aggregate["metrics"]["macro_concept_accuracy"]["sd"], statistics.stdev([.8, .9]))
        self.assertAlmostEqual(aggregate["metrics"]["macro_concept_f1"]["mean"], .75)

    def test_audit_reject_split_missing_metric_and_synthetic(self):
        directory, original, training, receipt = self.audit_run()
        cases = [lambda d: d.update(split="test"), lambda d: d.update(test_evaluated=True),
                 lambda d: d.update(status="NO_OBSERVED_CONCEPT_GOLD"),
                 lambda d: d["overall"].update(macro_concept_accuracy=None),
                 lambda d: d["overall"].update(macro_concept_f1=True),
                 lambda d: d["source_hashes"].update(responder_checkpoint_sha256="d" * 64)]
        for mutate in cases:
            doc = copy.deepcopy(original)
            mutate(doc)
            self.write(directory / "semantic_audit/audit.json", doc)
            with self.assertRaises(ValueError):
                load_audit(f"hf_v1={directory}")
        self.write(directory / "semantic_audit/audit.json", original)
        receipt["data_evidence"] = "SYNTHETIC_FIXTURE"
        self.write(directory / "responder/receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "synthetic/unknown"):
            load_audit(f"hf_v1={directory}")

    def test_audit_training_seed_and_schema_must_match(self):
        directory, _, training, _ = self.audit_run()
        training["seed"] = 41
        self.write(directory / "responder/training.json", training)
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            load_audit(f"hf_v1={directory}")
        training["seed"] = 40
        self.write(directory / "responder/training.json", training)
        self.write(directory / "responder/schema.json", {"dataset": "cub"})
        with self.assertRaisesRegex(ValueError, "dataset mismatch"):
            load_audit(f"hf_v1={directory}")

    def test_audit_duplicate_seed_and_population_rejected(self):
        first, *_ = self.audit_run()
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            build([], [f"hf_v1={first}", f"hf_v1={first}"], self.root / "out")
        second, audit, *_ = self.audit_run("run41", 41)
        audit["selection"]["audited_sample_ids_hash"] = "d" * 64
        self.write(second / "semantic_audit/audit.json", audit)
        with self.assertRaisesRegex(ValueError, "audited_sample_ids_hash"):
            build([], [f"hf_v1={first}", f"hf_v1={second}"], self.root / "out")

    def test_audit_schema_mismatch_within_cohort_rejected(self):
        first, *_ = self.audit_run()
        second, audit, _, receipt = self.audit_run("run41", 41)
        schema = json.loads((second / "responder/schema.json").read_text())
        schema["concepts"][0]["description"] = "different protocol"
        digest = hashlib.sha256(json.dumps(schema, sort_keys=True, ensure_ascii=False,
                                          separators=(",", ":")).encode()).hexdigest()
        audit["schema_hash"] = receipt["schema_hash"] = digest
        self.write(second / "responder/schema.json", schema)
        self.write(second / "semantic_audit/audit.json", audit)
        self.write(second / "responder/receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "incomparable schema_hash"):
            build([], [f"hf_v1={first}", f"hf_v1={second}"], self.root / "out")

    def test_audit_different_training_protocol_and_code_rejected(self):
        first, _, training_first, receipt_first = self.audit_run()
        second, _, training_second, receipt_second = self.audit_run("run41", 41)
        for field in ("epochs", "batch_size", "learning_rate", "max_length", "freeze_backbone"):
            with self.subTest(field=field):
                one, two = copy.deepcopy(training_first), copy.deepcopy(training_second)
                one[field], two[field] = 1, 2
                self.write(first / "responder/training.json", one)
                self.write(second / "responder/training.json", two)
                with self.assertRaisesRegex(ValueError, "incomparable training_protocol"):
                    build([], [f"hf_v1={first}", f"hf_v1={second}"], self.root / "out")
        self.write(first / "responder/training.json", training_first)
        self.write(second / "responder/training.json", training_second)
        receipt_first["semantic_code_hash"], receipt_second["semantic_code_hash"] = "a" * 64, "b" * 64
        self.write(first / "responder/receipt.json", receipt_first)
        self.write(second / "responder/receipt.json", receipt_second)
        with self.assertRaisesRegex(ValueError, "incomparable semantic_code_hash"):
            build([], [f"hf_v1={first}", f"hf_v1={second}"], self.root / "out")

    def test_output_never_overwritten_and_empty_rejected(self):
        out = self.root / "out"
        with self.assertRaisesRegex(ValueError, "at least one"):
            build([], [], out)
        self.assertFalse(out.exists())
        out.mkdir()
        self.write(out / "user.json", {"preserve": True})
        with self.assertRaises(FileExistsError):
            build([self.summary_spec()], [], out)
        self.assertTrue((out / "user.json").exists())

    def test_tex_escaping_prevents_commands(self):
        text = latex_escape(r"name_1 & 50% \input{secret}")
        self.assertIn(r"\textbackslash{}input\{secret\}", text)
        self.assertIn(r"name\_1 \& 50\%", text)


if __name__ == "__main__":
    unittest.main()

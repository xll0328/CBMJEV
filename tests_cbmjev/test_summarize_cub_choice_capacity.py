from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, read_json, read_jsonl, write_json, write_jsonl
from scripts.summarize_cub_choice_capacity import MATCHED_FIT_KEYS, summarize_capacity
from tests_cbmjev.test_summarize_cub_choice_pair import evaluation, trace


def fit(directory, *, capacity, shared):
    directory.mkdir()
    names = ("scalar", "attention", "independent_mlp") if capacity else ("scalar", "attention")
    for name in names:
        (directory / (name + ".pt")).write_bytes(name.encode())
    details = {key: "same:" + key for key in MATCHED_FIT_KEYS}
    details["heads"] = {name: {"final_weights_sha256": "weight:" + name,
                                "initial_weights_sha256": "initial:" + name}
                        for name in names}
    if capacity:
        details["capacity_control"] = {"same_visible_features_and_log_candidate_count": True,
            "parameter_count": {"scalar": 10, "attention": 66187,
                                "independent_mlp": 66571}}
    report = {"selected_derived_ids": ["sample"], "package_bindings": [shared], "fit": details}
    write_json(directory / "config.json", {"capacity_control": capacity})
    write_json(directory / "report.json", report)
    files = {name: file_hash(directory / name) for name in
             (*(head + ".pt" for head in names), "config.json", "report.json")}
    receipt = {"format": ("cbmjev-structured-choice-capacity-artifact-v1" if capacity
                          else "cbmjev-structured-choice-pair-artifact-v1"),
               "status": "COMPLETE", "files": files}
    receipt["receipt_sha256"] = stable_hash(receipt)
    write_json(directory / "receipt.json", receipt)


def capacity_evaluation(directory, *, binding, fit_dir):
    directory.mkdir()
    source = {"nested_system": binding,
              "choice_receipt_sha256": file_hash(fit_dir / "receipt.json")}
    names = ("scalar", "attention", "independent_mlp")
    metrics = {"split": "validation", "mode": "offline_replay", "seed": 60,
               "source_binding": source,
               "policies": {"structured_choice_" + name: {} for name in names}}
    write_json(directory / "metrics.json", metrics)
    rows = [trace(i, policy_id="structured_choice_" + name,
                  method="structured_choice", queries=queries)
            for name, queries in zip(names, (0, 1, 2)) for i in range(594)]
    specs = {}
    for name in names:
        policy_id = "structured_choice_" + name
        spec = {"max_groups": 16, "split": "validation", "mode": "offline_replay",
                "source_binding": source, "head_weights_sha256": "weight:" + name,
                "system_hash": "system:" + name}
        specs[policy_id] = spec
        for row in rows:
            if row["policy_id"] == policy_id:
                row["system_hash"] = spec["system_hash"]
    write_json(directory / "settings.json", specs)
    write_jsonl(directory / "traces.jsonl", rows)
    receipt = {"format": "cbmjev-choice-crossfit-validation-v1", "status": "COMPLETE",
               "paper_evidence": False, "source_binding": source, "num_policies": 3,
               "files_sha256": {name: file_hash(directory / name)
        for name in ("metrics.json", "settings.json", "traces.jsonl")}
              }
    receipt["receipt_hash"] = stable_hash(receipt)
    write_json(directory / "receipt.json", receipt)


class CapacitySummaryTests(unittest.TestCase):
    def test_matched_fit_and_exact_k16_summary(self):
        with TemporaryDirectory() as root_string:
            root = Path(root_string)
            capacity, soft, choice_eval, fixed = (root / name for name in
                ("capacity", "soft", "capacity_eval", "fixed"))
            binding = {"same": "nested"}
            fit(capacity, capacity=True, shared=binding)
            fit(soft, capacity=False, shared=binding)
            capacity_evaluation(choice_eval, binding=binding, fit_dir=capacity)
            evaluation(fixed, choice=False, binding=binding)
            result = summarize_capacity(capacity, choice_eval, soft, fixed,
                                        draws=100, seed=60)
            self.assertTrue(result["same_training_examples_features_order_and_original_pair_weights_verified"])
            self.assertEqual(result["metrics"]["independent_mlp"]["mean_queried_groups"], 2)
            self.assertEqual(result["metrics"]["fixed_K16"]["mean_queried_groups"], 16)
            self.assertIn("attention_minus_independent_mlp", result["comparisons"])
            with (capacity / "report.json").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                summarize_capacity(capacity, choice_eval, soft, fixed,
                                   draws=100, seed=60)


if __name__ == "__main__":
    unittest.main()

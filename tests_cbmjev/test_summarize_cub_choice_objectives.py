from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, read_json, read_jsonl, write_json, write_jsonl
from scripts.summarize_cub_choice_objectives import summarize_objectives
from tests_cbmjev.test_summarize_cub_choice_pair import evaluation, trace


def risk_evaluation(directory, nested, soft_receipt_sha256):
    directory.mkdir()
    rows = []
    rows.extend(trace(i, policy_id="structured_choice_risk_scalar",
                      method="structured_choice", queries=2) for i in range(594))
    rows.extend(trace(i, policy_id="structured_choice_risk_attention",
                      method="structured_choice", queries=3) for i in range(594))
    binding = {"nested_system": nested,
               "soft_choice_receipt_sha256": soft_receipt_sha256}
    cost = {"setup": 0., "call": 0., "per_group": 1.}
    settings = {name: {"max_groups": 16, "mode": "offline_replay",
        "split": "validation", "runtime_method": "structured_choice",
        "source_binding": binding, "cost_weight": .03, "cost": cost,
        "system_hash": name + "-hash"}
        for name in ("structured_choice_risk_scalar", "structured_choice_risk_attention")}
    for row in rows:
        row["system_hash"] = settings[row["policy_id"]]["system_hash"]
    write_json(directory / "metrics.json", {"split": "validation", "mode": "offline_replay",
        "seed": 60, "source_binding": binding, "policies": {}})
    write_json(directory / "settings.json", settings)
    write_jsonl(directory / "traces.jsonl", rows)
    receipt = {"format": "cbmjev-choice-risk-crossfit-validation-v1",
        "status": "COMPLETE", "paper_evidence": False, "source_binding": binding,
        "files_sha256": {name: file_hash(directory / name)
            for name in ("metrics.json", "settings.json", "traces.jsonl")}}
    receipt["receipt_hash"] = stable_hash(receipt)
    write_json(directory / "receipt.json", receipt)


def reseal_evaluation(directory, settings):
    rows = read_jsonl(directory / "traces.jsonl")
    for row in rows:
        row["system_hash"] = settings[row["policy_id"]]["system_hash"]
    (directory / "traces.jsonl").unlink()
    write_jsonl(directory / "traces.jsonl", rows)
    (directory / "settings.json").unlink()
    write_json(directory / "settings.json", settings)
    receipt = read_json(directory / "receipt.json")
    receipt["files_sha256"].update({name: file_hash(directory / name)
        for name in ("settings.json", "traces.jsonl")})
    receipt.pop("receipt_hash")
    receipt["receipt_hash"] = stable_hash(receipt)
    (directory / "receipt.json").unlink()
    write_json(directory / "receipt.json", receipt)


class ChoiceObjectiveSummaryTests(unittest.TestCase):
    def test_same_system_objective_and_fixed_comparison(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            soft, risk, fixed = root / "soft", root / "risk", root / "fixed"
            nested = {"same": "nested"}
            evaluation(soft, choice=True, binding=nested)
            receipt = read_json(soft / "receipt.json")
            receipt["source_binding"]["choice_receipt_sha256"] = "source-soft-hash"
            receipt.pop("receipt_hash")
            receipt["receipt_hash"] = stable_hash(receipt)
            (soft / "receipt.json").unlink()
            write_json(soft / "receipt.json", receipt)
            metrics = read_json(soft / "metrics.json")
            metrics["source_binding"] = receipt["source_binding"]
            (soft / "metrics.json").unlink()
            write_json(soft / "metrics.json", metrics)
            receipt["files_sha256"]["metrics.json"] = file_hash(soft / "metrics.json")
            receipt.pop("receipt_hash")
            receipt["receipt_hash"] = stable_hash(receipt)
            (soft / "receipt.json").unlink()
            write_json(soft / "receipt.json", receipt)
            cost = {"setup": 0., "call": 0., "per_group": 1.}
            soft_settings = {name: {"max_groups": 16,
                "mode": "offline_replay", "split": "validation",
                "runtime_method": "structured_choice",
                "source_binding": receipt["source_binding"],
                "cost_weight": .03, "cost": cost, "system_hash": name + "-hash"}
                for name in ("structured_choice_scalar", "structured_choice_attention")}
            reseal_evaluation(soft, soft_settings)
            risk_evaluation(risk, nested, "source-soft-hash")
            evaluation(fixed, choice=False, binding=nested)
            fixed_settings = {"fixed": {"config": {"policy": {"max_groups": 16},
                "cost": cost}, "source_binding": nested,
                "system_hash": "fixed-hash"}}
            reseal_evaluation(fixed, fixed_settings)
            result = summarize_objectives(soft, risk, fixed, draws=100, seed=60)
            self.assertTrue(result["same_nested_system_and_soft_fit_verified"])
            self.assertEqual(result["metrics"]["risk_attention"]["mean_queried_groups"], 3)
            self.assertEqual(result["metrics"]["fixed_K16"]["mean_queried_groups"], 16)
            self.assertEqual(result["comparisons"]["risk_scalar_minus_soft_scalar"]
                             ["mean_queries_delta_a_minus_b"], 2)
            risk_settings = read_json(risk / "settings.json")
            risk_settings["structured_choice_risk_attention"]["max_groups"] = 8
            reseal_evaluation(risk, risk_settings)
            with self.assertRaisesRegex(ValueError, "budget/source/protocol"):
                summarize_objectives(soft, risk, fixed, draws=100, seed=60)
            risk_settings["structured_choice_risk_attention"]["max_groups"] = 16
            reseal_evaluation(risk, risk_settings)
            wrong = read_json(risk / "metrics.json")
            wrong["source_binding"]["nested_system"] = {"different": "nested"}
            (risk / "metrics.json").unlink()
            write_json(risk / "metrics.json", wrong)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                summarize_objectives(soft, risk, fixed, draws=100, seed=60)


if __name__ == "__main__":
    unittest.main()

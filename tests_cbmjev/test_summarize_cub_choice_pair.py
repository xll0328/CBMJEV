from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cbmjev.contracts import stable_hash
from cbmjev.io import file_hash, write_json, write_jsonl
from scripts.summarize_cub_choice_pair import summarize


def trace(index, *, policy_id, method, queries):
    return {"sample_id": str(index), "group_id": "group:" + str(index),
        "split": "validation", "mode": "offline_replay", "method": method,
        "policy_id": policy_id, "y": 0, "prediction": 0,
        "probabilities": [.8] + [.2 / 199] * 199,
        "queried_groups": list(range(queries)), "queried_atoms": list(range(queries)),
        "calls": queries, "declared_cost": float(queries), "steps": []}


def evaluation(directory, *, choice, binding):
    directory.mkdir()
    rows = []
    if choice:
        rows.extend(trace(i, policy_id="structured_choice_scalar",
                          method="structured_choice", queries=0) for i in range(594))
        rows.extend(trace(i, policy_id="structured_choice_attention",
                          method="structured_choice", queries=1) for i in range(594))
        source = {"nested_system": binding}
        policies = {"structured_choice_scalar": {}, "structured_choice_attention": {}}
        kind = "cbmjev-choice-crossfit-validation-v1"
    else:
        rows.extend(trace(i, policy_id="fixed", method="fixed", queries=16)
                    for i in range(594))
        source = binding
        policies = {"fixed": {"mean_queried_groups": 16}}
        kind = "cbmjev-crossfit-validation-v1"
    write_json(directory / "metrics.json", {"split": "validation", "mode": "offline_replay",
        "seed": 60, "source_binding": source, "policies": policies})
    write_json(directory / "settings.json", {})
    write_jsonl(directory / "traces.jsonl", rows)
    receipt = {"format": kind, "status": "COMPLETE", "paper_evidence": False,
        "source_binding": source, "files_sha256": {name: file_hash(directory / name)
            for name in ("metrics.json", "settings.json", "traces.jsonl")}}
    receipt["receipt_hash"] = stable_hash(receipt)
    write_json(directory / "receipt.json", receipt)


class ChoiceSummaryTests(unittest.TestCase):
    def test_source_matched_choice_and_exact_fixed_comparison(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            choice, fixed = root / "choice", root / "fixed"
            evaluation(choice, choice=True, binding={"same": "source"})
            evaluation(fixed, choice=False, binding={"same": "source"})
            result = summarize(choice, fixed, draws=100, seed=60)
            self.assertTrue(result["same_nested_system_verified"])
            self.assertEqual(result["metrics"]["fixed_K16"]["mean_queried_groups"], 16)
            self.assertEqual(result["metrics"]["attention"]["mean_queried_groups"], 1)
            self.assertEqual(result["metrics"]["scalar"]["zero_query_fraction"], 1)
            self.assertEqual(result["comparisons"]["attention_minus_scalar"]["accuracy_delta_a_minus_b"], 0)

    def test_rejects_different_nested_source(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            choice, fixed = root / "choice", root / "fixed"
            evaluation(choice, choice=True, binding={"same": "source"})
            evaluation(fixed, choice=False, binding={"other": "source"})
            with self.assertRaisesRegex(ValueError, "identical nested system"):
                summarize(choice, fixed, draws=100, seed=60)


if __name__ == "__main__":
    unittest.main()

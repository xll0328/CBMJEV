import json
from pathlib import Path
import tempfile
import unittest

from cbmjev.contracts import Concept, QueryGroup, Schema
from scripts.evaluate_cub_matched_gold_heads import (
    complete_train_pairs, evaluate_pair, fit_pair)


class MatchedGoldHeadsTests(unittest.TestCase):
    def test_train_pairs_require_same_membership_and_complete_gold(self):
        schema = Schema("tiny", 2,
            (Concept("a", "a", ("no", "yes")),
             Concept("b", "b", ("no", "yes"))),
            (QueryGroup("a", (0,)), QueryGroup("b", (1,))))
        automatic = [
            {"sample_id": "train-a", "group_id": "ga", "y": 1, "z": [0, 0]},
            {"sample_id": "train-b", "group_id": "gb", "y": 0, "z": [0, 0]}]
        with tempfile.TemporaryDirectory() as temporary:
            prepared = Path(temporary)
            rows = []
            for sid, gid, y, values in (
                    ("train-a", "ga", 1, (1, 0)),
                    ("train-b", "gb", 0, (0, None))):
                rows.append({"sample_id": sid, "group_id": gid, "split": "train",
                    "target": {"status": "OBSERVED", "value": y},
                    "concepts": [{"concept_id": c, "annotation_status":
                                  "OBSERVED" if v is not None else "NOT_VISIBLE",
                                  "value": v} for c, v in zip(("a", "b"), values)]})
            rows.append({"sample_id": "val", "split": "validation"})
            (prepared / "samples.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            pairs = complete_train_pairs(prepared, schema, automatic)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]["sample_id"], "train-a")
            self.assertEqual(pairs[0]["automatic"], (0, 0))
            self.assertEqual(pairs[0]["gold"], (1, 0))
            automatic[0]["y"] = 0
            with self.assertRaises(ValueError):
                complete_train_pairs(prepared, schema, automatic)

    def test_same_inputs_keep_heads_identical(self):
        schema = Schema("tiny", 2,
            (Concept("a", "a", ("no", "yes")),
             Concept("b", "b", ("no", "yes"))),
            (QueryGroup("a", (0,)), QueryGroup("b", (1,))))
        pairs = [{"sample_id": f"s{i}", "group_id": f"g{i}", "y": i % 2,
                  "automatic": (i % 2, (i // 2) % 2),
                  "gold": (i % 2, (i // 2) % 2)} for i in range(8)]
        heads, config, losses = fit_pair(pairs, schema,
            {"hidden": 8, "batch_size": 4, "learning_rate": .001},
            seed=7, epochs=2, cpu_threads=1)
        self.assertEqual(config["head_epochs"], 2)
        self.assertEqual(losses["automatic"], losses["gold"])
        for pair in pairs:
            self.assertEqual(heads["automatic"].probabilities(pair["automatic"]),
                             heads["gold"].probabilities(pair["gold"]))
        validation = [{"sample_id": "v", "y": 1, "z": [1, 0]}]
        result = evaluate_pair(validation, {"v": (1, 0)}, schema, heads,
                               [0, 1], [1, 2], 2, 7)
        self.assertEqual(result["1"]["ce"]["gold_minus_automatic"], 0)
        self.assertEqual(result["2"]["correct"]["gold_minus_automatic"], 0)


if __name__ == "__main__":
    unittest.main()

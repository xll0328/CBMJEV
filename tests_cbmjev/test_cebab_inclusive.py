import json
from pathlib import Path
import tempfile
import unittest

from cbmjev.data import prepare_dataset
from tests_cbmjev.test_data import cebab_fixture, read_jsonl, review, write


class CEBaBInclusiveTests(unittest.TestCase):
    def test_explicit_replacement_transitive_text_filter_and_frozen_heldout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tables = cebab_fixture(root / "raw")
            old = prepare_dataset("cebab", root / "raw", root / "old")
            train = list(tables["train_exclusive"])
            # Family0 reaches heldout after NFKC/casefold/whitespace normalization.
            train.append(review(100, family="0", original=False,
                                text="  Ａ DISTINCT review NUMBER 20.  "))
            # Family1 reaches family0 by a different text; remove entire closure.
            train.append(review(101, family="0", original=False, text="A bridge"))
            train.append(review(102, family="1", original=False, text="a   BRIDGE"))
            train.append(review(103, family="2", original=False, text="A new safe edit"))
            write(root / "raw/train_inclusive.json", json.dumps(train))
            new = prepare_dataset("cebab", root / "raw", root / "new",
                                  train_variant="train_inclusive", heldout_reference=root / "old")
            rows = read_jsonl(new["samples"])
            kept_train = [s for s in rows if s["split"] == "train"]
            self.assertEqual(len(kept_train), 9)
            self.assertTrue(all(s["provenance"]["source_split"] == "train_inclusive" for s in kept_train))
            self.assertNotIn("cebab:0", {s["sample_id"] for s in kept_train})
            self.assertNotIn("cebab:1", {s["sample_id"] for s in kept_train})
            old_membership = [m for m in read_jsonl(old["membership"]) if m["outer_split"] != "train"]
            new_membership = [m for m in read_jsonl(new["membership"]) if m["outer_split"] != "train"]
            self.assertEqual(old_membership, new_membership)
            self.assertEqual(len(read_jsonl(new["exclusions"])), 5)
            # Coexisting inclusive file must never be silently appended by default.
            default = prepare_dataset("cebab", root / "raw", root / "default")
            self.assertEqual(default["report"]["observed_counts"]["train"], 10)

    def test_frozen_label_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tables = cebab_fixture(root / "raw")
            prepare_dataset("cebab", root / "raw", root / "old")
            write(root / "raw/train_inclusive.json", json.dumps(tables["train_exclusive"]))
            tables["test"][0]["review_majority"] = "5"
            write(root / "raw/test.json", json.dumps(tables["test"]))
            with self.assertRaisesRegex(ValueError, "frozen heldout contents/labels changed"):
                prepare_dataset("cebab", root / "raw", root / "new",
                                train_variant="train_inclusive", heldout_reference=root / "old")

    def test_safe_text_components_never_cross_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tables = cebab_fixture(root / "raw")
            train = list(tables["train_exclusive"])
            train.append(review(100, family="0", original=False, text="train shared"))
            train.append(review(101, family="1", original=False, text="TRAIN  SHARED"))
            write(root / "raw/train_inclusive.json", json.dumps(train))
            result = prepare_dataset("cebab", root / "raw", root / "new", train_variant="train_inclusive")
            rows = [m for m in read_jsonl(result["membership"]) if m["sample_id"] in ("cebab:0", "cebab:1", "cebab:100", "cebab:101")]
            self.assertEqual(len({r["group_id"] for r in rows}), 1)
            self.assertEqual(len({r["split"] for r in rows}), 1)


if __name__ == "__main__":
    unittest.main()

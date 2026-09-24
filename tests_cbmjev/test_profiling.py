import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cbmjev.contracts import Concept, QueryGroup, Schema
from cbmjev.profiling import profile_inputs, profile_backend


class CountingResponder:
    def __init__(self, dependent=False, shared=False):
        self.dependent, self.shared = dependent, shared
        self.sessions, self.payloads = 0, []

    def start_session(self, payload):
        self.sessions += 1
        self.payloads.append(payload)
        parent = self

        class Session:
            calls = 0
            last_stats = {}

            def respond(self, atom_ids):
                self.last_stats = {"encoder_forwards": int(not parent.shared or self.calls == 0),
                    "atoms_computed": 3 if parent.shared and self.calls == 0 else len(atom_ids),
                    "cache_hit": parent.shared and self.calls > 0,
                    "cost_mode": "SHARED_CHEAP_ALL_HEADS" if parent.shared else "FAKE_TYPED"}
                self.calls += 1
                return tuple(int(len(atom_ids) > 1) if parent.dependent else j % 2 for j in atom_ids)

        return Session()


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.schema = Schema("profile_test", 2,
            tuple(Concept(str(j), f"concept {j}", ("no", "yes")) for j in range(3)),
            (QueryGroup("g0", (0, 1)), QueryGroup("g1", (2,))))
        self.rows = [{"sample_id": f"s{i}", "group_id": f"g{i}", "split": "validation",
                      "input": {"modality": "text", "text": "content only"}, "y": 999,
                      "concepts": "must not reach responder"} for i in range(2)]

    def test_live_group_calls_new_sessions_and_streamed_statistics(self):
        responder = CountingResponder(shared=True)
        with tempfile.TemporaryDirectory() as folder:
            summary = profile_inputs(self.schema, iter(self.rows), responder, folder,
                repeats=2, warmup=1, batch_sizes=(1,), model_hash="frozen-model")
            records = [json.loads(line) for line in (Path(folder) / "timings.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 8)
            self.assertEqual(responder.sessions, 10)
            self.assertTrue(summary["batch_independent"])
            self.assertFalse(summary["flat_cache_certified"])
            self.assertEqual(summary["full_vs_split_comparisons"], 4)
            self.assertEqual(summary["configurations"]["batch_1"]["actions"], [[0], [1]])
            self.assertTrue(all(set(vars(p)) == {"text", "images"} for p in responder.payloads))
            for row in records:
                self.assertEqual(row["calls"], 1 if row["configuration"] == "all" else 2)
                self.assertGreaterEqual(row["total_wall_ms"], row["response_ms"])
                self.assertIsNotNone(row["shared_encoder_first_call_ms"])
                self.assertEqual(row["model_hash"], "frozen-model")
                self.assertFalse(row["cold_model"])

    def test_batch_dependent_output_blocks_independence_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = profile_inputs(self.schema, self.rows[:1], CountingResponder(dependent=True),
                                     folder, repeats=1, warmup=0, batch_sizes=(1,))
            self.assertFalse(summary["batch_independent"])
            self.assertEqual(summary["full_vs_split_mismatches"], 1)

    def test_no_test_or_fit_data_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            bad = {**self.rows[0], "split": "test"}
            responder = CountingResponder()
            with self.assertRaisesRegex(ValueError, "validation"):
                profile_inputs(self.schema, [bad], responder, folder, warmup=0)
            self.assertEqual(responder.sessions, 0)
            profile_inputs(self.schema, self.rows, responder, folder, repeats=1, warmup=0, batch_sizes=())
            with self.assertRaises(ValueError):
                profile_inputs(self.schema, self.rows, responder, folder, repeats=1, warmup=0)

    def test_limit_does_not_load_remaining_inputs(self):
        def rows():
            yield self.rows[0]
            raise AssertionError("read beyond limit")
        with tempfile.TemporaryDirectory() as folder:
            summary = profile_inputs(self.schema, rows(), CountingResponder(), folder,
                                     limit=1, repeats=1, warmup=0, batch_sizes=())
            self.assertEqual(summary["n_cases"], 1)

    def test_image_preprocessing_and_content_hash_ignore_filename(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow unavailable")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = Image.new("RGB", (8, 8), (50, 100, 150))
            image.save(root / "one.png")
            image.save(root / "two.png")
            rows = [{**row, "input": {"modality": "image", "image_paths": [name]}}
                    for row, name in zip(self.rows, ("one.png", "two.png"))]
            responder = CountingResponder()
            profile_inputs(self.schema, rows, responder, root / "profile", raw_root=root,
                           repeats=1, warmup=0, batch_sizes=())
            records = [json.loads(line) for line in (root / "profile" / "timings.jsonl").read_text().splitlines()]
            self.assertEqual(records[0]["input_digest"], records[1]["input_digest"])
            self.assertTrue(all(r["preprocess_ms"] > 0 for r in records))
            self.assertTrue(all(p.text is None and isinstance(p.images[0], bytes) for p in responder.payloads))

    def test_profile_backend_selects_validation_and_guards_membership(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "samples.jsonl").write_text("fixture")
            (root / "membership.jsonl").write_text("fixture")
            (root / "receipt.json").write_text("fixture")
            from cbmjev.io import file_hash
            receipt = {"membership_sha256": file_hash(root / "membership.jsonl"), "supervised_group_ids": [],
                       "checkpoint_sha256": "model", "kind": "injected", "artifact_id": "R:fixture"}
            memberships = {"s0": {"split": "validation"}, "s1": {"split": "test"}}
            with patch("cbmjev.pipeline.load_prepared", return_value=(self.schema, self.rows, memberships)), \
                 patch("cbmjev.pipeline.load_backend", return_value=(CountingResponder(), receipt)):
                result = profile_backend(root, root, root / "out", repeats=1, warmup=0, batch_sizes=())
                self.assertEqual(result["n_cases"], 1)
                receipt["membership_sha256"] = "changed"
                with self.assertRaisesRegex(ValueError, "manifest"):
                    profile_backend(root, root, root / "bad", repeats=1, warmup=0)

    def test_rejects_invalid_profiling_parameters(self):
        for args in ({"limit": 0}, {"repeats": 0}, {"warmup": -1}, {"batch_sizes": (0,)},
                     {"batch_sizes": (1, 1)}):
            with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
                profile_inputs(self.schema, self.rows, CountingResponder(), folder, **args)

    def test_all_only_has_no_batch_independence_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            result = profile_inputs(self.schema, self.rows[:1], CountingResponder(), folder,
                                    repeats=1, warmup=0, batch_sizes=())
            self.assertIsNone(result["batch_independent"])

    def test_stochastic_full_responses_fail_repeat_stability(self):
        class ChangingResponder:
            calls = 0

            def respond(self, payload, atom_ids):
                self.calls += 1
                return (self.calls % 2,) * len(atom_ids)

        with tempfile.TemporaryDirectory() as folder:
            result = profile_inputs(self.schema, self.rows[:1], ChangingResponder(), folder,
                                    repeats=2, warmup=0, batch_sizes=())
            self.assertFalse(result["batch_independent"])
            self.assertEqual(result["full_repeat_mismatches"], 1)


if __name__ == "__main__":
    unittest.main()

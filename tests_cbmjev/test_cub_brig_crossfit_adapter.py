"""Focused checks for the CUB BRiG crossfit adapter's data boundary."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from cbmjev.brig import BRiGPolicy, GroupQ
from cbmjev.contracts import Concept, QueryGroup, Schema
from cbmjev.io import read_json
from cbmjev.provenance import make_fit_record
from scripts import train_evaluate_cub_brig_crossfit as adapter


def schema():
    return Schema("tiny", 2,
        tuple(Concept(str(i), "feature", ("0", "1")) for i in range(3)),
        tuple(QueryGroup(str(i), (i,)) for i in range(3)))


class FixedPolicy:
    def __init__(self, schema, scores):
        self.schema, self.scores = schema, scores

    def predict_budget(self, observed, remaining_budget):
        return tuple((g, self.scores[g]) for g, seen in
                     enumerate(self.schema.group_mask(observed)) if not seen)


class AdapterTests(TestCase):
    def test_mean_q_uses_current_visible_state_and_budget(self):
        s = schema()
        policy = adapter.MeanQPolicy([FixedPolicy(s, (0, 3, 1)),
                                      FixedPolicy(s, (4, 1, 1))])
        self.assertEqual(policy.choose(s.empty_state(), 1), (2,))
        self.assertEqual(policy.choose((0, -1, -1), 1), (2,))
        self.assertEqual(policy.choose((0, -1, -1), 0), ())

    def test_fit_gets_only_verified_outer_rows_and_its_excluded_head(self):
        s = schema()
        rows = [dict(sample_id="outer-a", group_id="outer-g", split="train",
                     z=[0, 1, 0], y=1)]
        head = object()
        binding = {"training_group_ids": ["outer-g"], "training_sample_count": 1,
                   "outer_head_artifact_id": "head:fold", "outer_responder_artifact_id": "R:fold"}
        head_report = {"provenance": [make_fit_record(
            "head:fold", supervised_group_ids=["inner-g"])]}
        manifest = {"provenance": [make_fit_record(
            "R:fold", supervised_group_ids=["responder-g"])]}
        model = GroupQ(s, 4)
        policy = BRiGPolicy(s, {1: model})
        report = {"sample_count": 1, "schema_hash": s.hash, "config": {"max_budget": 1}}
        with TemporaryDirectory() as temp:
            args = SimpleNamespace(out=str(Path(temp) / "fit"), outer_fold=0,
                max_budget=1, seed=60, device="cpu", hidden=4, epochs=8,
                batch_size=1, learning_rate=.001, no_empty_rollout=True)
            with patch.object(adapter, "_fold_sources", return_value=(
                    s, rows, head, binding, head_report, manifest)) as sources, \
                 patch.object(adapter, "fit_brig", return_value=(policy, report)) as fit, \
                 patch.object(adapter, "code_fingerprint", return_value="fixed-source"):
                receipt = adapter.train_fold(args)
            self.assertEqual(sources.call_count, 2)
            self.assertIs(fit.call_args.args[1], head)
            self.assertEqual(fit.call_args.kwargs["excluded_head_group_ids"], ["outer-g"])
            self.assertEqual(fit.call_args.args[0], [{**rows[0], "split": "policy_fit"}])
            self.assertEqual(receipt["status"], "COMPLETE")
            self.assertEqual(read_json(Path(args.out) / "training.json")["binding"], binding)

    def test_fold_ensemble_requires_complete_coverage(self):
        args = adapter.parse_args(["evaluate", "--prepared", "p", "--planned", "q",
            "--merged", "m", "--responder", "r", "--cache", "c", "--out", "o",
            "--fold-models", "f0", "f1"])
        self.assertEqual(args.budget, 16)
        self.assertEqual(args.fold_models, ["f0", "f1"])


if __name__ == "__main__":
    import unittest
    unittest.main()

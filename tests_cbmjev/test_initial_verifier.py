from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from cbmjev.contracts import Concept, QueryGroup, Schema
from cbmjev.initial_verifier import (InitialConceptEncoder, fit_initial_source,
    load_initial_checkpoint, masked_concept_loss, predict_initial_records,
    predict_initial_source, train_initial_source)
from cbmjev.io import file_hash, read_json, read_jsonl, write_json, write_jsonl
from cbmjev.typed_verifier import SourceRecord, concept_metrics, load_source_records, record_metadata, source_group_split
from scripts.train_initial_verification_source import build_parser, main


class FakeTokenizer:
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"

    def __call__(self, texts, *, padding, truncation, max_length=None,
                 return_attention_mask=True, return_tensors=None):
        rows = []
        for text in texts:
            content = [3 + sum(map(ord, word)) % 29 for word in text.split()]
            if truncation:
                content = content[:max_length - 2]
            rows.append([1] + content + [2])
        if return_tensors is None:
            return {"input_ids": rows}
        width = max(map(len, rows))
        ids = torch.tensor([row + [0] * (width - len(row)) for row in rows])
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=6, is_decoder=False, is_encoder_decoder=False)
        self.embedding = nn.Embedding(32, 6)
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        embedded = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        context = (embedded * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        return SimpleNamespace(last_hidden_state=embedded + context)


def native_schema(width=21):
    return Schema("synthetic_appraisal", 13,
                  tuple(Concept("c" + str(j), "Concept " + str(j), ("1", "2", "3", "4", "5")) for j in range(width)),
                  tuple(QueryGroup("c" + str(j), (j,)) for j in range(width)))


def source_records(schema):
    return [SourceRecord("s" + str(i), "g" + str(i // 2), "responder_fit", "event number " + str(i),
                         tuple((i + j) % 5 if (i + j) % 7 else None for j in range(schema.num_atoms)))
            for i in range(20)]


def write_prepared(path, schema, target_offset=0):
    path.mkdir()
    records = source_records(schema)
    records += [SourceRecord("head", "headgroup", "head_fit", "private head event", (4,) * schema.num_atoms),
                SourceRecord("confirm", "confirmgroup", "confirmation", "private confirmation event", (2,) * schema.num_atoms)]
    samples, members = [], []
    for i, record in enumerate(records):
        outer = "train" if record.role in ("responder_fit", "head_fit") else record.role
        samples.append({"dataset": schema.dataset, "sample_id": record.sample_id, "group_id": record.group_id,
            "split": outer, "input": {"modality": "text", "text": record.text},
            "target": {"value": (i + target_offset) % 13, "status": "OBSERVED"},
            "concepts": [{"concept_id": concept.id, "value": value,
                          "annotation_status": "OBSERVED" if value is not None else "MISSING_ANNOTATION"}
                         for concept, value in zip(schema.concepts, record.concepts)]})
        members.append({"sample_id": record.sample_id, "group_id": record.group_id,
                        "split": record.role, "outer_split": outer})
    write_json(path / "schema.json", schema.to_dict())
    write_jsonl(path / "samples.jsonl", samples)
    write_jsonl(path / "membership.jsonl", members)


class InitialVerifierTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.set_num_threads(1)
        self.schema = native_schema()

    def factory(self, schema=None, max_length=512):
        return InitialConceptEncoder(schema or self.schema, FakeBackbone(), FakeTokenizer(),
                                     binding={"model_revision": "SYNTHETIC_TEST_ONLY"}, max_length=max_length)

    def test_native_shape_shared_encoding_probabilities_and_truncation(self):
        model = self.factory(max_length=5)
        records = [SourceRecord("short", "one", "head_fit", "one", (None,) * 21),
                   SourceRecord("long", "two", "head_fit", "one two three four five six", (None,) * 21)]
        hard, probabilities = predict_initial_records(model, records, batch_size=2)
        self.assertEqual(model.backbone.calls, 1)
        self.assertEqual(tuple(hard.shape), (2, 21))
        self.assertEqual(len(probabilities), 21)
        for probability in probabilities:
            self.assertEqual(tuple(probability.shape), (2, 5))
            self.assertTrue(torch.allclose(probability.sum(1), torch.ones(2)))
            self.assertEqual(probability.dtype, torch.float32)
        report = model.truncation_report([r.text for r in records])
        self.assertEqual(report["truncated_samples"], 1)
        self.assertEqual(report["truncation_rate"], .5)
        self.assertEqual(report["removed_tokens"], 3)
        separate, _ = predict_initial_records(model, records, batch_size=1)
        self.assertTrue(torch.equal(hard, separate))

    def test_masked_ce_ignores_missing_and_has_no_unknown_class(self):
        logits = (torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True),
                  torch.tensor([[2., 1.], [1., 2.]], requires_grad=True))
        labels = torch.tensor([[1, -1], [-1, 0]])
        loss = masked_concept_loss(logits, labels)
        expected = (F.cross_entropy(logits[0][:1], torch.tensor([1])) +
                    F.cross_entropy(logits[1][1:], torch.tensor([0]))) / 2
        self.assertEqual(float(loss.detach()), float(expected.detach()))
        loss.backward()
        self.assertTrue(torch.equal(logits[0].grad[1], torch.zeros(3)))
        self.assertTrue(torch.equal(logits[1].grad[0], torch.zeros(2)))
        with self.assertRaises(ValueError):
            masked_concept_loss(logits, torch.tensor([[3, -1], [-1, 0]]))
        self.assertEqual(float(masked_concept_loss(logits, torch.full((2, 2), -1)).detach()), 0.)

    def test_fixed_category_macro_f1_and_shared_group_split(self):
        schema = native_schema(1)
        metrics = concept_metrics(torch.zeros((3, 1), dtype=torch.long), torch.zeros((3, 1), dtype=torch.long), schema)
        self.assertAlmostEqual(metrics["concept_macro_f1"], .2)
        records = source_records(self.schema)
        _, _, split = source_group_split([record_metadata(r) for r in records])
        self.assertFalse(set(split["fit_group_ids"]) & set(split["tune_group_ids"]))
        self.assertEqual(len(split["tune_group_ids"]), 2)
        _, _, reordered = source_group_split([record_metadata(r) for r in reversed(records)])
        self.assertEqual(split["tune_group_ids"], reordered["tune_group_ids"])

    def test_source_role_guard_before_any_model_or_output(self):
        record = SourceRecord("bad", "group", "head_fit", "event", (1,) * 21)
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            with self.assertRaisesRegex(ValueError, "responder_fit"):
                fit_initial_source([record], self.schema, output, model_factory=lambda: self.fail("model initialized"))
            self.assertFalse(output.exists())

    def test_train_ignores_y_refits_source_only_and_serializes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, offset in (("first", 0), ("second", 8)):
                write_prepared(root / name, self.schema, target_offset=offset)
            loader = lambda path, schema, **kwargs: self.factory(schema)
            before = torch.get_rng_state().clone()
            with patch("cbmjev.initial_verifier.load_initial_encoder", side_effect=loader):
                first, report = train_initial_source(root / "first", root / "fit1", "unused",
                    model_revision="synthetic", device="cpu", epochs=1, batch_size=8)
                second, _ = train_initial_source(root / "second", root / "fit2", "unused",
                    model_revision="synthetic", device="cpu", epochs=1, batch_size=8)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(first["fit_sample_ids"], [r.sample_id for r in sorted(source_records(self.schema), key=lambda r: r.sample_id)])
            self.assertNotIn("head", first["fit_sample_ids"])
            self.assertEqual(len(report["trials"]), 2)
            self.assertEqual([r["learning_rate"] for r in report["trials"]], [2e-5, 5e-5])
            self.assertEqual(report["full_source_refit_n"], 20)
            self.assertEqual(first["selected_recipe"], second["selected_recipe"])
            _, model1 = load_initial_checkpoint(root / "fit1", model_factory=self.factory)
            _, model2 = load_initial_checkpoint(root / "fit2", model_factory=self.factory)
            for key, tensor in model1.state_dict().items():
                self.assertTrue(torch.equal(tensor, model2.state_dict()[key]), key)
            manifest = read_jsonl(root / "fit1" / "source_manifest.jsonl")
            self.assertFalse(any("text" in row or "target" in row for row in manifest))
            self.assertEqual(first["source_split"], source_group_split(manifest)[2])
            with patch("cbmjev.initial_verifier.load_initial_checkpoint", return_value=(first, model1)):
                prediction = root / "predictions.jsonl"
                summary = predict_initial_source(root / "first", root / "fit1", prediction,
                    roles=("head_fit",), device="cpu")
            rows = read_jsonl(prediction)
            self.assertEqual(summary["n"], 1)
            self.assertEqual(rows[0]["sample_id"], "head")
            self.assertEqual(set(rows[0]), {"sample_id", "group_id", "split", "A", "A_probabilities"})
            self.assertEqual(len(rows[0]["A"]), 21)
            self.assertNotIn("private head event", prediction.read_text())
            self.assertEqual(read_json(prediction.with_suffix(".jsonl.metadata.json"))["roles"], ["head_fit"])
            self.assertEqual(summary["fit_sample_ids"], first["fit_sample_ids"])
            self.assertEqual(summary["fit_group_ids"], first["fit_group_ids"])
            self.assertEqual(summary["source_training_provenance"], first["provenance"])
            self.assertEqual(summary["source_training_provenance"]["roles"], ["responder_fit"])
            self.assertEqual(summary["provenance"]["roles"], ["head_fit"])
            self.assertEqual(summary["source_training_membership_sha256"],
                             first["provenance"]["membership_sha256"])
            self.assertEqual(summary["checkpoint_metadata_sha256"], file_hash(root / "fit1" / "checkpoint.json"))
            self.assertEqual(summary["model_sha256"], file_hash(root / "fit1" / "model.pt"))
            self.assertEqual(summary["predictions_sha256"], file_hash(prediction))

    def test_membership_join_protected_roles_and_observed_only(self):
        with TemporaryDirectory() as tmp:
            prepared = Path(tmp) / "prepared"
            write_prepared(prepared, self.schema)
            _, source, _ = load_source_records(prepared)
            self.assertEqual(len(source), 20)  # outer split is train, not responder_fit
            _, heads, _ = load_source_records(prepared, ("head_fit",))
            self.assertEqual(heads[0].concepts, (None,) * 21)
            with self.assertRaisesRegex(ValueError, "protected"):
                predict_initial_source(prepared, Path(tmp) / "absent", Path(tmp) / "out",
                                       roles=("confirmation",), device="cpu")
            samples = read_jsonl(prepared / "samples.jsonl")
            samples[0]["concepts"][1]["annotation_status"] = None
            # Test deliberately corrupts a synthetic fixture, not a dataset artifact.
            with (prepared / "samples.jsonl").open("w") as stream:
                for row in samples:
                    stream.write(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "missing annotation"):
                load_source_records(prepared)

    def test_cli_defaults_and_seed_contract(self):
        args = build_parser().parse_args(["train", "--prepared", "p", "--output", "o", "--backbone", "b"])
        self.assertEqual((args.seed, args.epochs, args.batch_size, args.split_seed), (40, 5, 16, 20260930))
        args = build_parser().parse_args(["predict", "--prepared", "p", "--checkpoint", "c", "--output", "o"])
        self.assertFalse(args.allow_protected_roles)
        self.assertNotIn("confirmation", args.roles)

    def test_cli_summary_excludes_private_ids_and_full_metrics(self):
        truncation = {"n_samples": 2, "max_length": 512, "truncated_samples": 1, "truncation_rate": .5}
        metrics = {"concept_macro_f1": .2, "n_samples": 2, "concepts": [{"confusion": [[1, 0], [0, 1]]}]}
        train_report = {"selected": {"learning_rate": 5e-5, "epoch": 5, "metrics": metrics}, "truncation": truncation}
        checkpoint = {"fit_sample_ids": ["private-source-id"], "checkpoint_identity": "checkpoint-hash"}
        with patch("scripts.train_initial_verification_source.train_initial_source", return_value=(checkpoint, train_report)):
            output = StringIO()
            with redirect_stdout(output):
                main(["train", "--prepared", "p", "--output", "o", "--backbone", "b"])
        summary = json.loads(output.getvalue())
        self.assertEqual(summary["selected"], {"learning_rate": 5e-5, "epochs": 5, "concept_macro_f1": .2, "n_samples": 2})
        self.assertNotIn("private-source-id", output.getvalue())
        self.assertNotIn("confusion", output.getvalue())
        prediction_report = {"n": 2, "seed": 40, "predictions_sha256": "prediction-hash", "truncation": truncation,
                             "fit_sample_ids": ["private-source-id"], "schema": {"private": "schema"}}
        with patch("scripts.train_initial_verification_source.predict_initial_source", return_value=prediction_report):
            output = StringIO()
            with redirect_stdout(output):
                main(["predict", "--prepared", "p", "--checkpoint", "c", "--output", "o"])
        summary = json.loads(output.getvalue())
        self.assertEqual((summary["n"], summary["seed"]), (2, 40))
        self.assertNotIn("private-source-id", output.getvalue())
        self.assertNotIn("schema", output.getvalue())


if __name__ == "__main__":
    unittest.main()

"""Synthetic engineering checks only: no downloaded model/data or quality claim."""
import copy
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from cbmjev.contracts import Concept, QueryGroup, Schema, stable_hash
from cbmjev.io import file_hash, read_json, read_jsonl, write_json, write_jsonl
from cbmjev.typed_verifier import (
    ConceptReadout, FrozenTypedEncoder, SourceRecord, audit_source_conditions,
    concept_metrics, evaluate_source_gate, extract_feature_cache, load_feature_cache,
    load_readouts, load_source_records, predict_cached_sources, record_metadata,
    predict_text_sources, source_gate_definition, source_group_split, train_cached_sources)
from scripts.train_verification_sources import build_parser, main


def schema(counts=(3, 5)):
    concepts = tuple(Concept("c" + str(j), "A native question about concept " + str(j),
                            ("negative", "positive", "not mentioned") if count == 3 else
                            tuple(str(i + 1) for i in range(count))) for j, count in enumerate(counts))
    return Schema("synthetic_engineering", 3, concepts,
                  tuple(QueryGroup(c.id, (j,)) for j, c in enumerate(concepts)))


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 3

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("segmented prompts must not duplicate special tokens")
        return [int(value) + 2 for value in text.encode("utf-8")]

    def decode(self, tokens):
        return bytes(token - 2 for token in tokens).decode("utf-8")

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        if tokenize or add_generation_prompt or enable_thinking:
            raise AssertionError("complete nonthinking paths required")
        return "".join("<" + message["role"] + ">\n" + message["content"] + "\x01\n"
                       for message in messages)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, use_cache=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            self.embedding = nn.Embedding(258, 8)
        self.calls = 0
        self.seen = []

    def forward(self, input_ids, attention_mask, position_ids, return_dict, use_cache):
        self.calls += 1
        assert return_dict and not use_cache
        expected = (attention_mask.cumsum(-1) - 1).clamp_min(0)
        assert torch.equal(position_ids, expected), "padding-aware positions required"
        self.seen.extend(input_ids[i][attention_mask[i].bool()].tolist() for i in range(len(input_ids)))
        embeddings = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        hidden = embeddings.cumsum(1) + position_ids.unsqueeze(-1) * .001
        return SimpleNamespace(last_hidden_state=hidden)


def encoder(the_schema=None, max_length=768):
    return FrozenTypedEncoder(Backbone(), Tokenizer(), the_schema or schema(),
                              binding={"model_id": "synthetic", "model_revision": "a" * 40,
                                       "attention_implementation": "synthetic"}, max_length=max_length)


def records(n=20):
    return [SourceRecord("sample" + str(i), "group" + str(i // 2), "responder_fit",
                         "Distinct review " + str(i) + " with variable words " * (i % 3 + 1),
                         (i % 3, None if i == 0 else i % 5)) for i in range(n)]


def prepare(folder, the_records, the_schema=None):
    the_schema = the_schema or schema()
    folder.mkdir(parents=True)
    write_json(folder / "schema.json", the_schema.to_dict())
    members, samples = [], []
    for record in the_records:
        outer = "train" if record.role in ("responder_fit", "head_fit", "policy_fit") else record.role
        members.append({"sample_id": record.sample_id, "group_id": record.group_id,
                        "split": record.role, "outer_split": outer})
        samples.append({"sample_id": record.sample_id, "group_id": record.group_id,
                        "split": outer, "dataset": the_schema.dataset,
                        "input": {"modality": "text", "text": record.text},
                        "target": {"value": 999999}, "audit_metadata": {"secret": "unused"},
                        "concepts": [{"concept_id": concept.id, "value": value,
                                      "annotation_status": "OBSERVED" if value is not None else "MISSING_ANNOTATION"}
                                     for concept, value in zip(the_schema.concepts, record.concepts)]})
    write_jsonl(folder / "membership.jsonl", members)
    write_jsonl(folder / "samples.jsonl", samples)


class TypedVerifierTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)

    def test_three_and_five_class_order_padding_batch_full_path(self):
        model = encoder()
        texts = ["Short review.", "A much longer review with context about several concepts."]
        reference, shared_reference, stats = model.encode_texts(texts, batch_size=1)
        self.assertEqual(tuple(reference.shape), (2, 8, 8))
        self.assertEqual(stats["candidate_paths"], 16)
        self.assertEqual(stats["shared_text_paths"], 2)
        for side, reverse in (("left", False), ("right", False), ("left", True), ("right", True)):
            actual, shared, _ = model.encode_texts(texts, batch_size=7, padding_side=side,
                                                  reverse_candidates=reverse)
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            torch.testing.assert_close(shared, shared_reference, atol=0, rtol=0)
        self.assertFalse(torch.equal(reference[:, 0], reference[:, 1]))
        for i, value in enumerate(schema().concepts[0].values):
            decoded = model.tokenizer.decode(model.backbone.seen[i])
            self.assertTrue(decoded.endswith("<assistant>\n" + value + "\x01\n"))
            self.assertIn("A native question", decoded)
        self.assertEqual(model.backbone.calls, stats["encoder_forwards"] + 4 * 4)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))

    def test_native_schema_sizes_need_no_hardcoded_unknown_alias(self):
        for counts in ((3,) * 4, (5,) * 21):
            model = encoder(schema(counts), max_length=800)
            features, shared, _ = model.encode_texts(["An event."], batch_size=64)
            self.assertEqual(features.shape[1], sum(counts))
            readout = ConceptReadout(8, counts, "typed")
            self.assertEqual([x.shape[1] for x in readout(features)], list(counts))
            self.assertEqual(tuple(shared.shape), (1, 8))

    def test_truncation_reserves_complete_question_candidate_and_same_text(self):
        model = encoder(max_length=600)
        text = "BEGIN " + "middle " * 500 + " END"
        _, _, stats = model.encode_texts([text], batch_size=20)
        decoded = [model.tokenizer.decode(tokens) for tokens in model.backbone.seen]
        self.assertEqual(stats["truncated_texts"], 1)
        evidence = []
        for prompt, tokens in zip(decoded, model.backbone.seen):
            self.assertLessEqual(len(tokens), 600)
            self.assertIn("BEGIN ", prompt)
            self.assertIn(" END", prompt)
            self.assertIn("[TEXT TRUNCATED]", prompt)
            evidence.append(prompt.split("Quoted text:\n")[1].split("\n\n")[0])
        self.assertEqual(len(set(evidence)), 1)
        for i, value in enumerate(schema().concepts[0].values):
            self.assertTrue(decoded[i].endswith("<assistant>\n" + value + "\x01\n"))
        with self.assertRaisesRegex(ValueError, "evidence token budget"):
            encoder(max_length=10)

    def test_short_original_text_and_special_tokens_preserved(self):
        model = encoder()
        text = 'A quoted "review"\nwith spacing  intact.'
        _, _, stats = model.encode_texts([text], batch_size=9)
        self.assertEqual(stats["truncated_texts"], 0)
        for tokens in model.backbone.seen:
            decoded = model.tokenizer.decode(tokens)
            self.assertIn(text, decoded)
            self.assertEqual(decoded.count("<system>"), 1)

    def test_live_singleton_and_shared_only_do_not_compute_other_paths(self):
        model = encoder()
        heads = {kind: ConceptReadout(8, schema().value_counts, kind) for kind in ("typed", "shared")}
        text = "This is an event with evidence."
        full = predict_text_sources(model, heads, [text], batch_size=16)
        calls = model.backbone.calls
        selected = predict_text_sources(model, heads, [text], atom_ids=(1,), include_shared=False, batch_size=16)
        self.assertEqual(model.backbone.calls - calls, 1)
        self.assertEqual(selected["cost_record"]["candidate_paths"], 5)
        self.assertEqual(selected["cost_record"]["shared_text_paths"], 0)
        self.assertEqual(selected["typed_probabilities"][0][0], full["typed_probabilities"][0][1])
        self.assertNotIn("shared_values", selected)
        calls = model.backbone.calls
        shared = predict_text_sources(model, heads, [text], atom_ids=(), batch_size=16)
        self.assertEqual(model.backbone.calls - calls, 1)
        self.assertEqual(shared["cost_record"]["candidate_paths"], 0)
        self.assertEqual(shared["cost_record"]["shared_text_paths"], 1)
        self.assertEqual(shared["shared_probabilities"], full["shared_probabilities"])
        self.assertNotIn("typed_values", shared)
        with self.assertRaisesRegex(ValueError, "at least one"):
            model.encode_texts([text], atom_ids=(), include_shared=False)

    def test_prepared_roles_missingness_and_no_task_or_metadata_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prepared"
            raw = records(4) + [SourceRecord("protected", "protected_group", "confirmation", "Secret text", (2, 4)),
                                SourceRecord("head", "head_group", "head_fit", "Head text", (1, 2))]
            prepare(path, raw)
            actual_schema, selected, provenance = load_source_records(path)
            self.assertEqual(selected, sorted(raw[:4], key=lambda r: r.sample_id))
            self.assertEqual(actual_schema.hash, schema().hash)
            self.assertEqual(selected[0].concepts, (0, None))
            self.assertFalse(hasattr(selected[0], "target"))
            _, head, _ = load_source_records(path, ("head_fit",))
            self.assertEqual(head[0].concepts, (None, None))
            with self.assertRaisesRegex(ValueError, "protected"):
                load_source_records(path, ("confirmation",))
            _, protected, _ = load_source_records(path, ("confirmation",), allow_protected=True)
            self.assertEqual(protected[0].concepts, (None, None))
            with self.assertRaisesRegex(ValueError, "only responder_fit"):
                source_group_split([record_metadata(protected[0])])
            self.assertNotIn("target", repr(provenance))

    def test_annotation_status_required_and_nonfit_group_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prepared"
            prepare(path, records(4))
            original = read_jsonl(path / "samples.jsonl")
            malformed = copy.deepcopy(original)
            malformed[0]["concepts"][0].pop("annotation_status")
            with patch("cbmjev.typed_verifier.read_jsonl", side_effect=[read_jsonl(path / "membership.jsonl"), malformed]):
                with self.assertRaisesRegex(ValueError, "missing annotation"):
                    load_source_records(path)
            members = read_jsonl(path / "membership.jsonl")
            members[1]["split"] = "head_fit"
            with patch("cbmjev.typed_verifier.read_jsonl", return_value=members):
                with self.assertRaisesRegex(ValueError, "group crosses"):
                    load_source_records(path)

    def test_group_split_is_fixed_label_independent_and_rejects_duplicates(self):
        rows = [record_metadata(record) for record in records()]
        fit, tune, manifest = source_group_split(rows)
        self.assertFalse(set(manifest["fit_group_ids"]) & set(manifest["tune_group_ids"]))
        changed = [{**row, "concepts": [2, 4], "task_label": -999} for row in rows]
        self.assertEqual((fit, tune, manifest), source_group_split(changed))
        changed[1]["text_sha256"] = changed[2]["text_sha256"]
        with self.assertRaisesRegex(ValueError, "duplicate text"):
            source_group_split(changed)

    def test_cache_budget_rejects_before_forward_and_index_binds_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = encoder()
            with self.assertRaisesRegex(ValueError, "before allocation"):
                extract_feature_cache(model, records(2), Path(tmp) / "too_large", max_cache_bytes=1)
            self.assertEqual(model.backbone.calls, 0)
            progress = []
            index = extract_feature_cache(model, records(4), Path(tmp) / "cache", batch_size=8,
                                          shard_size=2, progress=progress.append)
            self.assertEqual([item["completed_samples"] for item in progress], [2, 4])
            self.assertFalse(any("sample_id" in item or "concepts" in item for item in progress))
            loaded, typed, shared = load_feature_cache(Path(tmp) / "cache")
            self.assertEqual(index, loaded)
            self.assertEqual(tuple(typed.shape), (4, 8, 8))
            self.assertEqual(tuple(shared.shape), (4, 8))
            corrupt = copy.deepcopy(index)
            corrupt["binding"]["batch_size"] += 1
            with patch("cbmjev.typed_verifier.read_json", return_value=corrupt):
                with self.assertRaisesRegex(ValueError, "index binding"):
                    load_feature_cache(Path(tmp) / "cache")

    def test_source_gate_exact_macro_and_collapse_rules_include_missing_classes(self):
        the_schema = schema()
        gold = torch.tensor([[i % 3, i % 5] for i in range(100)])
        collapsed = torch.zeros_like(gold)
        report = concept_metrics(collapsed, gold, the_schema)
        majority = copy.deepcopy(report)
        gate = source_gate_definition(the_schema, {}, "cache")
        result = evaluate_source_gate(report, majority, gate)
        self.assertTrue(result["upgrade_triggered"])
        self.assertTrue(result["macro_trigger"])
        self.assertTrue(result["collapse_trigger"])
        self.assertEqual(result["eligible_concepts"], ["c0", "c1"])
        perfect = concept_metrics(gold, gold, the_schema)
        self.assertFalse(evaluate_source_gate(perfect, majority, gate)["upgrade_triggered"])
        small = concept_metrics(gold[:12], gold[:12], the_schema)
        self.assertEqual(evaluate_source_gate(small, majority, gate)["eligible_concepts"], [])
        absent_classes = concept_metrics(torch.zeros((2, 2), dtype=torch.long),
                                         torch.zeros((2, 2), dtype=torch.long), the_schema)
        self.assertAlmostEqual(absent_classes["concepts"][0]["macro_f1"], 1 / 3)
        self.assertAlmostEqual(absent_classes["concepts"][1]["macro_f1"], 1 / 5)
        self.assertEqual(report["concepts"][1]["ordinal_mae"], 2.)

    def test_cached_training_freezes_gate_refits_both_and_never_runs_backbone(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            model = encoder()
            extract_feature_cache(model, records(), path / "cache", batch_size=16)
            calls = model.backbone.calls
            state = torch.random.get_rng_state().clone()
            real_metrics = concept_metrics

            def guarded_metrics(*args, **kwargs):
                self.assertTrue((path / "run" / "SOURCE_GATE.json").is_file())
                return real_metrics(*args, **kwargs)

            with patch("cbmjev.typed_verifier.concept_metrics", side_effect=guarded_metrics):
                metadata, report = train_cached_sources(path / "cache", path / "run", epochs=2, batch_size=8)
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            self.assertEqual(model.backbone.calls, calls)
            self.assertEqual(metadata["backbone_forwards_during_training"], 0)
            self.assertEqual(set(metadata["configs"]), {"typed", "shared"})
            self.assertEqual(report["typed"]["full_source_refit_n"], 20)
            self.assertEqual(report["shared"]["full_source_refit_n"], 20)
            prediction = predict_cached_sources(path / "cache", path / "run", path / "predictions.jsonl")
            self.assertEqual(prediction["timing_scope"], "CACHE_REPLAY_ONLY_NOT_END_TO_END_LATENCY")
            self.assertEqual(model.backbone.calls, calls)
            rows = read_jsonl(path / "predictions.jsonl")
            self.assertEqual(len(rows), 20)
            self.assertNotIn("concepts", rows[0])
            self.assertAlmostEqual(sum(rows[0]["typed_probabilities"][1]), 1., places=6)
            audit = audit_source_conditions(model, records(), path / "run", batch_size=7, limit=3)
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["n"], 3)
            altered = [SourceRecord(r.sample_id, r.group_id, r.role, r.text + " changed", r.concepts)
                       for r in records()]
            with self.assertRaisesRegex(ValueError, "audit source text"):
                audit_source_conditions(model, altered, path / "run", limit=3)
            binding = copy.deepcopy(metadata["binding"])
            binding["batch_size"] += 1
            with self.assertRaisesRegex(ValueError, "binding differ"):
                load_readouts(path / "run", expected_binding=binding)
            train_cached_sources(path / "cache", path / "repeat", epochs=2, batch_size=8)
            _, first = load_readouts(path / "run")
            _, repeated = load_readouts(path / "repeat")
            for kind in first:
                for key in first[kind].state_dict():
                    torch.testing.assert_close(first[kind].state_dict()[key], repeated[kind].state_dict()[key],
                                               rtol=0, atol=0)

    def test_fitting_rejects_non_source_cache_even_if_labels_are_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            raw = records(4)
            raw[0] = SourceRecord(raw[0].sample_id, raw[0].group_id, "policy_fit", raw[0].text, (None, None))
            extract_feature_cache(encoder(), raw, path / "cache")
            with self.assertRaisesRegex(ValueError, "only responder_fit"):
                train_cached_sources(path / "cache", path / "run", epochs=1)
            self.assertFalse((path / "run").exists())

    def test_prediction_metadata_binds_fit_roles_and_emitted_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            model = encoder()
            fit_provenance = {"membership_sha256": "b" * 64, "schema_file_sha256": "c" * 64,
                              "roles": ["responder_fit"], "protected_override": False}
            prediction_provenance = {**fit_provenance, "roles": ["head_fit"]}
            fit_index = extract_feature_cache(model, records(), path / "source", provenance=fit_provenance)
            metadata, _ = train_cached_sources(path / "source", path / "run", epochs=1)
            head = [SourceRecord("head", "head_group", "head_fit", "An independent head-fit text", (None, None))]
            prediction_index = extract_feature_cache(model, head, path / "head", provenance=prediction_provenance)
            report = predict_cached_sources(path / "head", path / "run", path / "pred.jsonl")
            self.assertEqual(report, read_json(path / "pred.jsonl.metadata.json"))
            self.assertEqual(report["fit_sample_ids"], metadata["fit_sample_ids"])
            self.assertEqual(report["fit_group_ids"], metadata["fit_group_ids"])
            self.assertNotIn("head", report["fit_sample_ids"])
            self.assertEqual(report["source_split"], metadata["source_split"])
            self.assertEqual(report["fit_provenance"], fit_provenance)
            self.assertEqual(report["provenance"], prediction_provenance)
            self.assertEqual(report["fit_source_cache_sha256"], fit_index["index_sha256"])
            self.assertEqual(report["source_cache_sha256"], prediction_index["index_sha256"])
            self.assertEqual(report["source_checkpoint_sha256"], file_hash(path / "run" / "checkpoint.json"))
            self.assertEqual(report["readout_weights_sha256"], file_hash(path / "run" / "readouts.pt"))
            self.assertEqual(report["predictions_sha256"], file_hash(path / "pred.jsonl"))
            self.assertEqual(set(read_jsonl(path / "pred.jsonl")[0]), {
                "sample_id", "group_id", "split", "typed_values", "typed_probabilities",
                "shared_values", "shared_probabilities"})
            self.assertNotIn(head[0].text, repr(report))
            self.assertNotIn("task_label", report)

    def test_prediction_protected_override_is_separate_from_fitting_permission(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            model = encoder()
            extract_feature_cache(model, records(), path / "source")
            train_cached_sources(path / "source", path / "run", epochs=1)
            protected = [SourceRecord("confirmation", "locked", "confirmation", "A held out text", (None, None))]
            extract_feature_cache(model, protected, path / "protected")
            with self.assertRaisesRegex(ValueError, "protected"):
                predict_cached_sources(path / "protected", path / "run", path / "pred.jsonl")
            self.assertFalse((path / "pred.jsonl").exists())
            predict_cached_sources(path / "protected", path / "run", path / "pred.jsonl", allow_protected=True)
            with self.assertRaisesRegex(ValueError, "protected roles forbidden"):
                train_cached_sources(path / "protected", path / "forbidden", epochs=1)

    def test_cli_defaults_match_frozen_recipe_and_never_download(self):
        parser = build_parser()
        args = parser.parse_args(["train", "--cache", "cache", "--output", "run"])
        self.assertEqual((args.seed, args.epochs, args.split_seed), (40, 10, 20260930))
        args = parser.parse_args(["extract", "--prepared", "data", "--cache", "cache",
                                  "--backbone", "local", "--model-revision", "a" * 40])
        self.assertFalse(args.allow_protected_roles)
        self.assertEqual(args.roles, ["responder_fit"])

    def test_prediction_cli_prints_only_aggregate_metadata_without_fit_ids(self):
        report = {"n": 2, "fit_sample_ids": ["PRIVATE_SOURCE_ID"], "fit_group_ids": ["PRIVATE_GROUP_ID"],
                  "source_checkpoint_sha256": "a" * 64, "readout_weights_sha256": "b" * 64,
                  "predictions_sha256": "c" * 64, "cached_readout_replay_seconds": .1,
                  "timing_scope": "CACHE_REPLAY_ONLY_NOT_END_TO_END_LATENCY"}
        output = StringIO()
        with patch("scripts.train_verification_sources.predict_cached_sources", return_value=report), redirect_stdout(output):
            main(["predict", "--cache", "cache", "--checkpoint", "run", "--output", "predictions.jsonl"])
        parsed = json.loads(output.getvalue())
        self.assertEqual(parsed["n"], 2)
        self.assertEqual(parsed["fit_n"], 1)
        self.assertNotIn("PRIVATE_SOURCE_ID", output.getvalue())
        self.assertNotIn("PRIVATE_GROUP_ID", output.getvalue())
        self.assertNotIn("fit_sample_ids", parsed)


if __name__ == "__main__":
    unittest.main()

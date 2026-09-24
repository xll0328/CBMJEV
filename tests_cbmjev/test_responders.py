import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from cbmjev.contracts import Concept, ModelInput, QueryGroup, Schema
from cbmjev import responders as responder_module
from cbmjev.responders import (ConceptTrainingExample, HashingTextResponder, SharedVisionResponder,
    fit_text_responder, load_responder, sanitize_model_input, save_responder, validate_examples)
from cbmjev.nanojev import (NanoCandidateScorer, NanoRiskController, NanoSemanticResponder,
    RiskTrainingExample, fit_nano_risk, fit_nano_semantics, last_valid_pool,
    load_local_nano, load_nano_head, risk_prompts, save_nano_head)


def schema():
    return Schema("synthetic_engineering", 2,
        (Concept("food", "Food is good", ("bad", "good")), Concept("price", "Price is low", ("high", "low"))),
        (QueryGroup("food", (0,)), QueryGroup("price", (1,))))


class FakeTokenizer:
    padding_side = "right"

    def __call__(self, prompts, **kwargs):
        values = [[1 + ord(char) % 127 for char in prompt] for prompt in prompts]
        length = max(map(len, values))
        ids, masks = [], []
        for row in values:
            pad = [0] * (length-len(row))
            ids.append(pad+row if self.padding_side == "left" else row+pad)
            masks.append([0]*len(pad)+[1]*len(row) if self.padding_side == "left" else [1]*len(row)+[0]*len(pad))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(128, 8)

    def forward(self, input_ids, attention_mask, position_ids, return_dict=True):
        hidden = self.embedding(input_ids) * attention_mask[..., None]
        # Context-dependent but deterministic fake, NOT a language model.
        hidden = hidden.cumsum(1) / (position_ids[..., None] + 1)
        return SimpleNamespace(last_hidden_state=hidden)


class FakeHFConfig:
    hidden_size = 8
    max_position_embeddings = 1024
    is_decoder = False
    is_encoder_decoder = False
    _commit_hash = "a" * 40

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "config.json").write_text(json.dumps({"model_type": "TEST_FAKE"}))


class FakeHFEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or FakeHFConfig()
        self.embedding = nn.Embedding(128, 8)
        self.dropout = nn.Dropout(.2)

    def forward(self, input_ids, attention_mask, return_dict=True):
        return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))


class FakeHFTokenizer(FakeTokenizer):
    pad_token_id = 0
    name_or_path = "TEST_FAKE_TOKENIZER"

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "tokenizer_config.json").write_text(json.dumps({"test_fake": True}))


def fake_hf_library():
    """No transformers import, network, or pretrained performance claim."""
    return SimpleNamespace(
        __version__="TEST_FAKE_NOT_TRANSFORMERS",
        AutoModel=SimpleNamespace(from_pretrained=Mock(side_effect=lambda *a, **k: FakeHFEncoder()),
                                  from_config=Mock(side_effect=lambda config, **k: FakeHFEncoder(config))),
        AutoConfig=SimpleNamespace(from_pretrained=Mock(side_effect=lambda *a, **k: FakeHFConfig())),
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(side_effect=lambda *a, **k: FakeHFTokenizer())))


class ResponderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.addCleanup(torch.use_deterministic_algorithms,
                        torch.are_deterministic_algorithms_enabled(),
                        warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
        torch.manual_seed(7)
        self.schema = schema()

    def scorer(self):
        return NanoCandidateScorer(FakeBackbone(), FakeTokenizer(), max_length=2048)

    def test_nano_candidate_scoring_disables_unused_decoder_cache(self):
        class CacheBackbone(FakeBackbone):
            def forward(self, *args, use_cache=True, **kwargs):
                self.last_use_cache = use_cache
                return super().forward(*args, **kwargs)
        backbone = CacheBackbone()
        backbone.config.use_cache = True
        scorer = NanoCandidateScorer(backbone, FakeTokenizer())
        scores = scorer.score_prompts(["a", "longer"])
        self.assertFalse(backbone.last_use_cache)
        self.assertTrue(backbone.config.use_cache)  # No global configuration mutation.
        scores.sum().backward()
        self.assertIsNotNone(scorer.head.weight.grad)
        self.assertIsNone(backbone.embedding.weight.grad)

    def test_hf_text_masked_pooling_fit_and_frozen_encoder_contract(self):
        examples = [ConceptTrainingExample(ModelInput(text="good good"), (1, None)),
                    ConceptTrainingExample(ModelInput(text="bad cheap"), (0, 1))]
        for freeze in (False, True):
            with self.subTest(freeze=freeze):
                model = responder_module.HFTextResponder(self.schema, FakeHFEncoder(), FakeHFTokenizer(),
                                                        freeze_backbone=freeze, max_length=64)
                before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
                model, report = responder_module.fit_responder(model, examples, epochs=2, seed=13)
                changes = [not torch.equal(v, model.encoder.state_dict()[k]) for k, v in before.items()]
                self.assertEqual(any(changes), not freeze)
                self.assertFalse(report["task_label_gradient"])
                self.assertEqual(report["observed_labels_per_atom"], [2, 1])
                model.train()
                self.assertEqual(model.encoder.training, not freeze)
                model.eval()
                one = model([ModelInput(text="a")])
                batch = model([ModelInput(text="a"), ModelInput(text="much longer")])
                for single, padded in zip(one, batch):
                    self.assertTrue(torch.allclose(single[0], padded[0], atol=1e-6))
                with self.assertRaisesRegex(ValueError, "max_length"):
                    model([ModelInput(text="x" * 65)])
                with self.assertRaisesRegex(ValueError, "text only"):
                    model([ModelInput(images=(b"not decoded",))])

    def test_hf_loader_requires_explicit_source_and_pins_tokenizer_revision(self):
        library = fake_hf_library()
        with patch.dict("sys.modules", {"transformers": library}):
            with self.assertRaisesRegex(ValueError, "explicit"):
                responder_module.load_hf_text_responder(self.schema, "")
            model = responder_module.load_hf_text_responder(self.schema, "org/explicit-encoder", revision="release")
        load_kwargs = library.AutoModel.from_pretrained.call_args.kwargs
        self.assertTrue(load_kwargs["local_files_only"])
        self.assertFalse(load_kwargs["trust_remote_code"])
        self.assertEqual(load_kwargs["revision"], "release")
        self.assertEqual(library.AutoTokenizer.from_pretrained.call_args.kwargs["revision"], "a" * 40)
        self.assertEqual(model.initialization["resolved_revision"], "a" * 40)
        self.assertEqual(len(model.initialization["encoder_initial_state_sha256"]), 64)

    def test_hf_invalid_masks_and_missing_label_heads_fail_or_stay_untrained(self):
        model = responder_module.HFTextResponder(self.schema, FakeHFEncoder(), FakeHFTokenizer())
        examples = [ConceptTrainingExample(ModelInput(text="some text"), (1, None))]
        untouched = {key: value.clone() for key, value in model.heads[1].state_dict().items()}
        responder_module.fit_responder(model, examples, epochs=1)
        for key, value in untouched.items():
            self.assertTrue(torch.equal(value, model.heads[1].state_dict()[key]))
        tokenizer = Mock(padding_side="right", return_value={"input_ids": torch.tensor([[0]]),
                                                             "attention_mask": torch.tensor([[0]])})
        model.tokenizer = tokenizer
        with self.assertRaisesRegex(ValueError, "all-padding"):
            model([ModelInput(text="x")])
        with self.assertRaisesRegex(ValueError, "responder_fit"):
            responder_module.fit_responder(model, [ConceptTrainingExample(ModelInput(text="x"), (1, 0), "test")])

    def test_hf_checkpoint_is_full_local_and_detects_tokenizer_mutation(self):
        library = fake_hf_library()
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"transformers": library}):
            root = Path(directory)
            model = responder_module.load_hf_text_responder(self.schema, "org/test-only", allow_download=True)
            payload = ModelInput(text="hello")
            model.eval()
            before = model.respond(payload, (0, 1))
            save_responder(model, root / "original" / "responder.pt")
            shutil.copytree(root / "original", root / "moved")
            library.AutoModel.from_pretrained.reset_mock()
            restored = load_responder(root / "moved" / "responder.pt", self.schema)
            library.AutoModel.from_pretrained.assert_not_called()
            self.assertEqual(restored.respond(payload, (0, 1)), before)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]))
            self.assertTrue(library.AutoConfig.from_pretrained.call_args.kwargs["local_files_only"])
            self.assertTrue(library.AutoTokenizer.from_pretrained.call_args.kwargs["local_files_only"])
            checkpoint = torch.load(root / "moved" / "responder.pt", map_location="cpu", weights_only=True)
            checkpoint["state_dict"].pop("encoder.embedding.weight")
            torch.save(checkpoint, root / "moved" / "missing_weight.pt")
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                load_responder(root / "moved" / "missing_weight.pt", self.schema)
            config_path = next((root / "moved").glob("*_hf_assets/tokenizer_config.json"))
            config_path.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "HF.*assets.*changed"):
                load_responder(root / "moved" / "responder.pt", self.schema)

    def test_optional_real_transformers_tiny_local_roundtrip_without_download(self):
        try:
            import transformers
        except ImportError:
            self.skipTest("optional transformers unavailable; fake contract tests are not a real-library gate")
        # Actual library/architecture compatibility gate, NOT pretrained performance.
        # Every config/tokenizer/weight file is generated in the temporary fixture.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tiny_random_encoder"
            source.mkdir()
            (source / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngood\nbad\ncheap\nfood\n")
            tokenizer = transformers.BertTokenizer(vocab_file=str(source / "vocab.txt"))
            tokenizer.save_pretrained(source)
            config = transformers.BertConfig(vocab_size=9, hidden_size=16, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=32, max_position_embeddings=64)
            transformers.BertModel(config).save_pretrained(source)
            model = responder_module.load_hf_text_responder(self.schema, source, max_length=32)
            examples = [ConceptTrainingExample(ModelInput(text="good cheap food"), (1, None)),
                        ConceptTrainingExample(ModelInput(text="bad food"), (0, 0))]
            model, _ = responder_module.fit_responder(model, examples, epochs=1, batch_size=2, learning_rate=3e-5)
            expected = model([e.payload for e in examples])
            checkpoint = root / "artifact" / "responder.pt"
            save_responder(model, checkpoint)
            source.rename(root / "unavailable_original_source")
            with patch.object(transformers.AutoModel, "from_pretrained", side_effect=AssertionError("reload must not fetch source")):
                restored = load_responder(checkpoint, self.schema)
            for left, right in zip(expected, restored([e.payload for e in examples])):
                self.assertTrue(torch.equal(left, right))

    def test_model_input_rejects_metadata_and_paths(self):
        for field in ("sample_id", "path", "split", "y", "gold", "concept_descriptions"):
            with self.assertRaises(ValueError):
                sanitize_model_input({"text": "good food", field: "secret"})
        self.assertEqual(sanitize_model_input({"text": "good food"}).text, "good food")
        with self.assertRaises((TypeError, ValueError)):
            sanitize_model_input({"images": ("/private/file.jpg",)})

    def test_text_responder_actually_fits_semantics(self):
        examples = [ConceptTrainingExample(ModelInput(text=f"{'great' if a else 'awful'} {'cheap' if b else 'costly'}"), (a, b))
                    for _ in range(8) for a in (0, 1) for b in (0, 1)]
        model, report = fit_text_responder(examples, self.schema, vocab_size=512, embedding_dim=16,
                                          epochs=25, batch_size=16, learning_rate=.03, seed=8)
        predictions = [model.respond(e.payload, (0, 1)) for e in examples]
        self.assertEqual(predictions, [e.concepts for e in examples])
        self.assertLess(report["training_loss"][-1], report["training_loss"][0])
        self.assertFalse(report["task_label_gradient"])

    def test_sanitize_rejects_modelinput_with_added_metadata(self):
        payload = ModelInput(text="good food")
        object.__setattr__(payload, "sample_id", "forbidden")
        with self.assertRaises(ValueError):
            sanitize_model_input(payload)

    def test_lazy_example_validation_scans_once_not_per_concept(self):
        class LazyExamples:
            reads = 0

            def __len__(self):
                return 3

            def __iter__(self):
                for _ in range(3):
                    self.reads += 1
                    yield ConceptTrainingExample(ModelInput(text="good cheap"), (1, 1))

        examples = LazyExamples()
        self.assertEqual(validate_examples(examples, self.schema), [3, 3])
        self.assertEqual(examples.reads, 3)

    def test_gold_missing_is_masked_not_runtime_unknown(self):
        examples = [ConceptTrainingExample(ModelInput(text="good"), (1, None))]
        model, report = fit_text_responder(examples, self.schema, epochs=2, seed=4)
        self.assertEqual(report["observed_labels_per_atom"], [1, 0])
        self.assertIn(model.respond(examples[0].payload, (1,))[0], (0, 1))

    def test_training_reports_explicit_determinism_and_environment(self):
        examples = [ConceptTrainingExample(ModelInput(text="good good cheap"), (1, None))]
        for deterministic in (True, False):
            with self.subTest(deterministic=deterministic):
                kwargs = {} if deterministic else {"deterministic": False}
                _, report = fit_text_responder(examples, self.schema, epochs=1, seed=11,
                                              device="cpu", **kwargs)
                self.assertIs(report["deterministic"], deterministic)
                self.assertIs(torch.are_deterministic_algorithms_enabled(), deterministic)
                self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
                self.assertEqual(report["device"], "cpu")
                self.assertEqual(report["torch_version"], str(torch.__version__))
                self.assertEqual(report["cublas_workspace_config"], os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
                self.assertEqual(report["seed"], 11)

    def test_training_rejects_nonboolean_determinism(self):
        examples = [ConceptTrainingExample(ModelInput(text="good good"), (1, None))]
        for value in (0, 1, "true", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "deterministic must be boolean"):
                fit_text_responder(examples, self.schema, epochs=1, deterministic=value)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable on local test machine")
    def test_cuda_mean_embedding_bag_backward_is_same_seed_repeatable(self):
        # CUBLAS_WORKSPACE_CONFIG must be set by the launcher before CUDA initialization.
        # Repeated token indices and partial labels exercise both bag and gather gradients.
        examples = [
            ConceptTrainingExample(ModelInput(text="good good cheap cheap"), (1, None)),
            ConceptTrainingExample(ModelInput(text="bad bad costly costly"), (0, 0)),
            ConceptTrainingExample(ModelInput(text="good good costly"), (None, 0)),
            ConceptTrainingExample(ModelInput(text="bad bad cheap"), (0, 1)),
        ]
        states = []
        for _ in range(2):
            model, report = fit_text_responder(examples, self.schema, vocab_size=32, embedding_dim=8,
                                              epochs=2, batch_size=2, seed=23, device="cuda:0",
                                              deterministic=True)
            torch.cuda.synchronize("cuda:0")
            self.assertEqual(model.encoder.mode, "mean")
            self.assertIs(report["deterministic"], True)
            self.assertEqual(report["observed_labels_per_atom"], [3, 3])
            self.assertEqual(report["device"], "cuda:0")
            self.assertTrue(all(torch.isfinite(torch.tensor(report["training_loss"]))))
            states.append({name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
        self.assertEqual(states[0].keys(), states[1].keys())
        for name in states[0]:
            self.assertTrue(torch.equal(states[0][name], states[1][name]), name)

    def test_training_rejects_heldout_or_runtime_labels(self):
        for example in (ConceptTrainingExample(ModelInput(text="good"), (1, 0), "test"),
                        ConceptTrainingExample(ModelInput(text="good"), (2, 0)),
                        ConceptTrainingExample(ModelInput(text="good"), (None, None))):
            with self.assertRaises(ValueError):
                fit_text_responder([example], self.schema, epochs=1)

    def test_cheap_episode_encodes_once_and_counts_all_heads(self):
        model = HashingTextResponder(self.schema)
        calls = []
        hook = model.encoder.register_forward_hook(lambda *args: calls.append(1))
        session = model.start_session(ModelInput(text="good food"))
        session.respond(())
        self.assertEqual(calls, [])
        session.respond((0,))
        self.assertEqual(session.last_stats["atoms_computed"], 2)
        session.respond((1,))
        self.assertEqual(len(calls), 1)
        self.assertTrue(session.last_stats["cache_hit"])
        hook.remove()

    def test_threshold_and_request_validation(self):
        model = HashingTextResponder(self.schema, threshold=1.)
        self.assertEqual(model.respond(ModelInput(text="good"), (0, 1)), (2, 2))
        for action in ((0, 0), (2,), (True,)):
            with self.assertRaises(ValueError):
                model.respond(ModelInput(text="good"), action)

    def test_responder_checkpoint_roundtrip_and_schema_guard(self):
        model = HashingTextResponder(self.schema)
        payload = ModelInput(text="great cheap")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "responder.pt"
            save_responder(model, path)
            restored = load_responder(path, self.schema)
            self.assertEqual(model.respond(payload, (0, 1)), restored.respond(payload, (0, 1)))
            with self.assertRaises(FileExistsError):
                save_responder(model, path)
            other = Schema("changed", 2, self.schema.concepts, self.schema.groups)
            with self.assertRaises(ValueError):
                load_responder(path, other)

    def test_last_valid_pool_left_and_right_padding(self):
        hidden = torch.arange(24.).reshape(2, 3, 4)
        mask = torch.tensor([[0, 1, 1], [1, 1, 0]])
        self.assertTrue(torch.equal(last_valid_pool(hidden, mask), torch.stack([hidden[0, 2], hidden[1, 1]])))
        with self.assertRaises(ValueError):
            last_valid_pool(hidden, torch.zeros(2, 3))

    def test_scorer_padding_invariance_and_no_truncation(self):
        model = self.scorer().eval()
        right = model.score_prompts(["a", "longer input"])
        model.tokenizer.padding_side = "left"
        left = model.score_prompts(["a", "longer input"])
        self.assertTrue(torch.allclose(left, right, atol=1e-6))
        model.max_length = 3
        with self.assertRaisesRegex(ValueError, "truncate"):
            model.score_prompts(["too long"])

    def test_nano_risk_is_independent_bce_not_action_softmax(self):
        model = self.scorer()
        original = {k: v.clone() for k, v in model.backbone.state_dict().items()}
        examples = [RiskTrainingExample((-1, -1), action, 1.) for _ in range(4) for action in ((), (0,), (1,))]
        model, report = fit_nano_risk(model, examples, self.schema, epochs=30, batch_size=12, learning_rate=.1)
        controller = NanoRiskController(model, self.schema)
        probabilities = controller.predict((-1, -1), ((), (0,), (1,)))
        self.assertGreater(min(probabilities), .8)
        self.assertGreater(sum(probabilities), 2.4)
        self.assertEqual(controller.objective, "risk")
        self.assertFalse(report["calibrated"])
        for key, value in original.items():
            self.assertTrue(torch.equal(value, model.backbone.state_dict()[key]))

    def test_risk_prompt_contains_only_semantic_history_and_candidate(self):
        prompts = risk_prompts(self.schema, (1, -1), ((), (1,)))
        self.assertIn("good", prompts[0])
        self.assertIn("STOP", prompts[0])
        self.assertNotIn("sample_id", prompts[0])
        with self.assertRaises(ValueError):
            risk_prompts(self.schema, (1, -1), ((0,),))
        with self.assertRaises(ValueError):
            fit_nano_risk(self.scorer(), [RiskTrainingExample((-1, -1), (), 0, "test")], self.schema)

    def test_nano_semantic_training_and_typed_response(self):
        model = self.scorer()
        examples = [ConceptTrainingExample(ModelInput(text="awful costly"), (0, None))]
        model, report = fit_nano_semantics(model, examples, self.schema, epochs=3)
        self.assertEqual(report["loss"], "masked_semantic_candidate_CE")
        session = NanoSemanticResponder(model, self.schema).start_session(examples[0].payload)
        self.assertIn(session.respond((0,))[0], (0, 1))
        self.assertEqual(session.last_stats["candidate_paths"], 2)
        self.assertEqual(session.last_stats["atoms_computed"], 1)

    def test_nano_export_roundtrip_and_task_guard(self):
        model = self.scorer()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nano.pt"
            save_nano_head(model, path, self.schema)
            restored = self.scorer()
            restored.backbone.load_state_dict(model.backbone.state_dict())
            load_nano_head(path, self.schema, scorer=restored, expected_task="risk")
            self.assertTrue(torch.equal(model.score_prompts(["same"]), restored.score_prompts(["same"])))
            with self.assertRaises(ValueError):
                load_nano_head(path, self.schema, scorer=restored, expected_task="semantic")
            with self.assertRaises(ValueError):
                load_local_nano("/definitely/nonexistent/local/checkpoint")

    def test_nano_local_backbone_manifest_rejects_mutation(self):
        model = self.scorer()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text('{"hidden_size":8}')
            (root / "model.safetensors").write_bytes(b"synthetic manifest test, not real weights")
            model.backbone_local_path = str(root)
            save_nano_head(model, root / "head.pt", self.schema)
            data = torch.load(root / "head.pt", weights_only=True)
            self.assertNotIn("head.pt", data["backbone_manifest"]["files"])
            (root / "config.json").write_text('{"hidden_size":16}')
            with self.assertRaisesRegex(ValueError, "content changed"):
                load_nano_head(root / "head.pt", self.schema)

    def test_frozen_backbone_stays_eval_during_head_training(self):
        model = self.scorer()
        model.train()
        self.assertTrue(model.training)
        self.assertTrue(model.head.training)
        self.assertFalse(model.backbone.training)

    def test_export_rejects_source_changed_since_model_load(self):
        from cbmjev.nanojev import backbone_manifest
        model = self.scorer()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text('{"hidden_size":8}')
            model.backbone_local_path = str(root)
            model.backbone_source_manifest = backbone_manifest(root)
            (root / "config.json").write_text('{"hidden_size":16}')
            with self.assertRaisesRegex(ValueError, "after model loading"):
                save_nano_head(model, root / "head.pt", self.schema)

    def test_optional_vision_bytes_forward_without_download(self):
        try:
            import torchvision  # noqa: F401
            from PIL import Image
        except ImportError:
            self.skipTest("optional torchvision/Pillow unavailable")
        encoded = io.BytesIO()
        Image.new("RGB", (32, 32), (30, 60, 90)).save(encoded, format="PNG")
        model = SharedVisionResponder(self.schema, image_size=32).eval()
        session = model.start_session(ModelInput(images=(encoded.getvalue(), encoded.getvalue())))
        self.assertEqual(len(session.respond((0, 1))), 2)
        self.assertEqual(session.last_stats["encoder_forwards"], 1)


if __name__ == "__main__":
    unittest.main()

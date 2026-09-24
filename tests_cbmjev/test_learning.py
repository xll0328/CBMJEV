import copy
import json
from itertools import islice
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from cbmjev.baselines import EmpiricalLookaheadPolicy, fit_static_order
from cbmjev.config import resolve_config
from cbmjev.contracts import Concept, DeclaredCost, QueryGroup, Schema, candidate_actions
from cbmjev.learning import (ActionController, action_targets, encode_states, fit_models,
                            iter_risk_training_examples, load_models, mask_answers,
                            normalize_config, policy_examples, sample_group_masks,
                            save_models, training_rows)


def fixture_schema():
    return Schema("synthetic-learning-tests", 2,
                  tuple(Concept("c%d" % i, "Synthetic binary %d" % i, ("absent", "present")) for i in range(4)),
                  (QueryGroup("paired_atoms", (0, 1)), QueryGroup("second", (2,)), QueryGroup("third", (3,))))


def fixture_rows():
    rng = random.Random(19)
    rows = []
    for split, count in (("head_fit", 32), ("policy_fit", 24), ("validation", 8), ("test", 8)):
        for index in range(count):
            z = [rng.randrange(2) for _ in range(4)]
            rows.append({"sample_id": "%s-%d" % (split, index), "group_id": "%s-g%d" % (split, index),
                         "split": split, "z": z, "y": z[0] ^ z[2]})
    return rows


SMALL_CONFIG = {"seed": 23, "device": "cpu", "hidden": 16, "head_epochs": 2,
                "policy_epochs": 2, "batch_size": 16, "masks_per_sample": 3,
                "max_pair_actions": 3, "learning_rate": 0.01}


class GroupMaskTests(unittest.TestCase):
    def test_masks_are_group_atomic_and_independent_of_hidden_values(self):
        schema = fixture_schema()
        first = mask_answers((1, 0, 1, 0), (False, True, False), schema)
        poisoned = mask_answers((123456, -99999, 1, 987654), (False, True, False), schema)
        self.assertEqual(first, poisoned)
        self.assertEqual(first, (-1, -1, 1, -1))
        self.assertTrue(torch.equal(encode_states((first,), schema), encode_states((poisoned,), schema)))
        rng_a, rng_b = random.Random(13), random.Random(13)
        self.assertEqual(sample_group_masks(40, schema.num_groups, rng_a),
                         sample_group_masks(40, schema.num_groups, rng_b))

    def test_partial_group_and_bad_values_are_rejected(self):
        schema = fixture_schema()
        for state in ((1, -1, -1, -1), (1, 0, 999, -1), (1, 0, float("nan"), -1), (0, 1)):
            with self.assertRaises(ValueError):
                encode_states((state,), schema)

    def test_runtime_statuses_distinct_from_unqueried(self):
        schema = fixture_schema()
        states = ((-1, -1, -1, -1), (2, 2, -1, -1), (3, 3, -1, -1), (0, 0, -1, -1))
        encodings = encode_states(states, schema)
        for index in range(len(states)):
            for other in range(index):
                self.assertFalse(torch.equal(encodings[index], encodings[other]))

    def test_policy_generation_stream_is_bounded_and_group_complete(self):
        schema, rows = fixture_schema(), fixture_rows()
        config = normalize_config(SMALL_CONFIG, schema)
        stream = policy_examples(training_rows(rows, "policy_fit", schema), schema, config)
        self.assertIs(iter(stream), stream)
        examples = list(islice(stream, 12))
        self.assertTrue(examples)
        for before, action, after, _ in examples:
            schema.validate_state(before)
            schema.validate_state(after)
            changed_visibility = tuple(i for i, value in enumerate(before) if value < 0 and after[i] >= 0)
            self.assertEqual(changed_visibility, schema.expand(action))

    def test_candidate_sampling_never_uses_future_answers(self):
        schema = fixture_schema()
        config = normalize_config({**SMALL_CONFIG, "actions_per_state": 2}, schema)
        left = list(policy_examples([((0, 0, 0, 0), 0)], schema, config))
        right = list(policy_examples([((1, 1, 1, 1), 1)], schema, config))
        # Visible values may differ, but sampled acquisition masks/actions cannot.
        project = lambda examples: [(tuple(v >= 0 for v in before), action) for before, action, _, _ in examples]
        self.assertEqual(project(left), project(right))


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema, cls.rows = fixture_schema(), fixture_rows()
        cls.head, cls.controller, cls.report = fit_models(cls.rows, cls.schema, SMALL_CONFIG)

    def test_trained_interfaces_are_finite_and_frozen(self):
        state = self.schema.empty_state()
        probabilities = self.head.probabilities(state)
        self.assertAlmostEqual(sum(probabilities), 1, places=6)
        self.assertIn(self.head.predict(state), (0, 1))
        actions = candidate_actions(state, self.schema, pairs=self.controller.pairs)
        risks = self.controller.predict(state, actions)
        self.assertEqual(len(risks), len(actions))
        self.assertTrue(all(0 <= value <= 1 for value in risks))
        self.assertTrue(all(not parameter.requires_grad for parameter in self.head.network.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in self.controller.network.parameters()))
        self.assertEqual(self.report["training_protocol"], "DISJOINT_HEAD_FIT_POLICY_FIT_NO_FINAL_REFIT")

    def test_risks_are_independent_sigmoids_not_action_softmax(self):
        config = normalize_config(SMALL_CONFIG, self.schema)
        controller = ActionController(self.schema, config)
        for parameter in controller.network.parameters():
            parameter.data.zero_()
        actions = candidate_actions(self.schema.empty_state(), self.schema, pairs=controller.pairs)
        risks = controller.predict(self.schema.empty_state(), actions)
        self.assertEqual(risks, (0.5,) * len(actions))
        self.assertNotAlmostEqual(sum(risks), 1.0)

    def test_hidden_validation_test_targets_cannot_change_weights(self):
        poisoned = copy.deepcopy(self.rows)
        for row in poisoned:
            if row["split"] in {"validation", "test"}:
                row["y"] = -987654  # Training does not even inspect these labels.
                row["z"] = [999999] * self.schema.num_atoms
        head, controller, report = fit_models(poisoned, self.schema, SMALL_CONFIG)
        for left, right in zip(self.head.network.parameters(), head.network.parameters()):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(self.controller.network.parameters(), controller.network.parameters()):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(self.report["head_loss"], report["head_loss"])
        self.assertEqual(self.report["policy_loss"], report["policy_loss"])

    def test_policy_fit_labels_do_not_train_task_head(self):
        changed = copy.deepcopy(self.rows)
        for row in changed:
            if row["split"] == "policy_fit":
                row["y"] = 1 - row["y"]
        head, _, _ = fit_models(changed, self.schema, SMALL_CONFIG)
        for left, right in zip(self.head.network.parameters(), head.network.parameters()):
            self.assertTrue(torch.equal(left, right))

    def test_group_leakage_between_training_roles_rejected(self):
        changed = copy.deepcopy(self.rows)
        next(row for row in changed if row["split"] == "policy_fit")["group_id"] = changed[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "overlaps"):
            fit_models(changed, self.schema, SMALL_CONFIG)

    def test_actual_risk_and_signed_value_targets(self):
        before = self.schema.empty_state()
        after = (1, 0, -1, -1)
        y = 1
        examples = [(before, (), before, y), (before, (0,), after, y)]
        risks = action_targets(self.head, examples, "risk").tolist()
        self.assertEqual(risks, [float(self.head.predict(before) != y), float(self.head.predict(after) != y)])
        drops = action_targets(self.head, examples, "value").tolist()
        expected = math.log(self.head.probabilities(after)[y]) - math.log(self.head.probabilities(before)[y])
        self.assertEqual(drops[0], 0.0)
        self.assertAlmostEqual(drops[1], expected, places=6)

    def test_value_training_and_signed_outputs(self):
        head, controller, report = fit_models(self.rows, self.schema, {**SMALL_CONFIG, "objective": "value"})
        self.assertEqual(report["objective"], "value")
        for parameter in controller.network.parameters():
            parameter.data.zero_()
        controller.network[-1].bias.data.fill_(-2.0)
        scores = controller.predict(self.schema.empty_state(), ((), (0,), (0, 1)))
        self.assertEqual(scores, (0.0, -2.0, -2.0))
        for left, right in zip(self.head.network.parameters(), head.network.parameters()):
            self.assertTrue(torch.equal(left, right))

    def test_public_nano_risk_examples_are_streamed_and_label_restricted(self):
        examples = list(islice(iter_risk_training_examples(self.rows, self.head, self.schema, SMALL_CONFIG), 20))
        self.assertTrue(examples)
        for example in examples:
            self.assertEqual(set(example), {"observed", "action", "error", "split"})
            self.assertEqual(example["split"], "policy_fit")
            self.assertIn(example["error"], (0.0, 1.0))
            self.schema.validate_state(example["observed"])

    def test_checkpoint_roundtrip_uses_weights_only_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_models(path, self.head, self.controller, self.report)
            with patch("cbmjev.learning.torch.load", wraps=torch.load) as loader:
                head, controller, report = load_models(path, self.schema)
                self.assertTrue(loader.call_args.kwargs["weights_only"])
            state = (1, 0, -1, -1)
            actions = candidate_actions(state, self.schema, pairs=self.controller.pairs)
            self.assertEqual(self.head.probabilities(state), head.probabilities(state))
            self.assertEqual(self.controller.predict(state, actions), controller.predict(state, actions))
            self.assertEqual(report, self.report)
            with self.assertRaises(ValueError):
                save_models(path, self.head, self.controller)
            wrong = Schema("another-dataset", self.schema.num_classes, self.schema.concepts, self.schema.groups)
            with self.assertRaises(ValueError):
                load_models(path, wrong)

    def test_controller_invalid_state_action_or_nonfinite_output_rejected(self):
        with self.assertRaises(ValueError):
            self.controller.predict((1, -1, -1, -1), ((),))
        with self.assertRaises(ValueError):
            self.controller.predict(self.schema.empty_state(), ((1, 0),))
        with self.assertRaises(ValueError):
            self.controller.predict((1, 0, -1, -1), ((0,),))
        damaged = ActionController(self.schema, normalize_config(SMALL_CONFIG, self.schema))
        damaged.network[-1].bias.data.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            damaged.predict(self.schema.empty_state(), ((),))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable on local test machine")
    def test_cuda_fit_and_cpu_checkpoint_reload(self):
        head, controller, report = fit_models(self.rows, self.schema, {**SMALL_CONFIG, "device": "cuda"})
        self.assertEqual(head.device.type, "cuda")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cuda.pt"
            save_models(path, head, controller, report)
            cpu_head, cpu_controller, _ = load_models(path, self.schema, "cpu")
            self.assertEqual(cpu_head.device.type, "cpu")
            self.assertTrue(all(math.isfinite(score) for score in cpu_controller.predict(self.schema.empty_state(), ((), (0,)))))


class ClassWeightingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = fixture_schema()
        rows = fixture_rows()
        head_rows = [row for row in rows if row["split"] == "head_fit"][:20]
        for index, row in enumerate(head_rows):
            row["y"] = int(index >= 16)
        cls.rows = head_rows + [row for row in rows if row["split"] != "head_fit"]
        cls.config = {**SMALL_CONFIG, "objective": "value", "class_weighting": "inverse_frequency"}
        cls.head, cls.controller, cls.report = fit_models(cls.rows, cls.schema, cls.config)

    def test_config_opt_in_is_strict_and_defaults_to_none(self):
        self.assertEqual(normalize_config(SMALL_CONFIG, self.schema)["class_weighting"], "none")
        self.assertEqual(resolve_config({})["learning"]["class_weighting"], "none")
        self.assertEqual(resolve_config({"learning": {"class_weighting": "inverse_frequency"}})
                         ["learning"]["class_weighting"], "inverse_frequency")
        for value in (True, None, [], "balanced", "inverse_frequencies"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "class_weighting"):
                normalize_config({**SMALL_CONFIG, "class_weighting": value}, self.schema)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "class_weighting"):
                resolve_config({"learning": {"class_weighting": value}})

    def test_weights_use_head_rows_only_and_are_reported_in_class_index_order(self):
        self.assertEqual(self.report["head_class_counts"], [16, 4])
        self.assertEqual(self.report["head_class_weights"], [0.625, 2.5])
        self.assertEqual(self.head.class_weights, (0.625, 2.5))
        self.assertEqual(self.report["class_weight_source"], "head_fit_labels_only")
        self.assertEqual(self.report["head_loss_definition"], "mean_examples(w_y * CE)")
        self.assertEqual(self.report["value_target_definition"], "w_y * (CE_before - CE_after)")
        self.assertEqual(sum(n * w for n, w in zip(self.report["head_class_counts"],
                                                  self.head.class_weights)) / 20, 1.0)

    def test_singleton_batch_retains_fixed_weight_in_loss_and_gradient(self):
        logits = torch.tensor([[0.2, -0.4]], requires_grad=True)
        target = torch.tensor([1])
        loss = self.head.cross_entropy(logits, target)
        expected = torch.nn.functional.cross_entropy(logits, target) * 2.5
        self.assertTrue(torch.equal(loss, expected))
        actual_gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, logits)[0]
        self.assertTrue(torch.equal(actual_gradient, expected_gradient))
        both_logits = torch.tensor([[0.2, -0.4], [0.2, -0.4]])
        both_targets = torch.tensor([0, 1])
        per_sample = self.head.cross_entropy(both_logits, both_targets, reduction="none")
        for index in range(2):
            self.assertEqual(per_sample[index].item(),
                             self.head.cross_entropy(both_logits[index:index + 1], both_targets[index:index + 1]).item())
        self.assertTrue(torch.equal(self.head.cross_entropy(both_logits, both_targets), per_sample.mean()))

    def test_weighted_value_is_actual_signed_loss_drop_and_risk_stays_binary(self):
        before, after = self.schema.empty_state(), (1, 0, -1, -1)
        examples = [(before, (), before, 1), (before, (0,), after, 0), (before, (0,), after, 1)]
        values = action_targets(self.head, examples, "value").tolist()
        self.assertEqual(values[0], 0.0)
        for index, y in enumerate((0, 1), 1):
            expected = self.head.class_weights[y] * (math.log(self.head.probabilities(after)[y])
                                                     - math.log(self.head.probabilities(before)[y]))
            self.assertAlmostEqual(values[index], expected, places=6)
        risk = action_targets(self.head, examples, "risk").tolist()
        self.assertEqual(risk, [float(self.head.predict(state) != y) for _, _, state, y in examples])
        nano_examples = list(islice(iter_risk_training_examples(self.rows, self.head, self.schema, self.config), 10))
        self.assertTrue(all(row["error"] in (0.0, 1.0) for row in nano_examples))

    def test_heldout_poison_does_not_change_weighted_fit(self):
        rows = copy.deepcopy(self.rows)
        rows += [{"split": "calibration", "y": "FORBIDDEN", "z": "FORBIDDEN"},
                 {"split": "responder_fit", "y": "FORBIDDEN", "z": "FORBIDDEN"}]
        for row in rows:
            if row["split"] not in {"head_fit", "policy_fit"}:
                row["y"], row["z"] = "FORBIDDEN", "FORBIDDEN"
        head, controller, report = fit_models(rows, self.schema, self.config)
        self.assertEqual(report, self.report)
        for left, right in ((self.head, head), (self.controller, controller)):
            for first, second in zip(left.network.parameters(), right.network.parameters()):
                self.assertTrue(torch.equal(first, second))

    def test_policy_labels_never_set_head_weights_or_parameters(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["split"] == "policy_fit":
                row["y"] = 0
        head, _, report = fit_models(rows, self.schema, self.config)
        self.assertEqual(head.class_weights, self.head.class_weights)
        self.assertEqual(report["head_class_counts"], self.report["head_class_counts"])
        for first, second in zip(self.head.network.parameters(), head.network.parameters()):
            self.assertTrue(torch.equal(first, second))

    def test_weighted_risk_and_value_share_identical_head(self):
        head, controller, report = fit_models(self.rows, self.schema, {**self.config, "objective": "risk"})
        self.assertEqual(head.class_weights, self.head.class_weights)
        self.assertEqual(report["risk_target_definition"], "unweighted_indicator_of_post_action_error")
        for first, second in zip(self.head.network.parameters(), head.network.parameters()):
            self.assertTrue(torch.equal(first, second))
        self.assertTrue(all(0 <= score <= 1 for score in controller.predict(self.schema.empty_state(), ((), (0,)))))

    def test_missing_head_class_fails_closed_only_when_weighting_enabled(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["split"] == "head_fit":
                row["y"] = 0
        with self.assertRaisesRegex(ValueError, "every class in head_fit"):
            fit_models(rows, self.schema, self.config)
        head, _, report = fit_models(rows, self.schema, {**self.config, "class_weighting": "none"})
        self.assertIsNone(head.class_weights)
        self.assertEqual(report["head_class_counts"], [20, 0])

    def test_default_and_explicit_none_are_bitwise_identical(self):
        implicit = {key: value for key, value in self.config.items() if key != "class_weighting"}
        head, controller, report = fit_models(self.rows, self.schema, implicit)
        other_head, other_controller, other_report = fit_models(self.rows, self.schema,
                                                               {**implicit, "class_weighting": "none"})
        self.assertEqual(report, other_report)
        self.assertIsNone(head.class_weights)
        for left, right in ((head, other_head), (controller, other_controller)):
            for first, second in zip(left.network.parameters(), right.network.parameters()):
                self.assertTrue(torch.equal(first, second))
        logits = torch.tensor([[0.2, -0.4], [-0.7, 0.8]])
        targets = torch.tensor([0, 1])
        for reduction in ("mean", "none"):
            self.assertTrue(torch.equal(head.cross_entropy(logits, targets, reduction),
                                        torch.nn.functional.cross_entropy(logits, targets, reduction=reduction)))
        self.assertTrue(any(not torch.equal(first, second)
                            for first, second in zip(head.network.parameters(), self.head.network.parameters())))

    def test_weighted_checkpoint_roundtrip_and_malformed_metadata_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weighted.pt"
            save_models(path, self.head, self.controller, self.report)
            head, controller, report = load_models(path, self.schema)
            self.assertEqual(head.class_weights, self.head.class_weights)
            self.assertEqual(report, self.report)
            observed = self.schema.empty_state()
            self.assertEqual(head.probabilities(observed), self.head.probabilities(observed))
            self.assertEqual(controller.predict(observed, ((), (0,))), self.controller.predict(observed, ((), (0,))))
            examples = [(observed, (0,), (1, 0, -1, -1), 1)]
            self.assertTrue(torch.equal(action_targets(head, examples, "value"),
                                        action_targets(self.head, examples, "value")))
            payload = torch.load(path, weights_only=True)
            for index, invalid in enumerate((None, [1.0], [1.0, float("nan")], [1.0, 0.0],
                                             [True, 1.0], [1.0, "2.5"], [1.0, 1e300])):
                damaged = {**payload, "head_class_weights": invalid}
                candidate = Path(directory) / ("bad-%d.pt" % index)
                torch.save(damaged, candidate)
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "class weights"):
                    load_models(candidate, self.schema)

    def test_legacy_none_checkpoint_loads_without_loss_metadata(self):
        head, controller, _ = fit_models(self.rows, self.schema, {**self.config, "class_weighting": "none"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current-none.pt"
            save_models(path, head, controller)
            payload = torch.load(path, weights_only=True)
            payload["config"].pop("class_weighting")
            payload.pop("head_class_weights")
            legacy = Path(directory) / "legacy-none.pt"
            torch.save(payload, legacy)
            restored, _, _ = load_models(legacy, self.schema)
            self.assertIsNone(restored.class_weights)
            self.assertEqual(restored.probabilities(self.schema.empty_state()),
                             head.probabilities(self.schema.empty_state()))


class ExactXorHead:
    def __init__(self, schema):
        self.schema = schema

    def predict(self, state):
        return state[0] ^ state[1] if all(value >= 0 for value in state) else 0

    def probabilities(self, state):
        pred = self.predict(state)
        return (0.99, 0.01) if pred == 0 else (0.01, 0.99)

    def probabilities_many(self, states):
        return tuple(self.probabilities(state) for state in states)


def xor_fixture():
    schema = Schema("synthetic-xor", 2,
                    (Concept("a", "bit a", ("0", "1")), Concept("b", "bit b", ("0", "1"))),
                    (QueryGroup("a", (0,)), QueryGroup("b", (1,))))
    rows = [{"sample_id": "p-%d-%d" % (a, b), "group_id": "g-%d-%d" % (a, b),
             "split": "policy_fit", "z": [a, b], "y": a ^ b}
            for a in range(2) for b in range(2)]
    return schema, rows, ExactXorHead(schema)


class BaselineTests(unittest.TestCase):
    def test_static_order_is_one_training_only_global_order(self):
        schema, rows, head = xor_fixture()
        order, report = fit_static_order(rows, head, schema)
        changed = rows + [{"sample_id": "test", "group_id": "test", "split": "test", "z": [999], "y": -1}]
        other, other_report = fit_static_order(changed, head, schema)
        self.assertEqual(order, (0, 1))
        self.assertEqual(order, other)
        self.assertEqual(report, other_report)
        self.assertEqual(report["fit_split"], "policy_fit")

    def test_two_step_lookahead_finds_training_distribution_complementarity(self):
        schema, rows, head = xor_fixture()
        one = EmpiricalLookaheadPolicy.fit(rows, head, schema, depth=1, include_pairs=False, include_all=False)
        two = EmpiricalLookaheadPolicy.fit(rows, head, schema, depth=2, include_pairs=False, include_all=False)
        actions = ((), (0,), (1,))
        cost = DeclaredCost(per_group=1)
        self.assertEqual(one.choose(schema.empty_state(), actions, 2, cost, 0.1), ())
        self.assertEqual(two.choose(schema.empty_state(), actions, 2, cost, 0.1), (0,))
        values = two.action_values(schema.empty_state(), actions, 2, cost, 0.1)
        self.assertAlmostEqual(values[0], 0.5)
        self.assertAlmostEqual(values[1], 0.2)
        self.assertFalse(two.report["population_bayes_optimal"])
        self.assertFalse(two.report["reproduces_aco_or_brig"])

    def test_simulated_setup_cost_uses_each_successor_history(self):
        schema, rows, head = xor_fixture()
        policy = EmpiricalLookaheadPolicy.fit(rows, head, schema, depth=None, include_pairs=False, include_all=False)
        cost = DeclaredCost(setup=2.0, call=0.0, per_group=1.0)
        values = policy.action_values(schema.empty_state(), ((), (0,)), 4, cost, 0.1)
        self.assertAlmostEqual(values[1], 0.4)  # setup+first=3, second=1, NOT 3+3.
        self.assertEqual(policy.choose(schema.empty_state(), ((), (0,)), 4, cost, 0.1), (0,))

    def test_unseen_history_is_explicit_backoff_not_test_oracle(self):
        schema, rows, head = xor_fixture()
        policy = EmpiricalLookaheadPolicy.fit(rows, head, schema, depth=2)
        # Runtime UNCERTAIN code 2 was not in the training response support.
        scores = policy.action_values((2, -1), ((), (1,)), 1, DeclaredCost(), 0.1)
        self.assertTrue(all(math.isfinite(value) for value in scores))
        self.assertIn("global", policy.report["unseen_history_rule"])

    def test_large_k_and_excessive_search_rejected(self):
        schema, rows, head = xor_fixture()
        with self.assertRaises(ValueError):
            EmpiricalLookaheadPolicy.fit(rows, head, schema, max_groups=1)
        policy = EmpiricalLookaheadPolicy.fit(rows, head, schema, depth=None, max_states=1,
                                            include_pairs=False, include_all=False)
        with self.assertRaisesRegex(ValueError, "max_states"):
            policy.action_values(schema.empty_state(), ((0,),), 2, DeclaredCost(), 0.1)

    def test_budget_feasibility_and_stop_tie(self):
        schema, rows, head = xor_fixture()
        policy = EmpiricalLookaheadPolicy.fit(rows, head, schema)
        self.assertEqual(policy.choose(schema.empty_state(), ((), (0,)), 0, DeclaredCost(), 0.0), ())
        values = policy.action_values(schema.empty_state(), ((), (0,)), 0, DeclaredCost(), 0.0)
        self.assertTrue(math.isinf(values[1]))

    def test_positive_infinite_budget_is_valid_but_nan_negative_infinity_are_not(self):
        schema, rows, head = xor_fixture()
        policy = EmpiricalLookaheadPolicy.fit(rows, head, schema, depth=None,
                                            include_pairs=False, include_all=False)
        actions = ((), (0,), (1,))
        cost = DeclaredCost(per_group=0.2)
        values = policy.action_values(schema.empty_state(), actions, math.inf, cost, 0.1)
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertEqual(policy.choose(schema.empty_state(), actions, math.inf, cost, 0.1), (0,))
        for budget in (float("nan"), -math.inf, -1.0):
            with self.assertRaises(ValueError):
                policy.choose(schema.empty_state(), actions, budget, cost)

    def test_nonunit_cost_does_not_replace_remaining_group_cap(self):
        schema, rows, head = xor_fixture()
        policy = EmpiricalLookaheadPolicy.fit(rows, head, schema, depth=None,
                                            include_pairs=True, include_all=True)
        cost = DeclaredCost(setup=0.3, call=0.2, per_group=0.05)
        singles = ((), (0,), (1,))
        # Monetary budget is unlimited, but the deployment only permits one
        # further group. A two-group XOR completion is therefore unavailable.
        capped = policy.action_values(schema.empty_state(), singles, math.inf, cost, 0.1,
                                      remaining_groups=1)
        self.assertAlmostEqual(capped[0], 0.5)
        self.assertAlmostEqual(capped[1], 0.5 + 0.1 * 0.55)
        self.assertEqual(policy.choose(schema.empty_state(), singles, math.inf, cost, 0.1,
                                      remaining_groups=1), ())
        uncapped = policy.action_values(schema.empty_state(), singles, math.inf, cost, 0.1,
                                        remaining_groups=2)
        self.assertAlmostEqual(uncapped[1], 0.1 * (0.55 + 0.25))
        self.assertEqual(policy.choose(schema.empty_state(), singles, math.inf, cost, 0.1,
                                      remaining_groups=2), (0,))
        values = policy.action_values(schema.empty_state(), ((), (0, 1)), math.inf, cost,
                                      remaining_groups=1)
        self.assertTrue(math.isinf(values[1]))

    def test_remaining_groups_validation_and_zero_stop(self):
        schema, rows, head = xor_fixture()
        policy = EmpiricalLookaheadPolicy.fit(rows, head, schema)
        for remaining in (-1, 3, 1.5, True):
            with self.assertRaises(ValueError):
                policy.choose(schema.empty_state(), ((), (0,)), math.inf, DeclaredCost(),
                              remaining_groups=remaining)
        self.assertEqual(policy.choose(schema.empty_state(), ((), (0,)), math.inf,
                                       DeclaredCost(), remaining_groups=0), ())

    def test_runtime_does_not_write_infinite_budget_into_json_scores(self):
        from cbmjev.runtime import choose_action
        schema, rows, head = xor_fixture()
        policy = EmpiricalLookaheadPolicy.fit(rows, head, schema)
        action, scores = choose_action(schema.empty_state(), ((), (0,), (1,)), method="lookahead",
                                       controller=policy, cost=DeclaredCost(per_group=0.1),
                                       cost_weight=0.1, remaining_budget=math.inf, remaining_groups=1)
        serialized = json.dumps({"action": action, "scores": scores}, allow_nan=False)
        self.assertNotIn("Infinity", serialized)
        self.assertEqual(scores, {})


if __name__ == "__main__":
    unittest.main()

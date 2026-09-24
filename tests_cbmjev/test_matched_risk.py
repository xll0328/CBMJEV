import copy
import dataclasses
import unittest
from unittest.mock import patch

import torch
from torch import nn

from cbmjev.contracts import Concept, QueryGroup, Schema
from cbmjev.learning import ActionController, encode_actions, encode_states, normalize_config
from cbmjev.matched_risk import fit_matched_mlp_risk, matched_event_manifest
from cbmjev.nanojev import RiskTrainingExample, fit_nano_risk, risk_prompts


def fixture():
    schema = Schema("matched-risk-unit", 2,
                    (Concept("a", "A", ("no", "yes")), Concept("b", "B", ("no", "yes"))),
                    (QueryGroup("a", (0,)), QueryGroup("b", (1,))))
    events = (RiskTrainingExample((-1, -1), (), 1.),
              RiskTrainingExample((-1, -1), (0,), 0.),
              RiskTrainingExample((1, -1), (1,), .25),
              RiskTrainingExample((2, -1), (), 1.),
              RiskTrainingExample((3, -1), (1,), 0.))
    return schema, events


class ReferenceScorer(nn.Module):
    """Exercise the actual Nano fitting loop with identical MLP forward math.

    NOT a fake empirical Nano result: this only checks optimizer/exposure parity.
    """
    freeze_backbone = False

    def __init__(self, schema, events, seed, hidden):
        super().__init__()
        cfg = normalize_config({"seed": seed, "hidden": hidden, "objective": "risk"}, schema)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.network = ActionController(schema, cfg).network
        self.inputs, self.seen = {}, []
        for e in events:
            prompt = risk_prompts(schema, e.observed, [e.action])[0]
            self.inputs[prompt] = torch.cat((encode_states([e.observed], schema),
                                            encode_actions([e.action], schema)), -1)

    def score_prompts(self, prompts):
        self.seen.extend(prompts)
        return self.network(torch.cat([self.inputs[p] for p in prompts])).flatten()


class MatchedRiskTests(unittest.TestCase):
    def test_actual_nano_loop_optimizer_and_tail_weighting_match(self):
        schema, events = fixture()
        args = dict(epochs=2, batch_size=3, learning_rate=.003, seed=21)
        scorer = ReferenceScorer(schema, events, seed=21, hidden=8)
        _, nano_report = fit_nano_risk(scorer, events, schema, **args)
        controller, report = fit_matched_mlp_risk(events, schema, hidden=8, **args)
        for key, value in scorer.network.state_dict().items():
            self.assertTrue(torch.equal(value, controller.network.state_dict()[key]), key)
        self.assertEqual(nano_report["training_loss"], report["training_loss_minibatch_mean"])
        self.assertEqual(report["event_exposures"], 10)
        self.assertEqual(report["optimizer_steps"], 4)
        self.assertEqual(len(scorer.seen), 10)
        generator = torch.Generator().manual_seed(21)
        expected = [risk_prompts(schema, events[int(i)].observed, [events[int(i)].action])[0]
                    for _ in range(2) for i in torch.randperm(len(events), generator=generator)]
        self.assertEqual(scorer.seen, expected)
        self.assertNotEqual(report["training_loss"], report["training_loss_minibatch_mean"])

    def test_deterministic_frozen_runtime_and_events_unchanged(self):
        schema, events = fixture()
        previous = copy.deepcopy(events)
        rng = torch.random.get_rng_state().clone()
        a, ra = fit_matched_mlp_risk(events, schema, epochs=2, hidden=4, batch_size=2)
        b, rb = fit_matched_mlp_risk(events, schema, epochs=2, hidden=4, batch_size=2)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(events, previous)
        self.assertEqual(ra, rb)
        self.assertEqual(a.objective, "risk")
        self.assertEqual(a.predict((-1, -1), [(), (0,)]), b.predict((-1, -1), [(), (0,)]))
        self.assertTrue(all(0 <= x <= 1 for x in a.predict((-1, -1), [(), (0,)])))
        self.assertTrue(all(not p.requires_grad for p in a.network.parameters()))

    def test_manifest_binds_schema_event_order_duplicates_and_targets(self):
        schema, events = fixture()
        base = matched_event_manifest(events, schema)
        self.assertEqual(base, matched_event_manifest(list(events), schema))
        for changed in (events[::-1], events + events[:1],
                        (dataclasses.replace(events[0], error=0.),) + events[1:]):
            self.assertNotEqual(base["event_sha256"], matched_event_manifest(changed, schema)["event_sha256"])
        renamed = dataclasses.replace(schema, dataset="other-schema")
        self.assertNotEqual(base["event_sha256"], matched_event_manifest(events, renamed)["event_sha256"])

    def test_reject_invalid_or_leaky_event_inputs(self):
        schema, events = fixture()
        invalid = [dataclasses.replace(events[0], error=value)
                   for value in (float("nan"), float("inf"), -1., 2., True)]
        invalid += [dataclasses.replace(events[0], split="validation"),
                    dataclasses.replace(events[0], observed=[-1, -1]),
                    dataclasses.replace(events[0], action=[0]),
                    dataclasses.replace(events[0], observed=(99, -1)),
                    dataclasses.replace(events[0], action=(1, 0)),
                    RiskTrainingExample((1, -1), (0,), 0.),
                    {"observed": (-1, -1), "action": (), "error": 0., "y": 1}]
        for event in invalid:
            with self.subTest(event=event), self.assertRaises(ValueError):
                fit_matched_mlp_risk((event,), schema, epochs=1)
        for args in (dict(epochs=True), dict(batch_size=0), dict(seed=True),
                     dict(learning_rate=float("nan")), dict(hidden=0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                fit_matched_mlp_risk(events, schema, **args)

    def test_only_observed_and_action_project_into_forward(self):
        schema, events = fixture()
        # Changing offline target must leave the feature projection unchanged.
        alternate = tuple(dataclasses.replace(e, error=1-e.error) for e in events)
        captured = []

        def instrumented_controller(*args, **kwargs):
            controller = ActionController(*args, **kwargs)
            controller.network.register_forward_pre_hook(
                lambda module, inputs: captured.append(inputs[0].detach().clone()))
            return controller

        with patch("cbmjev.matched_risk.ActionController", side_effect=instrumented_controller):
            _, report = fit_matched_mlp_risk(events, schema, epochs=1, hidden=4, batch_size=3)
            before = list(captured)
            captured.clear()
            fit_matched_mlp_risk(alternate, schema, epochs=1, hidden=4, batch_size=3)
        self.assertEqual(len(before), len(captured))
        for left, right in zip(before, captured):
            self.assertTrue(torch.equal(left, right))
            self.assertEqual(left.shape[1], sum(schema.num_categories) + schema.num_atoms + schema.num_groups + 1)
        self.assertEqual(report["model_input_fields"], ["observed", "action"])
        self.assertFalse(report["targets_regenerated"])
        self.assertFalse(report["upstream_target_provenance_verified"])
        self.assertFalse(report["paper_evidence"])


if __name__ == "__main__":
    unittest.main()

import dataclasses
import math
import random
import unittest
from unittest.mock import patch

from cbmjev.config import resolve_config
from cbmjev.matched_risk import matched_event_manifest
from cbmjev.nanojev import RiskTrainingExample
from tests_cbmjev.test_matched_risk import fixture
from tools.diagnose_matched_risk_fit import (diagnose_events, losses, reconstruct_training_events,
                                            score_events, validation_events)


class MatchedRiskDiagnosticTests(unittest.TestCase):
    def test_bce_brier_references_and_stop_partition(self):
        events = (RiskTrainingExample((-1,-1), (), 0.),
                  RiskTrainingExample((-1,-1), (), 1.),
                  RiskTrainingExample((-1,-1), (0,), 1.))
        result = diagnose_events(events, {"constant": [.5,.5,.5]}, events)
        self.assertAlmostEqual(result["all"]["models"]["constant"]["bce"], math.log(2))
        self.assertEqual(result["all"]["models"]["constant"]["brier"], .25)
        self.assertEqual(result["all"]["singleton_buckets"], 1)
        self.assertEqual(result["all"]["references"]["state_action_insample"]["brier"], 1/6)
        self.assertEqual(result["stop"]["n"], 2)
        self.assertEqual(result["nonstop"]["n"], 1)
        self.assertEqual(result["nonstop"]["references"]["state_action_insample"]["brier"], 0.)

    def test_validation_reference_uses_training_labels_with_global_fallback(self):
        train = (RiskTrainingExample((-1,-1), (), 0.), RiskTrainingExample((-1,-1), (0,), 1.))
        val = (RiskTrainingExample((-1,-1), (), 1., "validation"),
               RiskTrainingExample((-1,-1), (1,), 0., "validation"))
        r = diagnose_events(val, {"model": [.5,.5]}, train)
        self.assertEqual(r["all"]["unseen_in_training_bucket_events"], 1)
        self.assertEqual(r["all"]["references"]["train_state_action_fallback_global"]["brier"], .625)
        self.assertEqual(r["all"]["references"]["state_action_insample"]["brier"], 0.)
        self.assertEqual(r["stop"]["references"]["train_state_action_fallback_global"]["brier"], 1.)

    def test_exact_training_reservoir_and_digest_fail_closed(self):
        schema, source = fixture()
        cfg = resolve_config({"seed": 7})
        reservoir, rng = [], random.Random(7)
        for n, e in enumerate(source,1):
            if len(reservoir)<3:
                reservoir.append(e)
            else:
                i=rng.randrange(n)
                if i<3: reservoir[i]=e
        receipt = {"sampling":"seeded_reservoir", "retained_examples":3,"available_examples":5,
                   "seed":7,"event_sha256":matched_event_manifest(tuple(reservoir),schema)["event_sha256"]}
        rows=[{"observed":e.observed,"action":e.action,"error":e.error} for e in source]
        with patch("tools.diagnose_matched_risk_fit.iter_risk_training_examples", return_value=iter(rows)):
            self.assertEqual(reconstruct_training_events([],None,schema,cfg,receipt),tuple(reservoir))
        with patch("tools.diagnose_matched_risk_fit.iter_risk_training_examples", return_value=iter(rows)):
            with self.assertRaisesRegex(ValueError,"digest/count"):
                reconstruct_training_events([],None,schema,cfg,{**receipt,"event_sha256":"changed"})

    def test_empty_slices_and_invalid_probabilities(self):
        e=(RiskTrainingExample((-1,-1), (), 0.),)
        r=diagnose_events(e,{"model":[.5]},e)
        self.assertEqual(r["nonstop"]["models"]["model"],{"n":0,"bce":None,"brier":None})
        for scores in ([],[float("nan")],[1.1]):
            with self.assertRaises(ValueError): losses(e,scores)
        with self.assertRaises(ValueError):
            losses((dataclasses.replace(e[0],error=float("nan")),),[.5])

    def test_forward_inputs_independent_of_event_targets(self):
        import torch
        schema, events = fixture()
        class MLP:
            calls=[]
            def predict(self,state,actions):
                self.calls.append((state,actions))
                return (.4,) * len(actions)
        class Scorer:
            calls=[]
            def score_prompts(self,prompts):
                self.calls.extend(prompts)
                return torch.zeros(len(prompts))
        m,s=MLP(),Scorer()
        first=score_events(events,schema,m,s,2)
        mc,sc=list(m.calls),list(s.calls)
        m.calls.clear();s.calls.clear()
        second=score_events(tuple(dataclasses.replace(e,error=1-e.error) for e in events),schema,m,s,2)
        self.assertEqual(first,second)
        self.assertEqual(mc,m.calls)
        self.assertEqual(sc,s.calls)

    def test_validation_selects_only_validation_and_preserves_split(self):
        schema,_=fixture(); cfg=resolve_config({})
        examples=[((-1,-1),(),(-1,-1),0)]
        import torch
        with patch("tools.diagnose_matched_risk_fit.training_rows",return_value=[]) as selection, \
             patch("tools.diagnose_matched_risk_fit.policy_examples",return_value=iter(examples)), \
             patch("tools.diagnose_matched_risk_fit.action_targets",return_value=torch.tensor([1.])):
            events=validation_events([],None,schema,cfg,3)
        selection.assert_called_once_with([],"validation",schema)
        self.assertEqual(events[0].split,"validation")


if __name__ == "__main__":
    unittest.main()

from dataclasses import replace
import json
import unittest

import torch

from cbmjev.choice_features import (choice_prompts, encode_choice_features,
                                   choice_feature_manifest, choice_feature_width)
from cbmjev.choice_targets import ChoiceInputs, ChoiceSupervision, ChoiceExample
from cbmjev.contracts import Schema, Concept, QueryGroup
from tests_cbmjev.test_contracts_runtime import fixture_schema


class ChoiceFeatureTests(unittest.TestCase):
    def setUp(self):
        self.schema = fixture_schema()
        self.inputs = ChoiceInputs((-1, -1, -1), ((), (0,), (1,)), 2, (0., 2., 1.), .1)

    def test_supervision_metadata_changes_cannot_enter_inputs(self):
        a = ChoiceExample(self.inputs, ChoiceSupervision((0, 1, 1), (.8, .1, .1), .3),
                          "secret_sample_a", {"y": 0, "hidden_answers": [0, 1, 2]})
        b = replace(a, supervision=ChoiceSupervision((1, 0, 0), (.1, .4, .5), 9.),
                    derived_id="secret_sample_b", source={"y": 1, "hidden_answers": [1, 0, 0]})
        self.assertEqual(choice_prompts(a.model_inputs, self.schema), choice_prompts(b.model_inputs, self.schema))
        self.assertTrue(torch.equal(encode_choice_features(a.model_inputs, self.schema),
                                    encode_choice_features(b.model_inputs, self.schema)))
        for forbidden in (a, a.supervision, a.source, vars(self.inputs)):
            with self.assertRaises(ValueError): choice_prompts(forbidden, self.schema)
        self.assertNotIn("secret_sample", str(choice_prompts(self.inputs, self.schema)))

    def test_candidate_permutation_equivariance(self):
        order = (2, 0, 1)
        changed = replace(self.inputs, actions=tuple(self.inputs.actions[i] for i in order),
                          incremental_costs=tuple(self.inputs.incremental_costs[i] for i in order))
        p = choice_prompts(self.inputs, self.schema)
        self.assertEqual(choice_prompts(changed, self.schema), tuple(p[i] for i in order))
        torch.testing.assert_close(encode_choice_features(changed, self.schema),
                                   encode_choice_features(self.inputs, self.schema)[list(order)])

    def test_missing_status_multiatom_group_and_no_ids(self):
        inputs = replace(self.inputs, observed=(1, 0, -1), actions=((), (1,)), incremental_costs=(0., 1.))
        p = json.loads(choice_prompts(inputs, self.schema)[0])
        self.assertEqual(p['observed_state'], [{'concept': 'A', 'status': 'OBSERVED', 'value': 'yes'},
                         {'concept': 'B', 'status': 'OBSERVED', 'value': 'no'}])
        self.assertEqual(p['unobserved_group_indices'], [1])
        self.assertIn('UNOBSERVED', p['missing_semantics'])
        acquired = json.loads(choice_prompts(self.inputs, self.schema)[1])
        self.assertEqual(len(acquired['candidate']['concepts']), 2)
        renamed = Schema('different_dataset', 2,
            tuple(Concept('id'+str(i), c.description, c.values) for i,c in enumerate(self.schema.concepts)),
            tuple(QueryGroup('group'+str(i), g.atoms) for i,g in enumerate(self.schema.groups)))
        self.assertEqual(choice_prompts(self.inputs, renamed), choice_prompts(self.inputs, self.schema))
        self.assertTrue(torch.equal(encode_choice_features(self.inputs, renamed), encode_choice_features(self.inputs, self.schema)))

    def test_budget_cost_and_runtime_statuses_visible(self):
        for changed in (replace(self.inputs, remaining_groups=1), replace(self.inputs, cost_weight=.2),
                        replace(self.inputs, incremental_costs=(0., 3., 1.))):
            self.assertNotEqual(choice_prompts(changed, self.schema), choice_prompts(self.inputs, self.schema))
            self.assertFalse(torch.equal(encode_choice_features(changed, self.schema), encode_choice_features(self.inputs, self.schema)))
        status = replace(self.inputs, observed=(2, 3, -1), actions=((), (1,)), incremental_costs=(0.,1.))
        p = json.loads(choice_prompts(status, self.schema)[0])
        self.assertEqual([x['status'] for x in p['observed_state']], ['UNCERTAIN','NOT_APPLICABLE'])
        self.assertEqual(p['unobserved_group_indices'], [1])

    def test_stop_only_width_hash_stability(self):
        inputs = replace(self.inputs, remaining_groups=0, actions=((),), incremental_costs=(0.,))
        x = encode_choice_features(inputs, self.schema)
        self.assertEqual(x.shape, (1, choice_feature_width(self.schema)))
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(json.loads(choice_prompts(inputs, self.schema)[0])['candidate'], {'type':'STOP'})
        self.assertEqual(choice_feature_manifest(inputs, self.schema), choice_feature_manifest(inputs, self.schema))
        self.assertEqual(x[0,-3:].tolist(), [0., float(torch.tensor(.1)), 0.])

    def test_illegal_visible_inputs_fail_closed(self):
        bad = [replace(self.inputs, observed=(0,-1,-1)), replace(self.inputs, observed=(9,0,-1)),
               replace(self.inputs, actions=((), (0,), (0,))),
               replace(self.inputs, actions=((0,),), incremental_costs=(1.,)),
               replace(self.inputs, observed=(0,0,-1)),
               replace(self.inputs, actions=((), (0,0), (1,))),
               replace(self.inputs, actions=((), (2,), (1,))),
               replace(self.inputs, remaining_groups=0), replace(self.inputs, remaining_groups=True),
               replace(self.inputs, incremental_costs=(1.,2.,1.)),
               replace(self.inputs, cost_weight=float('nan')),
               replace(self.inputs, incremental_costs=(0.,float('inf'),1.)),
               replace(self.inputs, cost_weight=1e100), replace(self.inputs, cost_weight=-1.),
               replace(self.inputs, incremental_costs=(0.,)), replace(self.inputs, actions=[])]
        for inputs in bad:
            for f in (choice_prompts, encode_choice_features):
                with self.subTest(inputs=inputs, function=f.__name__):
                    with self.assertRaises(ValueError): f(inputs, self.schema)


if __name__ == '__main__':
    unittest.main()

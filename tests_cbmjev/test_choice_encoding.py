import unittest
from unittest.mock import patch

import torch

from cbmjev.choice_encoding import encode_frozen_choice_features
from cbmjev.choice_features import choice_prompts
from cbmjev.choice_targets import ChoiceInputs
from cbmjev.nanojev import NanoCandidateScorer
from tests_cbmjev.test_responders import FakeTokenizer, FakeBackbone, schema


class ChoiceEncodingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.schema = schema()
        self.q = (ChoiceInputs((-1,-1), ((),(0,),(1,)), 2, (0.,1.,1.), .1),
                  ChoiceInputs((1,-1), ((),(1,)), 1, (0.,1.), .1))
        self.scorer = NanoCandidateScorer(FakeBackbone(), FakeTokenizer(), max_length=4096,
                                          feature_normalization="layernorm")

    def encode(self, **kwargs):
        options = dict(max_questions=2, max_padded_tokens=1600, max_candidates_per_batch=2)
        options.update(kwargs)
        return encode_frozen_choice_features(self.scorer, self.q, self.schema, **options)

    def test_chunk_budget_order_and_raw_feature_parity(self):
        paths = [p for q in self.q for p in choice_prompts(q,self.schema)]
        with torch.no_grad(): expected = self.scorer.features(paths)
        with patch.object(self.scorer, 'features', wraps=self.scorer.features) as call:
            values, report = self.encode()
        self.assertEqual([len(x) for x in values], [3,2])
        torch.testing.assert_close(torch.cat(values), expected)
        self.assertEqual(call.call_count, report['forward_calls'])
        self.assertEqual([p for c in call.call_args_list for p in c.args[0]], paths)
        self.assertTrue(all(c['padded_tokens']<=1600 for c in report['chunks']))
        self.assertTrue(all(c['end_candidate_exclusive']-c['first_candidate']<=2 for c in report['chunks']))
        self.assertEqual(report['actual_tokens'],sum(len(p) for p in paths))
        self.assertGreaterEqual(report['padded_tokens'],report['actual_tokens'])
        self.assertTrue(all(x.device.type=='cpu' and not x.requires_grad for x in values))
        self.assertFalse(report['normalization_cached'])
        # LayerNorm/scalar changes must not enter raw cached representations.
        with torch.no_grad(): self.scorer.norm.weight.fill_(10); self.scorer.head.weight.fill_(20)
        other,_ = self.encode()
        torch.testing.assert_close(torch.cat(values),torch.cat(other))

    def test_late_overlong_preflight_before_first_backbone_call(self):
        paths=[p for q in self.q for p in choice_prompts(q,self.schema)]
        # All first-question candidates fit; the last second-question candidate
        # is longer. Failure must still occur before ANY backbone forward.
        self.scorer.max_length = max(map(len,paths[:3]))
        self.assertGreater(len(paths[-1]),self.scorer.max_length)
        with patch.object(self.scorer,'features') as call:
            with self.assertRaisesRegex(ValueError,'max_length'): self.encode()
            call.assert_not_called()
        self.scorer.max_length=4096
        with patch.object(self.scorer,'features') as call:
            with self.assertRaisesRegex(ValueError,'max_padded_tokens'): self.encode(max_padded_tokens=1)
            call.assert_not_called()

    def test_frozen_eval_required_and_limits(self):
        self.scorer.freeze_backbone=False
        with self.assertRaises(ValueError):self.encode()
        self.scorer.freeze_backbone=True;self.scorer.backbone.train()
        with self.assertRaises(ValueError):self.encode()
        self.scorer.backbone.eval();next(self.scorer.backbone.parameters()).requires_grad_(True)
        with self.assertRaises(ValueError):self.encode()
        self.scorer.backbone.requires_grad_(False)
        for kw in ({'max_questions':1},{'max_candidates_per_batch':0},{'max_padded_tokens':True}):
            with self.assertRaises(ValueError):self.encode(**kw)

    def test_sequence_hash_invariant_to_chunking_and_manifest_explicit(self):
        a,ra=self.encode(max_candidates_per_batch=1)
        b,rb=self.encode(max_candidates_per_batch=5,max_padded_tokens=10000)
        torch.testing.assert_close(torch.cat(a),torch.cat(b))
        for k in ('prompt_sequence_sha256','token_id_sequence_sha256','actual_tokens'):
            self.assertEqual(ra[k],rb[k])
        self.assertIsNone(ra['backbone_manifest_record_sha256'])
        self.assertFalse(ra['paper_speed_evidence'])
        self.assertEqual(ra['forward_calls'],5)
        self.assertEqual(rb['forward_calls'],1)


if __name__=='__main__':unittest.main()

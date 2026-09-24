"""Small real-backbone chunk-parity probe; training inputs only, no task evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from cbmjev.choice_encoding import encode_frozen_choice_features
from cbmjev.choice_targets import ChoiceInputs
from cbmjev.contracts import Schema, stable_hash
from cbmjev.io import file_hash, write_json
from cbmjev.nanojev import load_local_nano
from cbmjev.pipeline import code_fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache','backbone','out'): p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-candidates',type=int,default=4)
    args=p.parse_args()
    if Path(args.out).exists(): raise FileExistsError(args.out)
    cache=Path(args.cache)
    def identity():
        return {'core_source':code_fingerprint(),'probe_source':file_hash(__file__),
                'schema':file_hash(cache/'schema.json'),'responses':file_hash(cache/'responses.jsonl'),
                'manifest':file_hash(cache/'manifest.json')}
    before=identity()
    schema=Schema.from_dict(json.loads((cache/'schema.json').read_text()))
    row=None
    with (cache/'responses.jsonl').open() as stream:
        for line in stream:
            candidate=json.loads(line)
            if candidate['split']=='policy_fit':
                row=candidate;break
    if row is None:raise ValueError('no policy_fit row')
    # Labels, task head and validation/test measurements are never consulted.
    observations=[];questions=[]
    for count in (0,4,14):
        if count>schema.num_groups:raise ValueError('probe requires at least14groups')
        state=[-1]*schema.num_atoms
        for group in schema.groups[:count]:
            for atom in group.atoms:state[atom]=row['z'][atom]
        actions=((),)+tuple((g,) for g in range(count,schema.num_groups))
        observations.append(tuple(state))
        questions.append(ChoiceInputs(tuple(state),actions,schema.num_groups-count,
                                      tuple(0. if not a else 1. for a in actions),.1))
    torch.manual_seed(17)
    scorer=load_local_nano(args.backbone,device=args.device,max_length=8192,freeze_backbone=True)
    scorer.eval()
    options=dict(max_questions=3,max_padded_tokens=8192)
    singles,single_report=encode_frozen_choice_features(scorer,tuple(questions),schema,
                                               max_candidates_per_batch=1,**options)
    chunked,chunk_report=encode_frozen_choice_features(scorer,tuple(questions),schema,
                                      max_candidates_per_batch=args.max_candidates,**options)
    for key in ('prompt_sequence_sha256','token_id_sequence_sha256','question_candidate_counts'):
        assert single_report[key]==chunk_report[key],key
    comparisons=[]
    for count,a,b in zip((0,4,14),singles,chunked):
        delta=(a-b).abs()
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-4)
        comparisons.append({'acquired_groups':count,'shape':list(a.shape),
            'max_abs_difference':float(delta.max()),'mean_abs_difference':float(delta.mean()),
            'single_raw_sha256':hashlib.sha256(a.contiguous().numpy().tobytes()).hexdigest(),
            'chunk_raw_sha256':hashlib.sha256(b.contiguous().numpy().tobytes()).hexdigest()})
    if identity()!=before:raise ValueError('source or cache changed during probe')
    report={'status':'REAL_QWEN_ENCODING_PROBE_ONLY','paper_evidence':False,'test_evaluated':False,
        'source_identity':before,'sampling':'first policy_fit row in cache file; canonical acquired-group prefixes0/4/14; every legal singleton plus STOP',
        'selected_input_identity_sha256':stable_hash({'sample_id':row['sample_id'],'states':observations}),
        'max_length':8192,'tolerance':{'atol':2e-4,'rtol':2e-4},'comparisons':comparisons,
        'single':single_report,'chunked':chunk_report,
        'limitations':'Three histories from one training row; not full dataset bound, downstream result, or speed benchmark. No raw features/text copied to receipt.'}
    write_json(args.out,report)
    print(json.dumps({'out':args.out,'comparisons':comparisons,'single_calls':single_report['forward_calls'],
                      'chunk_calls':chunk_report['forward_calls']}))


if __name__=='__main__':main()

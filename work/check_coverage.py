"""Numerical consistency and learned-coverage smoke test, no target PDF."""
import json,torch
from coverage_pointer import CoveragePointer
from extractive_tasks import ExtractiveTasks
from query_pointer import batch
from resumable_checkpoint import Checkpoints,atomic_json

@torch.no_grad()
def main():
    torch.set_num_threads(4); m=CoveragePointer().eval()
    saved,info=Checkpoints('outputs/coverage-v8').load(); m.load_state_dict(saved['model'])
    t=ExtractiveTasks(); x,s,sm,q,qm,_,_=batch(t,[t.example(861010000+i,'dev') for i in range(2)])
    x=x[:,:16]; enc=m.encode_source(s,sm); drive=m.encode_query(q,qm)
    p,cache,att,_=m(x,s,sm,encoded=enc,query_drive=drive)
    cc=None; pieces=[]
    for k in range(x.shape[1]):
        pp,cc,_,_=m(x[:,k:k+1],s,sm,cc,enc,query_drive=drive); pieces.append(pp)
    result={'checkpoint':info,'tokens_per_example':16,'examples':2,
            'probability_sum_error':float((p.exp().sum(-1)-1).abs().max()),
            'coverage_mass_error':float((cache[3].sum(-1)-16).abs().max()),
            'incremental_probability_error':float((p.exp()-torch.cat(pieces,1).exp()).abs().max()),
            'learned_coverage_input_norm':float(m.coverage_input.weight.norm()),
            'revisit_penalty':float(torch.nn.functional.softplus(m.coverage_penalty)),
            'target_pdf_used':False}
    assert result['probability_sum_error']<1e-5
    assert result['coverage_mass_error']<1e-4
    assert result['incremental_probability_error']<.002
    assert result['learned_coverage_input_norm']>0
    atomic_json('outputs/coverage-v8/numerical-check.json',result); print(json.dumps(result))
if __name__=='__main__': main()

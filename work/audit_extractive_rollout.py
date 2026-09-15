"""Development-only rollout diagnosis; never opens the target PDF."""
import json,time
from pathlib import Path
import torch
from extractive_pointer import ExtractivePointer
from extractive_tasks import ExtractiveTasks
from query_pointer import batch
from resumable_checkpoint import Checkpoints,atomic_json

@torch.no_grad()
def main():
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    m=ExtractivePointer().eval(); saved,info=Checkpoints('outputs/extractive-v5/best').load()
    m.load_state_dict(saved['model']); tasks=ExtractiveTasks(); rows=[]; consistency=[]
    examples=[tasks.example(830000000+i,'dev') for i in range(32)]
    for j,e in enumerate(examples):
        x,s,sm,q,qm,y,align=batch(tasks,[e]); enc=m.encode_source(s,sm); drive=m.encode_query(q,qm)
        if j<4:
            full=m(x,s,sm,encoded=enc,query_drive=drive)[0]
            cache=None; pieces=[]
            for k in range(x.shape[1]):
                p,cache,_,_=m(x[:,k:k+1],s,sm,cache,enc,query_drive=drive); pieces.append(p)
            incremental=torch.cat(pieces,1)
            consistency.append({'seed':e['seed'],'max_probability_difference':float((full.exp()-incremental.exp()).abs().max()),
                'argmax_disagreements':int((full.argmax(-1)!=incremental.argmax(-1)).sum()),'tokens':x.shape[1]})
        for mode in ['soft','hard-position-feedback']:
            cache=None; token=torch.ones(1,1,dtype=torch.long,device='cuda'); ids=[]; positions=[]
            for k in range(384):
                p,cache,a,g=m(token,s,sm,cache,enc,query_drive=drive)
                t=int(p[0,-1].argmax()); ids.append(t); positions.append(int(a[0,-1].argmax()))
                if t==2: break
                if mode=='hard-position-feedback':
                    # No forced words or gold positions: only sharpen the model's own memory.
                    state,prev,attention=cache
                    hard=torch.zeros_like(attention).scatter_(1,attention.argmax(-1,keepdim=True),1.)
                    cache=state,prev,hard
                token=torch.tensor([[t]],device='cuda')
            raw=tasks.tok.decode(ids)
            rows.append({'seed':e['seed'],'paper':e['queried_name'],'mode':mode,'source':e['source'],
                         'answer':e['answer'],'raw_output':raw,'generated_ids':ids,'positions':positions,
                         'exact':raw.strip()==e['answer'].strip(),'ended_eos':ids[-1]==2})
    out=Path('outputs/extractive-rollout-audit'); out.mkdir(exist_ok=True)
    atomic_json(out/'raw.json',rows)
    summary={'checkpoint':info,'split':'dev','target_pdf_used':False,'consistency':consistency,
             'results':{mode:{'exact':sum(r['exact'] for r in rows if r['mode']==mode)/32,
                              'ended_eos':sum(r['ended_eos'] for r in rows if r['mode']==mode)/32}
                        for mode in ['soft','hard-position-feedback']}}
    atomic_json(out/'summary.json',summary); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()

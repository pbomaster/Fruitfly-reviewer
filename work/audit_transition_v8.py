"""Check whether teacher-forced errors concentrate at the sentence transition."""
import json
import torch
from coverage_pointer import CoveragePointer
from extractive_tasks import ExtractiveTasks
from query_pointer import batch
from resumable_checkpoint import Checkpoints,atomic_json

@torch.no_grad()
def main():
    torch.set_num_threads(4); m=CoveragePointer().eval()
    saved,info=Checkpoints('outputs/coverage-v8/best').load(); m.load_state_dict(saved['model'])
    tasks=ExtractiveTasks(); rows=[]
    for i in range(32):
        e=tasks.example(850000000+i,'dev'); x,s,sm,q,qm,y,a=batch(tasks,[e])
        first=len(tasks.tok.encode(e['contribution']+' ').ids)
        # Gold first sentence is supplied only for diagnosis; this is not generation.
        drive=m.encode_query(q,qm)
        p,_,att,g=m(x[:,:first+1],s,sm,query_drive=drive)
        probs=p[0,-1].exp(); gold=int(y[0,first]); predicted=int(probs.argmax())
        rows.append({'seed':e['seed'],'paper':e['queried_name'],'first_sentence':e['contribution'],
                     'limitation':e['limitation'],'transition_index':first,'gold_id':gold,'predicted_id':predicted,
                     'p_gold':float(probs[gold]),'p_eos':float(probs[2]),'generation_gate':float(g[0,-1,0]),
                     'max_source_attention':float(att[0,-1].max()),'correct':gold==predicted,'eos_at_transition':predicted==2})
    summary={'checkpoint':info,'split':'dev','target_pdf_read':False,'n':len(rows),
             'teacher_transition_accuracy':sum(r['correct'] for r in rows)/len(rows),
             'teacher_premature_eos_fraction':sum(r['eos_at_transition'] for r in rows)/len(rows),
             'scope':'Teacher-forced diagnostic only; no generated review or success claim.'}
    atomic_json('outputs/coverage-v8/transition-audit-raw.json',rows)
    atomic_json('outputs/coverage-v8/transition-audit.json',summary); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()

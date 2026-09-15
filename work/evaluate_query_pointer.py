import json,copy
from pathlib import Path
import torch
from query_pointer import QueryPointer,predict
from science_pointer_tasks import ScienceTasks
from resumable_checkpoint import Checkpoints,atomic_json

torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
out=Path('outputs/query-pointer-v4'); tasks=ScienceTasks(); m=QueryPointer()
saved,info=Checkpoints(out/'best').load(); m.load_state_dict(saved['model'])
original=[tasks.example(940000000+i,split='test') for i in range(200)]
changed=[]
for e in original:
    r=copy.deepcopy(e); offsets=tasks.tok.encode(e['source']).offsets; a=offsets[e['align'][0]][0]
    while e['source'][a].isspace(): a+=1
    assert e['source'][a:a+len(e['answer'])]==e['answer']
    new=e['answer'][:-1]+str((int(e['answer'][-1])+1)%10)
    r['source']=e['source'][:a]+new+e['source'][a+len(e['answer']):]
    r['source_ids']=tasks.tok.encode(r['source']).ids; r['answer']=new; r['changed']=True
    changed.append(r)
allrows={}
for name,es,scale,zero in [('original',original,1.,False),('same-format-change',changed,1.,False),('graph-removed',original,0.,False),('query-removed',original,1.,True)]:
    rows=predict(m,tasks,es,graph_scale=scale,zero_query=zero); allrows[name]=rows
    atomic_json(out/f'{name}-raw.json',rows)
result={'checkpoint':info,'n':200,'test_seed':940000000,'accuracy':{k:sum(r['exact'] for r in rows)/len(rows) for k,rows in allrows.items()},
        'paired_both_correct':sum(a['exact'] and b['exact'] for a,b in zip(allrows['original'],allrows['same-format-change']))/200,
        'identical_changed_output':sum(a['raw_output']==b['raw_output'] for a,b in zip(allrows['original'],allrows['same-format-change']))/200,
        'paper_review_success':False}
atomic_json(out/'evaluation.json',result); print(json.dumps(result,indent=2))

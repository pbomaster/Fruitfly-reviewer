"""Additional control: change the last digit, preserving length/decimal format.

Added to distinguish the +137 stress test's number-distribution shift from
failure to follow a small source edit. No training or checkpoint selection.
"""
import copy,json
from pathlib import Path
import torch
from graph_pointer import GraphPointer
from pointer_tasks import predict
from science_pointer_tasks import ScienceTasks
from resumable_checkpoint import Checkpoints,atomic_json

out=Path('outputs/science-pointer-v3'); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
tasks=ScienceTasks(); saved,info=Checkpoints(out/'best').load(); m=GraphPointer(); m.load_state_dict(saved['model'])
examples=[]
for i in range(200):
    e=tasks.example(930000000+i,split='test'); r=copy.deepcopy(e)
    offsets=tasks.tok.encode(e['source']).offsets; a=offsets[e['align'][0]][0]
    while e['source'][a].isspace(): a+=1
    assert e['source'][a:a+len(e['answer'])]==e['answer']
    new=e['answer'][:-1]+str((int(e['answer'][-1])+1)%10)
    r['source']=e['source'][:a]+new+e['source'][a+len(e['answer']):]
    r['source_ids']=tasks.tok.encode(r['source']).ids; r['answer']=new; r['changed']=True
    examples.append(r)
rows=predict(m,tasks,examples); atomic_json(out/'same-format-changed-raw.json',rows)
original=json.loads((out/'original-raw.json').read_text(encoding='utf8'))
r={'checkpoint':info,'count':len(rows),'exact_accuracy':sum(x['exact'] for x in rows)/len(rows),
   'paired_original_and_changed_exact':sum(a['exact'] and b['exact'] for a,b in zip(original,rows))/len(rows),
   'identical_output_rate':sum(a['raw_output']==b['raw_output'] for a,b in zip(original,rows))/len(rows),
   'design':'Last digit +1 modulo 10, preserving number width and decimal places. Added diagnostic; not used for model selection.'}
atomic_json(out/'same-format-control.json',r); print(json.dumps(r,indent=2))

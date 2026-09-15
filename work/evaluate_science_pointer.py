"""Final held-out scientific passage evaluation; generated outputs unedited."""
import argparse,copy,json,time
from pathlib import Path
import torch
from graph_pointer import GraphPointer
from science_pointer_tasks import ScienceTasks
from pointer_tasks import predict
from resumable_checkpoint import Checkpoints,atomic_json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',default='outputs/science-pointer-v3'); p.add_argument('--count',type=int,default=200)
    args=p.parse_args(); out=Path(args.run)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    tasks=ScienceTasks(); saved,info=Checkpoints(out/'best').load(); model=GraphPointer(); model.load_state_dict(saved['model'])
    base=[tasks.example(930000000+i,split='test') for i in range(args.count)]
    changed=[]
    for e in base:
        r=copy.deepcopy(e); enc=tasks.tok.encode(e['source']); a=enc.offsets[e['align'][0]][0]
        while e['source'][a].isspace(): a+=1
        assert e['source'][a:a+len(e['answer'])]==e['answer']
        new=str(int(float(e['answer']))+137)
        r['source']=e['source'][:a]+new+e['source'][a+len(e['answer']):]
        r['answer']=new; r['source_ids']=tasks.tok.encode(r['source']).ids; r['changed']=True
        changed.append(r)
    report={'checkpoint':info,'selected_step':saved['step'],'dev_exact':saved['dev_exact'],'tests':{},
            'review_success':False,'scope':'Human-written scientific numeric cloze. Not free-form reviewing or independent scientific reasoning.'}
    predictions={}; start=time.monotonic()
    for name,examples,scale in [('original',base,1.),('changed_value',changed,1.),('graph_removed',base,0.)]:
        rows=predict(model,tasks,examples,graph_scale=scale); predictions[name]=rows
        atomic_json(out/(name+'-raw.json'),rows)
        score={'exact_accuracy':sum(x['exact'] for x in rows)/len(rows),'count':len(rows),'distinct_papers':len({x['queried_name'] for x in rows})}
        report['tests'][name]=score; print(name,json.dumps(score),flush=True)
    report['paired_original_and_changed_exact']=sum(a['exact'] and b['exact'] for a,b in zip(predictions['original'],predictions['changed_value']))/args.count
    report['changed_input_identical_output_rate']=sum(a['raw_output']==b['raw_output'] for a,b in zip(predictions['original'],predictions['changed_value']))/args.count
    report['seconds']=time.monotonic()-start
    report['scientific_cloze_gate_passed']=all(report['tests'][k]['exact_accuracy']>=.95 for k in ['original','changed_value'])
    atomic_json(out/'evaluation.json',report)

if __name__=='__main__': main()

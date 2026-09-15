"""Untouched tests and raw counterfactual generation for the short-input pilot."""
import argparse,copy,json,re,time
from pathlib import Path
import torch
from graph_pointer import GraphPointer
from pointer_tasks import Tasks,predict
from resumable_checkpoint import Checkpoints,atomic_json,sha256

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',default='outputs/graph-pointer-v2'); p.add_argument('--count',type=int,default=200)
    args=p.parse_args(); out=Path(args.run); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    saved,info=Checkpoints(out/'best').load(); tasks=Tasks(); model=GraphPointer(); model.load_state_dict(saved['model']); model.eval()
    base=[tasks.example(900000000+i) for i in range(args.count)]
    changed=[tasks.example(900000000+i,changed=True) for i in range(args.count)]
    question=[]; reordered=[]
    for e in base:
        records=re.findall(r'Sample (\w+) contains (\d+) cells\.',e['source'])
        name,number=next((n,v) for n,v in records if n!=e['queried_name'])
        q=copy.deepcopy(e); q['query']=e['query'].replace(e['queried_name'],name)
        q['prefix']=[1]+tasks.tok.encode(q['query']).ids; q['answer']=number; q['queried_name']=name; question.append(q)
        r=copy.deepcopy(e); r['source']=' '.join(f'Sample {n} contains {v} cells.' for n,v in reversed(records))
        r['source_ids']=tasks.tok.encode(r['source']).ids; reordered.append(r)
    cases=[('unseen_documents',base,1.),('changed_number',changed,1.),('changed_question',question,1.),
           ('reversed_records',reordered,1.),('eight_records',[tasks.example(910000000+i,records=8) for i in range(args.count)],1.),
           ('new_wording',[tasks.example(920000000+i,style=1) for i in range(args.count)],1.),('graph_removed',base,0.)]
    report={'checkpoint':info,'selected_step':saved['step'],'dev_exact':saved['dev_exact'],
            'test_count_each':args.count,'tests':{},'review_success':False,'scope':'Controlled factual copy only, not scientific comprehension.'}
    all_results={}; start=time.monotonic()
    for name,examples,scale in cases:
        predictions=predict(model,tasks,examples,graph_scale=scale)
        all_results[name]=predictions; atomic_json(out/(name+'-raw.json'),predictions)
        metrics={'exact_accuracy':sum(x['exact'] for x in predictions)/len(predictions),'count':len(predictions)}
        report['tests'][name]=metrics; print(name,json.dumps(metrics),flush=True)
    report['paired_original_and_changed_exact']=sum(a['exact'] and b['exact'] for a,b in zip(all_results['unseen_documents'],all_results['changed_number']))/args.count
    report['changed_input_identical_output_rate']=sum(a['raw_output']==b['raw_output'] for a,b in zip(all_results['unseen_documents'],all_results['changed_number']))/args.count
    report['basic_gate_passed']=report['tests']['unseen_documents']['exact_accuracy']>=.95 and report['tests']['changed_number']['exact_accuracy']>=.95
    report['seconds']=time.monotonic()-start
    atomic_json(out/'evaluation.json',report); print(json.dumps(report,indent=2))

if __name__=='__main__': main()

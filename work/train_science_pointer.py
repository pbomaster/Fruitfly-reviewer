"""Transfer the project's own graph copy weights to scientific passage matching.

Checkpoints use the same tested two-slot protocol; no pretrained LLM is loaded.
"""
import argparse,json,os,random,time
from pathlib import Path
import torch
from graph_pointer import GraphPointer
from pointer_tasks import predict,TOKENIZER
from science_pointer_tasks import ScienceTasks,PATH
from train_pointer import loss_step,payload
from resumable_checkpoint import Checkpoints,atomic_json,restore_rng,sha256

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='outputs/science-pointer-v3')
    p.add_argument('--steps',type=int,default=2000); p.add_argument('--batch',type=int,default=16)
    p.add_argument('--resume',action='store_true'); p.add_argument('--max-session-minutes',type=float,default=240)
    args=p.parse_args(); out=Path(args.out); out.mkdir(exist_ok=True)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True; torch.manual_seed(73117); random.seed(73117)
    model=GraphPointer(); initial,initial_info=Checkpoints('outputs/graph-pointer-v2/best').load()
    model.load_state_dict(initial['model']); opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.01)
    config={'version':3,'batch':args.batch,'lr':.0002,'weight_decay':.01,'initial_checkpoint_sha256':initial_info['sha256'],
            'corpus_sha256':sha256(PATH),'tokenizer_sha256':sha256(TOKENIZER),
            'code_sha256':{f:sha256('work/'+f) for f in ['train_science_pointer.py','science_pointer_tasks.py','train_pointer.py','graph_pointer.py','pointer_tasks.py','resumable_checkpoint.py','connectome_memory.py']}}
    cp=Checkpoints(out); tasks=ScienceTasks(); step=0; best=-1.; history=[]; elapsed_before=0
    if args.resume:
        saved,info=cp.load(); assert saved['config']==config,'Code/data/config mismatch'
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); restore_rng(saved['rng'])
        step=saved['step']; best=saved['best_dev']; history=saved['history']; elapsed_before=saved['elapsed_seconds']
    elif cp.valid_entries(): raise RuntimeError('Use --resume or a new run folder')
    start=time.monotonic(); last_save=start
    def save(state):
        nonlocal last_save
        elapsed=elapsed_before+time.monotonic()-start
        info=cp.save(payload(model,opt,step,config,best,history,elapsed)); last_save=time.monotonic()
        atomic_json(out/'status.json',{'state':state,'pid':os.getpid(),'step':step,'planned_steps':args.steps,
                    'elapsed_seconds':elapsed,'last_checkpoint':info,'resume_supported':True,'safe_to_close_process':state!='training'})
        print('checkpoint',json.dumps(info),flush=True)
    validation=[tasks.example(810000000+i,split='dev') for i in range(64)]
    save('training')
    if step==0:
        pred=predict(model,tasks,validation); atomic_json(out/'initial-validation.json',pred)
        print('initial_dev_exact',sum(x['exact'] for x in pred)/len(pred),flush=True)
    log=(out/'training.jsonl').open('a',encoding='utf8',buffering=1)
    try:
        while step<args.steps:
            if (out/'PAUSE.request').exists() or time.monotonic()-start>=args.max_session_minutes*60:
                save('paused'); return
            metric=loss_step(model,opt,tasks,step,args.batch); step+=1
            if step%25==0:
                row={'step':step,**metric,'elapsed_seconds':elapsed_before+time.monotonic()-start}
                log.write(json.dumps(row)+'\n'); print(json.dumps(row),flush=True)
            if step%250==0 or step==args.steps:
                pred=predict(model,tasks,validation); acc=sum(x['exact'] for x in pred)/len(pred)
                history.append({'step':step,'dev_exact':acc}); atomic_json(out/f'dev-{step}.json',pred)
                print('validation',json.dumps(history[-1]),flush=True)
                if acc>best:
                    best=acc; Checkpoints(out/'best').save({'model':model.state_dict(),'step':step,'config':config,'dev_exact':acc})
                save('training')
            elif time.monotonic()-last_save>=60: save('training')
        save('completed')
        atomic_json(out/'training-summary.json',{'step':step,'best_dev':best,'history':history,'config':config,
                     'elapsed_seconds':elapsed_before+time.monotonic()-start,'review_success':False})
    except KeyboardInterrupt:
        atomic_json(out/'status.json',{'state':'interrupted_use_last_checkpoint','step':step,
                       'safe_to_close_process':True,'resume_supported':True})
    finally: log.close()

if __name__=='__main__': main()

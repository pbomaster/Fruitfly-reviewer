import argparse,json,os,random,time,datetime
from pathlib import Path
import torch
from torch.nn import functional as F
from query_pointer import QueryPointer,batch,predict
from science_pointer_tasks import ScienceTasks,PATH
from pointer_tasks import TOKENIZER
from train_pointer import payload
from resumable_checkpoint import Checkpoints,atomic_json,restore_rng,sha256

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='outputs/query-pointer-v4')
    p.add_argument('--steps',type=int,default=3000); p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-step',type=int); args=p.parse_args()
    out=Path(args.out); out.mkdir(exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(99371); random.seed(99371)
    torch.backends.cuda.matmul.allow_tf32=True
    tasks=ScienceTasks(); m=QueryPointer(); initial,ii=Checkpoints('outputs/science-pointer-v3/best').load()
    missing,unexpected=m.load_state_dict(initial['model'],strict=False)
    assert set(missing)=={'query_pool.weight','query_pool.bias','query_sensory.weight'} and not unexpected
    opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01)
    config={'version':4,'batch':32,'seed':99371,'lr':.0003,'init':ii,'corpus_sha256':sha256(PATH),
      'code':{f:sha256('work/'+f) for f in ['train_query_pointer.py','query_pointer.py','science_pointer_tasks.py','graph_pointer.py','connectome_memory.py','resumable_checkpoint.py']},
      'tokenizer_sha256':sha256(TOKENIZER),'target_pdf_used':False}
    cp=Checkpoints(out); step=0; best=-1.; history=[]; elapsed=0.
    if args.resume:
        saved,info=cp.load(); assert config==saved['config']
        m.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); restore_rng(saved['rng'])
        step=saved['step']; best=saved['best_dev']; history=saved['history']; elapsed=saved['elapsed_seconds']
    elif cp.valid_entries(): raise RuntimeError('Run exists')
    start=time.monotonic(); last_save=start
    def save(state):
        nonlocal last_save
        info=cp.save(payload(m,opt,step,config,best,history,elapsed+time.monotonic()-start))
        last_save=time.monotonic()
        atomic_json(out/'status.json',{'state':state,'pid':os.getpid(),'step':step,'planned_steps':args.steps,'last_checkpoint':info,
                                      'safe_to_close_process':state!='training','resume_supported':True})
    validation=[tasks.example(810000000+i,split='dev') for i in range(64)]
    save('training')
    with (out/'training.jsonl').open('a',encoding='utf8',buffering=1) as log:
        while step<args.steps:
            now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
            daytime=(9,25)<=(now.hour,now.minute)<(19,30)
            if daytime or time.monotonic()-start>4*3600 or (out/'PAUSE.request').exists() or (args.stop_after_step and step>=args.stop_after_step):
                save('paused'); return
            m.train(); opt.zero_grad(set_to_none=True)
            x,s,sm,q,qm,y,a=batch(tasks,tasks.training_batch(step,32,seed=672117))
            drive=m.encode_query(q,qm); probs,_,att,_=m(x,s,sm,query_drive=drive)
            lm=F.nll_loss(probs.flatten(0,1),y.flatten())
            al=F.nll_loss(att.clamp_min(1e-9).log().flatten(0,1),a.flatten())
            loss=lm+.3*al
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite')
            loss.backward(); grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            if not torch.isfinite(grad): raise RuntimeError('Nonfinite gradient')
            opt.step(); step+=1
            if step%50==0:
                row={'step':step,'loss':loss.item(),'lm':lm.item(),'align':al.item(),'elapsed':elapsed+time.monotonic()-start}
                log.write(json.dumps(row)+'\n'); print(json.dumps(row),flush=True)
            if step%250==0 or step==args.steps:
                rows=predict(m,tasks,validation); acc=sum(r['exact'] for r in rows)/len(rows)
                atomic_json(out/f'dev-{step}.json',rows); history.append({'step':step,'dev_exact':acc})
                print(json.dumps(history[-1]),flush=True)
                if acc>best:
                    best=acc; Checkpoints(out/'best').save({'model':m.state_dict(),'step':step,'config':config,'dev_exact':acc})
                save('training')
            elif time.monotonic()-last_save>60: save('training')
    save('completed')
    atomic_json(out/'training-summary.json',{'step':step,'best_dev':best,'history':history,'elapsed_seconds':elapsed+time.monotonic()-start,'config':config,'review_success':False})

if __name__=='__main__': main()

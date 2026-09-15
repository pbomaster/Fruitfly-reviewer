import argparse,json,os,random,time,datetime
from pathlib import Path
import torch
from torch.nn import functional as F
from sequential_pointer import SequentialPointer
from extractive_tasks import ExtractiveTasks,PATH
from query_pointer import batch
from train_pointer import payload
from resumable_checkpoint import Checkpoints,atomic_json,restore_rng,sha256

def loss_batch(m,tasks,examples,opt=None):
    train=opt is not None; m.train(train)
    if train: opt.zero_grad(set_to_none=True)
    x,s,sm,q,qm,y,a=batch(tasks,examples); cache=None
    total=(y!=-100).sum().item(); lm_total=0.; al_total=0.
    for start in range(0,x.shape[1],48):
        yy=y[:,start:start+48]; aa=a[:,start:start+48]; n=int((yy!=-100).sum())
        if not n: break
        drive=m.encode_query(q,qm)
        prob,cache,att,_=m(x[:,start:start+48],s,sm,cache,query_drive=drive)
        lm=F.nll_loss(prob.flatten(0,1),yy.flatten())
        al=F.nll_loss(att.clamp_min(1e-9).log().flatten(0,1),aa.flatten()) if (aa!=-100).any() else lm*0
        loss=(lm+.5*al)*(n/total)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        if train: loss.backward()
        cache=tuple(t.detach() for t in cache)
        lm_total+=lm.item()*n/total; al_total+=al.item()*n/total
    if train:
        grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        if not torch.isfinite(grad): raise RuntimeError('Nonfinite gradient')
        opt.step()
    return {'token_nll':lm_total,'alignment_nll':al_total,'tokens':total}

@torch.no_grad()
def generate(m,tasks,e,graph_scale=1.,max_tokens=384):
    m.eval(); s=torch.tensor([e['source_ids']],device='cuda'); sm=s!=0
    q=torch.tensor([e['prefix']],device='cuda'); drive=m.encode_query(q,q!=0); enc=m.encode_source(s,sm)
    token=torch.ones(1,1,dtype=torch.long,device='cuda'); cache=None; ids=[]; positions=[]; gates=[]
    for _ in range(max_tokens):
        p,cache,a,g=m(token,s,sm,cache,enc,graph_scale,drive)
        t=int(p[0,-1].argmax()); ids.append(t); positions.append(int(a[0,-1].argmax())); gates.append(float(g[0,-1,0]))
        if t==2: break
        token=torch.tensor([[t]],device='cuda')
    raw=tasks.tok.decode(ids)
    return {**{k:e[k] for k in ['source','query','answer','seed','queried_name'] if k in e},
            'generated_ids':ids,'raw_output':raw,'source_positions':positions,'generation_gate':gates,
            'exact':raw.strip()==e.get('answer','').strip(),'ended_eos':ids[-1]==2,'graph_scale':graph_scale}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='outputs/sequential-v6')
    p.add_argument('--steps',type=int,default=1000); p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-step',type=int); args=p.parse_args()
    out=Path(args.out); out.mkdir(exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(55123); random.seed(55123); torch.backends.cuda.matmul.allow_tf32=True
    tasks=ExtractiveTasks(); m=SequentialPointer(); initial,ii=Checkpoints('outputs/extractive-v5/best').load()
    missing,unexpected=m.load_state_dict(initial['model'],strict=False)
    assert set(missing)=={'advance_gate.weight','advance_gate.bias'} and not unexpected
    opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01)
    config={'version':6,'batch':8,'lr':.0003,'seed':55123,'init':ii,'corpus_sha256':sha256(PATH),
            'code':{f:sha256('work/'+f) for f in ['train_sequential.py','sequential_pointer.py','extractive_tasks.py','extractive_pointer.py','query_pointer.py','graph_pointer.py','connectome_memory.py','resumable_checkpoint.py']},
            'scope':'Learned source advance/retrieval with two graph propagation steps. Weakly supervised extractive contribution-plus-limitation paragraph; not expert review labels.','target_pdf_used':False}
    cp=Checkpoints(out); step=0; best=1e9; history=[]; elapsed=0.
    if args.resume:
        saved,info=cp.load(); assert config==saved['config']
        m.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); restore_rng(saved['rng'])
        step=saved['step']; best=saved['best_dev']; history=saved['history']; elapsed=saved['elapsed_seconds']
        print('resumed',json.dumps(info),flush=True)
    elif cp.valid_entries(): raise RuntimeError('Run exists')
    start=time.monotonic(); last_save=start
    def save(state):
        nonlocal last_save
        info=cp.save(payload(m,opt,step,config,best,history,elapsed+time.monotonic()-start)); last_save=time.monotonic()
        atomic_json(out/'status.json',{'state':state,'pid':os.getpid(),'step':step,'planned_steps':args.steps,
                                     'last_checkpoint':info,'safe_to_close_process':state!='training','resume_supported':True})
    validation=[tasks.example(820000000+i,'dev') for i in range(24)]; save('training')
    with (out/'training.jsonl').open('a',encoding='utf8',buffering=1) as log:
        while step<args.steps:
            now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
            if (9,25)<=(now.hour,now.minute)<(19,30) or time.monotonic()-start>4*3600 or (out/'PAUSE.request').exists() or (args.stop_after_step and step>=args.stop_after_step):
                save('paused'); return
            metric=loss_batch(m,tasks,tasks.training_batch(step,8),opt); step+=1
            if step%25==0:
                row={'step':step,**metric,'elapsed':elapsed+time.monotonic()-start}
                log.write(json.dumps(row)+'\n'); print(json.dumps(row),flush=True)
            if step%250==0 or step==args.steps:
                with torch.no_grad(): scores=[loss_batch(m,tasks,validation[j:j+8]) for j in range(0,len(validation),8)]
                n=sum(r['tokens'] for r in scores); dev=sum(r['token_nll']*r['tokens'] for r in scores)/n
                rows=[generate(m,tasks,e) for e in validation[:4]]; atomic_json(out/f'dev-{step}-raw.json',rows)
                history.append({'step':step,'dev_token_nll':dev,'sample_exact':sum(r['exact'] for r in rows)/len(rows)})
                print(json.dumps(history[-1]),flush=True)
                if dev<best:
                    best=dev; Checkpoints(out/'best').save({'model':m.state_dict(),'step':step,'config':config,'dev_token_nll':dev})
                save('training')
            elif time.monotonic()-last_save>60: save('training')
    save('completed'); atomic_json(out/'training-summary.json',{'step':step,'best_dev_nll':best,'history':history,
      'elapsed_seconds':elapsed+time.monotonic()-start,'config':config,'review_success':False})

if __name__=='__main__': main()

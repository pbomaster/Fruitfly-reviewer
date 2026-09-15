"""Resumable short-document factual-copy pilot. No target-paper training."""
import argparse,json,os,random,time
from pathlib import Path
import torch
from torch.nn import functional as F
from graph_pointer import GraphPointer
from pointer_tasks import Tasks,predict,TOKENIZER
from resumable_checkpoint import Checkpoints,atomic_json,rng_state,restore_rng,sha256

CODE=['work/train_pointer.py','work/graph_pointer.py','work/pointer_tasks.py','work/connectome_memory.py','work/resumable_checkpoint.py']

def loss_step(model,opt,tasks,step,batch):
    model.train(); opt.zero_grad(set_to_none=True)
    tensors=tasks.batch(tasks.training_batch(step,batch))
    tokens,source,mask,y,alignment=tensors
    logp,_,attention,_=model(tokens,source,mask)
    lm=F.nll_loss(logp.flatten(0,1),y.flatten())
    alignment_loss=F.nll_loss(attention.clamp_min(1e-9).log().flatten(0,1),alignment.flatten())
    loss=lm+.3*alignment_loss
    if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
    loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(grad): raise RuntimeError('Non-finite gradients')
    opt.step()
    return {'loss':loss.item(),'token_nll':lm.item(),'pointer_nll':alignment_loss.item(),'gradient_norm':grad.item()}

def payload(model,opt,step,config,best,history,elapsed):
    return {'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'rng':rng_state(),
            'config':config,'best_dev':best,'history':history,'elapsed_seconds':elapsed}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='outputs/graph-pointer-v2')
    p.add_argument('--steps',type=int,default=3000); p.add_argument('--batch',type=int,default=32)
    p.add_argument('--eval-every',type=int,default=250); p.add_argument('--save-seconds',type=int,default=60)
    p.add_argument('--max-session-minutes',type=float,default=240)
    p.add_argument('--resume',action='store_true'); p.add_argument('--stop-after-step',type=int)
    args=p.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    status_file=out/'status.json'; stopfile=out/'PAUSE.request'
    config={'version':2,'batch':args.batch,'seed':66119,'lr':.0005,'weight_decay':.01,
            'tokenizer_sha256':sha256(TOKENIZER),'graph_sha256':sha256('work/fly-model/model.safetensors'),
            'code_sha256':{f:sha256(f) for f in CODE},'training_seed':44117,
            'validation_seed':800000000,'test_seed':900000000,'alignment_weight':.3,
            'task':'Synthetic queried sample -> random four-digit cell count amid four records',
            'scope':'A factual-reading pilot, not a paper review model or success claim.'}
    torch.set_num_threads(4); random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32=True
    model=GraphPointer(); opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    tasks=Tasks(); checkpoints=Checkpoints(out); step=0; best=-1.; history=[]; elapsed_before=0.
    if args.resume:
        saved,info=checkpoints.load()
        if saved['config']!=config: raise RuntimeError('Configuration/code/data changed; refusing silent resume.')
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng']); step=saved['step']; best=saved['best_dev']; history=saved['history']; elapsed_before=saved['elapsed_seconds']
        print('resumed',json.dumps(info),flush=True)
    elif checkpoints.valid_entries(): raise RuntimeError('Existing run found: use --resume or choose another output directory.')
    start=time.monotonic(); last_save=start; last_info=None
    def checkpoint(state):
        nonlocal last_save,last_info
        elapsed=elapsed_before+time.monotonic()-start
        last_info=checkpoints.save(payload(model,opt,step,config,best,history,elapsed)); last_save=time.monotonic()
        atomic_json(status_file,{'state':state,'pid':os.getpid(),'step':step,'planned_steps':args.steps,
                                'elapsed_seconds':elapsed,'last_checkpoint':last_info,
                                'safe_to_close_process':state in ['paused','completed','stopped_by_limit'],
                                'resume_supported':True})
        print('checkpoint',json.dumps(last_info),flush=True)
    checkpoint('training')
    validation=[tasks.example(800000000+i) for i in range(64)]
    log=(out/'training.jsonl').open('a',encoding='utf8',buffering=1)
    try:
        if step==0:
            initial=predict(model,tasks,validation)
            atomic_json(out/'initial-validation.json',initial)
            print('initial_accuracy',sum(x['exact'] for x in initial)/len(initial),flush=True)
        while step<args.steps:
            reason=None
            if stopfile.exists(): reason='paused'
            if args.stop_after_step is not None and step>=args.stop_after_step: reason='paused'
            if time.monotonic()-start>=args.max_session_minutes*60: reason='stopped_by_limit'
            if reason:
                checkpoint(reason); return
            metrics=loss_step(model,opt,tasks,step,args.batch); step+=1
            if step%25==0:
                row={'step':step,**metrics,'elapsed_seconds':elapsed_before+time.monotonic()-start}
                log.write(json.dumps(row)+'\n'); print(json.dumps(row),flush=True)
                atomic_json(status_file,{'state':'training','pid':os.getpid(),'step':step,'planned_steps':args.steps,
                    'elapsed_seconds':row['elapsed_seconds'],'last_checkpoint':last_info,'safe_to_close_process':False,'resume_supported':True})
            if step%args.eval_every==0 or step==args.steps:
                predictions=predict(model,tasks,validation); acc=sum(x['exact'] for x in predictions)/len(predictions)
                row={'step':step,'dev_exact':acc}; history.append(row)
                atomic_json(out/f'dev-{step}.json',predictions)
                print('validation',json.dumps(row),flush=True)
                if acc>best:
                    best=acc
                    Checkpoints(out/'best').save({'model':model.state_dict(),'step':step,'config':config,'dev_exact':acc})
                checkpoint('training')
            elif time.monotonic()-last_save>=args.save_seconds: checkpoint('training')
        checkpoint('completed')
        atomic_json(out/'training-summary.json',{'step':step,'best_dev':best,'history':history,
            'elapsed_seconds':elapsed_before+time.monotonic()-start,'config':config,'review_success':False})
    except KeyboardInterrupt:
        # A batch interrupted during opt.step may be incomplete. Roll back to the
        # last fully completed persisted boundary instead of saving partial state.
        atomic_json(status_file,{'state':'interrupted_use_last_checkpoint','step':step,
            'last_checkpoint':last_info,'safe_to_close_process':True,'resume_supported':True})
        print('Interrupted. Resume from the last complete checkpoint; current partial batch discarded.',flush=True)
    finally: log.close()

if __name__=='__main__': main()

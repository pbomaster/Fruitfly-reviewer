"""Real graph/optimizer continuation plus corrupt/incomplete-file recovery tests."""
import json,random,shutil,time
from pathlib import Path
import torch
from graph_pointer import GraphPointer
from pointer_tasks import Tasks
from train_pointer import loss_step,payload
from resumable_checkpoint import Checkpoints,restore_rng,atomic_json

def main():
    root=Path('outputs/pointer-resume-test'); root.mkdir(exist_ok=True)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    random.seed(611); torch.manual_seed(611)
    tasks=Tasks(); a=GraphPointer(); opt=torch.optim.AdamW(a.parameters(),lr=.0005)
    cp=Checkpoints(root/'actual-graph'); start=time.monotonic()
    first=[loss_step(a,opt,tasks,i,4) for i in range(2)]
    cp.save(payload(a,opt,2,{'test':True},0.,[],0.))
    expected=[loss_step(a,opt,tasks,i,4) for i in range(2,4)]
    expected_params={n:p.detach().cpu().clone() for n,p in a.named_parameters()}
    del a,opt; torch.cuda.empty_cache()
    b=GraphPointer(); optb=torch.optim.AdamW(b.parameters(),lr=.0005)
    saved,info=cp.load(); b.load_state_dict(saved['model']); optb.load_state_dict(saved['optimizer']); restore_rng(saved['rng'])
    actual=[loss_step(b,optb,tasks,i,4) for i in range(saved['step'],4)]
    errors={n:(p.detach().cpu()-expected_params[n]).abs().max().item() for n,p in b.named_parameters()}
    max_error=max(errors.values()); loss_error=max(abs(x['loss']-y['loss']) for x,y in zip(expected,actual))
    assert max_error<2e-5 and loss_error<2e-5,(max_error,loss_error)
    small=Checkpoints(root/'recovery')
    small.save({'step':1,'value':torch.tensor([1.])})
    small.save({'step':2,'value':torch.tensor([2.])})
    newest=small.valid_entries()[0]
    # Simulate a partially written temporary file, then a damaged newest slot.
    (small.root/'checkpoint-a.pt.tmp').write_bytes(b'interrupted write')
    assert small.load()[0]['step']==2
    (small.root/f"checkpoint-{newest['slot']}.pt").write_bytes(b'corrupt newest checkpoint')
    fallback=small.load()[0]['step']; assert fallback==1
    result={'actual_graph_resume_test_passed':True,'saved_next_step':2,'resumed_through_step':4,
            'uninterrupted_losses':expected,'resumed_losses':actual,'max_parameter_abs_difference':max_error,
            'max_loss_abs_difference':loss_error,'incomplete_tmp_ignored':True,'corrupt_latest_fallback_step':fallback,
            'seconds':time.monotonic()-start,
            'limits':'Tests program-level recovery, not abrupt physical power-loss or disk failure.'}
    atomic_json(root/'result.json',result); print(json.dumps(result,indent=2))

if __name__=='__main__': main()

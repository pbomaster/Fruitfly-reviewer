import json,time,torch
from pathlib import Path
from connectome_memory import ConnectomeMemoryLM
torch.set_num_threads(4); torch.manual_seed(20260915)
torch.backends.cuda.matmul.allow_tf32=True
m=ConnectomeMemoryLM(); results=[]
for b in [8,32,64]:
    x=torch.randint(4,4096,(b,16),device='cuda')
    bags=torch.rand(b,32,4096,device='cuda'); bags/=bags.sum(-1,keepdim=True)
    mask=torch.ones(b,32,dtype=torch.bool,device='cuda')
    for repeat in range(2):
        m.zero_grad(set_to_none=True); torch.cuda.synchronize(); t=time.monotonic()
        logits,_=m(x,m.memory(bags),mask)
        torch.nn.functional.cross_entropy(logits.flatten(0,1),x.flatten()).backward()
        torch.cuda.synchronize()
        item={'batch':b,'tokens':16,'repeat':repeat,'seconds':time.monotonic()-t,'query_grad':m.query.weight.grad.norm().item()}
        results.append(item); print(json.dumps(item),flush=True)
Path('outputs/training-benchmark.json').write_text(json.dumps(results,indent=2))

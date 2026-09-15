"""Numerically check real full-graph GPU forward and input gradient against CPU."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import scipy.sparse as sp
from safetensors.numpy import load_file

torch.manual_seed(991); torch.set_num_threads(4)
p=load_file('work/fly-model/model.safetensors'); n=len(p['brain.w_offsets'])-1
w=sp.csr_matrix((p['brain.w_values'],p['brain.w_indices'],p['brain.w_offsets']),shape=(n,n))
denom=np.maximum(np.asarray(abs(w).sum(axis=1)).ravel(),1e-6)
w=sp.diags(0.9/denom).dot(w).tocsr()
x=np.random.default_rng(991).standard_normal((n,4)).astype(np.float32)
g=np.random.default_rng(992).standard_normal((n,4)).astype(np.float32)
wg=torch.sparse_csr_tensor(torch.from_numpy(w.indptr).cuda(),torch.from_numpy(w.indices).cuda(),torch.from_numpy(w.data).cuda(),size=(n,n))
xt=torch.from_numpy(x).cuda().requires_grad_(); yt=torch.sparse.mm(wg,xt)
(yt*torch.from_numpy(g).cuda()).sum().backward(); torch.cuda.synchronize()
ref=w@x; back=w.T@g
y=yt.detach().cpu().numpy(); grad=xt.grad.cpu().numpy()
r={'nodes':n,'edges':len(w.data),'gpu':torch.cuda.get_device_name(),'driver_query_file':'outputs/nvidia-smi-5090.txt',
   'compiled_architectures':torch.cuda.get_arch_list(),'gpu_capability':list(torch.cuda.get_device_capability()),
   'forward_max_abs_error':float(abs(y-ref).max()),'gradient_max_abs_error':float(abs(grad-back).max()),
   'forward_relative_l2':float(np.linalg.norm(y-ref)/np.linalg.norm(ref)),
   'gradient_relative_l2':float(np.linalg.norm(grad-back)/np.linalg.norm(back)),
   'forward_pass':bool(np.allclose(y,ref,atol=2e-5,rtol=2e-4)),
   'gradient_pass':bool(np.allclose(grad,back,atol=2e-5,rtol=2e-4)),
   'model_sha256':hashlib.sha256(Path('work/fly-model/model.safetensors').read_bytes()).hexdigest()}
Path('outputs/full-graph-gpu-verification.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2)); assert r['forward_pass'] and r['gradient_pass']

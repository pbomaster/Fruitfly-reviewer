"""A language model whose only recurrent state is the MaleCNS central-brain graph.

No pretrained language weights. Persistent document memory contains bag-of-token
embeddings for consecutive chunks. Attention queries come from the graph state;
retrieved context enters sensory neurons, never directly the output logits.
"""
import math
import torch
from torch import nn
from safetensors.torch import load_file

class FixedSpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, wt):
        ctx.wt = wt
        return torch.sparse.mm(w, x)
    @staticmethod
    def backward(ctx, grad):
        return torch.sparse.mm(ctx.wt, grad.contiguous()), None, None

class ConnectomeMemoryLM(nn.Module):
    def __init__(self, vocab_size=4096, d=128, device='cuda'):
        super().__init__()
        p=load_file('work/fly-model/model.safetensors')
        offsets=p['brain.w_offsets'].long(); cols=p['brain.w_indices'].long()
        vals=p['brain.w_values']; n=len(offsets)-1
        rows=torch.repeat_interleave(torch.arange(n), offsets[1:]-offsets[:-1])
        denom=torch.zeros(n).scatter_add_(0,rows,vals.abs()).clamp_min(1e-6)
        # Preserve all edges and signs, normalise each incoming row for stability.
        w=torch.sparse_csr_tensor(offsets,cols,0.9*vals/denom[rows],size=(n,n))
        self.register_buffer('w',w.to(device),persistent=False)
        self.register_buffer('wt',w.t().to_sparse_csr().to(device),persistent=False)
        self.register_buffer('in_index',p['brain.in_index'].to(device),persistent=False)
        self.n=n; self.d=d; self.k=8; self.slot=len(self.in_index)//self.k
        self.emb=nn.Embedding(vocab_size,d,padding_idx=0)
        self.in_proj=nn.Parameter(torch.randn(self.k,d,self.slot)*0.04)
        self.context_proj=nn.Linear(d,len(self.in_index),bias=False)
        self.gain=nn.Parameter(torch.zeros(n))
        self.bias=nn.Parameter(torch.zeros(n))
        self.read=nn.Linear(n,d,bias=False)
        self.norm=nn.LayerNorm(d)
        self.query=nn.Linear(d,d,bias=False)
        self.head=nn.Linear(d,vocab_size)
        self.dropout=nn.Dropout(0.1)
        self.to(device)
        self.vocab_size=vocab_size

    def memory(self, bags):
        # bags[B,C,V] normalised token counts; every source token contributes.
        content=bags @ self.emb.weight
        pos=torch.arange(bags.shape[1],device=bags.device)[:,None]
        freq=torch.exp(torch.arange(0,self.d,2,device=bags.device)*(-math.log(10000.)/self.d))
        pe=torch.zeros(bags.shape[1],self.d,device=bags.device)
        pe[:,0::2]=torch.sin(pos*freq); pe[:,1::2]=torch.cos(pos*freq)
        return content+0.05*pe[None]

    def forward(self, tokens, memory, memory_mask, cache=None, graph_scale=1.):
        b,t=tokens.shape
        if cache is None:
            state=torch.zeros(self.n,b,device=tokens.device)
            prev=torch.zeros(b,self.k-1,dtype=torch.long,device=tokens.device)
        else: state,prev=cache
        ext=self.emb(torch.cat([prev,tokens],1))
        drives=[]
        for j in range(self.k):
            drives.append(ext[:,self.k-1-j:self.k-1-j+t]@self.in_proj[j])
        drive=torch.cat(drives,-1)
        logits=[]
        h=self.norm(self.read(state.t()))
        for i in range(t):
            q=self.query(h)
            scores=torch.einsum('bcd,bd->bc',memory,q)/math.sqrt(self.d)
            weights=scores.masked_fill(~memory_mask,-1e4).softmax(-1)
            context=torch.einsum('bc,bcd->bd',weights,memory)
            sensory=drive[:,i]+self.context_proj(context)
            rec=FixedSpMM.apply(state,self.w,self.wt)*graph_scale
            pre=(2*self.gain.sigmoid())[:,None]*rec
            pre=pre.index_add(0,self.in_index,sensory.t())
            state=0.1*state+0.9*torch.tanh(pre+self.bias[:,None])
            h=self.norm(self.read(state.t()))
            logits.append(self.head(self.dropout(h)))
        prev=torch.cat([prev,tokens],1)[:,-(self.k-1):]
        return torch.stack(logits,1),(state,prev)

def detach(cache):
    return tuple(x.detach() for x in cache)

if __name__=='__main__':
    import json,time
    torch.manual_seed(20260915); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    m=ConnectomeMemoryLM(); b,t=8,32
    x=torch.randint(4,4096,(b,t),device='cuda')
    bags=torch.zeros(b,64,4096,device='cuda'); bags[:,:,10:20]=.1
    mask=torch.ones(b,64,dtype=torch.bool,device='cuda')
    start=time.monotonic()
    logits,_=m(x,m.memory(bags),mask)
    loss=nn.functional.cross_entropy(logits.flatten(0,1),x.flatten())
    loss.backward(); torch.cuda.synchronize()
    r={'seconds':time.monotonic()-start,'batch':b,'tokens':t,'loss':loss.item(),
       'parameters':sum(p.numel() for p in m.parameters()),'nodes':m.n,'edges':m.w._nnz(),
       'peak_memory_bytes':torch.cuda.max_memory_allocated(),
       'gradient_norms':{n:p.grad.norm().item() for n,p in m.named_parameters()},
       'all_gradients_finite':all(torch.isfinite(p.grad).all().item() for p in m.parameters())}
    from pathlib import Path
    Path('outputs/architecture-smoke.json').write_text(json.dumps(r,indent=2))
    print(json.dumps(r,indent=2))

"""MaleCNS recurrent pointer-generator with order-preserving token memory.

The small feedforward source encoder preserves neighbouring token order; it is
randomly initialised, not a pretrained language model. Graph state supplies
every attention query and the generation/copy gate. Full source token indices
are retained. This short-document pilot deliberately keeps source length small.
"""
import math
import torch
from torch import nn
from connectome_memory import ConnectomeMemoryLM,FixedSpMM

class GraphPointer(ConnectomeMemoryLM):
    def __init__(self,vocab_size=4096,d=128,device='cuda'):
        super().__init__(vocab_size,d,device)
        self.local=nn.Conv1d(d,d,15,padding=7)
        self.key=nn.Linear(d,d,bias=False)
        self.copy_query=nn.Linear(d,d,bias=False)
        self.copy_gate=nn.Linear(d,1)
        self.location=nn.Conv1d(1,1,5,padding=2,bias=False)
        self.source_norm=nn.LayerNorm(d)
        nn.init.constant_(self.copy_gate.bias,-1.)
        self.to(device)

    def encode_source(self,source,mask):
        embedded=self.emb(source)
        local=self.local(embedded.transpose(1,2)).transpose(1,2)
        h=self.source_norm(embedded+torch.tanh(local))*mask[:,:,None]
        return self.key(h),h

    def forward(self,tokens,source,source_mask,cache=None,encoded=None,graph_scale=1.):
        b,t=tokens.shape; keys,values=encoded if encoded is not None else self.encode_source(source,source_mask)
        if cache is None:
            state=torch.zeros(self.n,b,device=tokens.device)
            prev=torch.zeros(b,self.k-1,dtype=torch.long,device=tokens.device)
            previous_attention=torch.zeros_like(source,dtype=torch.float32)
        else: state,prev,previous_attention=cache
        ext=self.emb(torch.cat([prev,tokens],1)); drives=[]
        for j in range(self.k): drives.append(ext[:,self.k-1-j:self.k-1-j+t]@self.in_proj[j])
        drive=torch.cat(drives,-1); logps=[]; attention=[]; gates=[]
        h=self.norm(self.read(state.t()))
        for i in range(t):
            # Source retrieval BEFORE graph update informs the graph's sensory input.
            scores=torch.einsum('bsd,bd->bs',keys,self.query(h))/math.sqrt(self.d)
            scores=scores.masked_fill(~source_mask,-1e4)
            context=torch.einsum('bs,bsd->bd',scores.softmax(-1),values)
            sensory=drive[:,i]+self.context_proj(context)
            pre=(2*self.gain.sigmoid())[:,None]*FixedSpMM.apply(state,self.w,self.wt)*graph_scale
            pre=pre.index_add(0,self.in_index,sensory.t())
            state=.1*state+.9*torch.tanh(pre+self.bias[:,None])
            h=self.norm(self.read(state.t()))
            # The updated graph state determines source positions to copy.
            scores=torch.einsum('bsd,bd->bs',keys,self.copy_query(h))/math.sqrt(self.d)
            scores=scores+self.location(previous_attention[:,None])[:,0]
            weights=scores.masked_fill(~source_mask,-1e4).softmax(-1)
            copied=torch.zeros(b,self.vocab_size,device=tokens.device).scatter_add(1,source,weights)
            gate=self.copy_gate(h).sigmoid()
            generated=self.head(self.dropout(h)).softmax(-1)
            probability=gate*generated+(1-gate)*copied
            logps.append(probability.clamp_min(1e-9).log()); attention.append(weights); gates.append(gate)
            previous_attention=weights
        prev=torch.cat([prev,tokens],1)[:,-(self.k-1):]
        return torch.stack(logps,1),(state,prev,previous_attention),torch.stack(attention,1),torch.stack(gates,1)

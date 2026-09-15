"""Coverage-aware MaleCNS pointer LM; tracks model attention, never gold positions.
Coverage memory is external source bookkeeping; graph state makes language decisions.
"""
import math
import torch
from torch import nn
from extractive_pointer import ExtractivePointer
from connectome_memory import FixedSpMM

class CoveragePointer(ExtractivePointer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.advance_gate=nn.Linear(self.d,1)
        nn.init.zeros_(self.advance_gate.weight); nn.init.constant_(self.advance_gate.bias,2.)
        self.coverage_input=nn.Linear(self.d+2,len(self.in_index),bias=False)
        nn.init.zeros_(self.coverage_input.weight)
        self.coverage_penalty=nn.Parameter(torch.tensor(1.))
        self.to(self.emb.weight.device)

    def forward(self,tokens,source,source_mask,cache=None,encoded=None,graph_scale=1.,query_drive=None):
        if query_drive is None: raise ValueError('Query-only drive required')
        b,t=tokens.shape; keys,values=encoded if encoded is not None else self.encode_source(source,source_mask)
        if cache is None:
            state=torch.zeros(self.n,b,device=tokens.device)
            prev=torch.zeros(b,self.k-1,dtype=torch.long,device=tokens.device)
            previous_attention=torch.zeros_like(source,dtype=torch.float32)
            coverage=torch.zeros_like(previous_attention)
        else: state,prev,previous_attention,coverage=cache
        ext=self.emb(torch.cat([prev,tokens],1))
        drive=torch.cat([ext[:,self.k-1-j:self.k-1-j+t]@self.in_proj[j] for j in range(self.k)],-1)
        logps=[]; attention=[]; gates=[]
        h=self.norm(self.read(state.t()))
        for i in range(t):
            scores=torch.einsum('bsd,bd->bs',keys,self.query(h))/math.sqrt(self.d)
            retrieval=scores.masked_fill(~source_mask,-1e4).softmax(-1)
            has_history=(previous_attention.sum(-1,keepdim=True)>0).float()
            # Feed the model's own last read position back through sensory neurons.
            context_weights=(1-.5*has_history)*retrieval+.5*previous_attention
            context=torch.einsum('bs,bsd->bd',context_weights,values)
            used_mass=coverage.sum(-1,keepdim=True)
            used_content=torch.einsum('bs,bsd->bd',coverage,values)/used_mass.clamp_min(1.)
            progress=torch.cat([used_mass/100.,coverage.clamp(max=1.).sum(-1,keepdim=True)/source_mask.sum(-1,keepdim=True)],1)
            sensory=drive[:,i]+self.context_proj(context)+query_drive+self.coverage_input(torch.cat([used_content,progress],1))
            # Two propagation steps let current sensory input reach non-input read neurons.
            for _ in range(2):
                pre=(2*self.gain.sigmoid())[:,None]*FixedSpMM.apply(state,self.w,self.wt)*graph_scale
                pre=pre.index_add(0,self.in_index,sensory.t())
                state=.1*state+.9*torch.tanh(pre+self.bias[:,None])
            h=self.norm(self.read(state.t()))
            scores=torch.einsum('bsd,bd->bs',keys,self.copy_query(h))/math.sqrt(self.d)
            scores=scores+self.location(previous_attention[:,None])[:,0]-torch.nn.functional.softplus(self.coverage_penalty)*coverage.clamp(max=2.)
            content=scores.masked_fill(~source_mask,-1e4).softmax(-1)
            shifted=torch.cat([torch.zeros_like(previous_attention[:,:1]),previous_attention[:,:-1]],1)*source_mask
            mass=shifted.sum(-1,keepdim=True)
            advance=self.advance_gate(h).sigmoid()*(mass>1e-8)
            weights=advance*shifted/mass.clamp_min(1e-8)+(1-advance)*content
            copied=torch.zeros(b,self.vocab_size,device=tokens.device).scatter_add(1,source,weights)
            gate=self.copy_gate(h).sigmoid(); generated=self.head(self.dropout(h)).softmax(-1)
            probability=gate*generated+(1-gate)*copied
            logps.append(probability.clamp_min(1e-9).log()); attention.append(weights); gates.append(gate)
            coverage=coverage+weights
            previous_attention=weights
        prev=torch.cat([prev,tokens],1)[:,-(self.k-1):]
        return torch.stack(logps,1),(state,prev,previous_attention,coverage),torch.stack(attention,1),torch.stack(gates,1)

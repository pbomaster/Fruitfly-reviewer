"""Persistent ordered query memory, injected through MaleCNS sensory neurons.

No answer tokens enter query memory. This extends the project's own model;
there are no externally pretrained language weights or query-to-logit skip.
"""
import torch
from torch import nn
from torch.nn import functional as F
from graph_pointer import GraphPointer

class QueryPointer(GraphPointer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.query_pool=nn.Linear(self.d*8,self.d)
        self.query_sensory=nn.Linear(self.d,len(self.in_index),bias=False)
        self.to(self.emb.weight.device)

    def encode_query(self,ids,mask):
        _,v=self.encode_source(ids,mask)
        rows=[]
        for row,m in zip(v,mask):
            n=int(m.sum())
            rows.append(F.adaptive_avg_pool1d(row[:n].t()[None],8).flatten())
        return self.query_sensory(torch.tanh(self.query_pool(torch.stack(rows))))

    def forward(self,tokens,source,source_mask,cache=None,encoded=None,graph_scale=1.,query_drive=None):
        # A scoped projection hook adds the immutable prompt to sensory input.
        # The hook has no persistent state and is removed even if forward fails.
        if query_drive is None: raise ValueError('Explicit query-only memory required')
        hook=self.context_proj.register_forward_hook(lambda module,args,output: output+query_drive)
        try: return super().forward(tokens,source,source_mask,cache,encoded,graph_scale)
        finally: hook.remove()

def batch(tasks,examples):
    source_len=max(len(e['source_ids']) for e in examples)
    query_len=max(len(e['prefix']) for e in examples)
    target_len=max(len(e['target']) for e in examples)
    b=len(examples)
    source=torch.zeros(b,source_len,dtype=torch.long,device='cuda')
    query=torch.zeros(b,query_len,dtype=torch.long,device='cuda')
    y=torch.full((b,target_len),-100,dtype=torch.long,device='cuda')
    x=torch.zeros_like(y); align=torch.full_like(y,-100)
    for j,e in enumerate(examples):
        source[j,:len(e['source_ids'])]=torch.tensor(e['source_ids'],device='cuda')
        query[j,:len(e['prefix'])]=torch.tensor(e['prefix'],device='cuda')
        y[j,:len(e['target'])]=torch.tensor(e['target'],device='cuda')
        x[j,:len(e['target'])]=torch.tensor([1]+e['target'][:-1],device='cuda')
        align[j,:len(e['align'])]=torch.tensor([a if a>=0 else -100 for a in e['align']],device='cuda')
    return x,source,source!=0,query,query!=0,y,align

@torch.no_grad()
def predict(model,tasks,examples,graph_scale=1.,zero_query=False):
    model.eval(); rows=[]
    for e in examples:
        x,s,sm,q,qm,_,_=batch(tasks,[e]); enc=model.encode_source(s,sm)
        drive=model.encode_query(q,qm)
        if zero_query: drive=drive*0
        cache=None; token=torch.ones(1,1,dtype=torch.long,device='cuda'); ids=[]; pos=[]
        for _ in range(14):
            p,cache,a,g=model(token,s,sm,cache,enc,graph_scale,query_drive=drive)
            t=int(p[0,-1].argmax()); ids.append(t); pos.append(int(a[0,-1].argmax()))
            if t==2: break
            token=torch.tensor([[t]],device='cuda')
        raw=tasks.tok.decode(ids)
        rows.append({**{k:e[k] for k in ['source','query','answer','seed','queried_name','changed']},
                     'generated_ids':ids,'raw_output':raw,'exact':raw.strip()==e['answer'],
                     'selected_source_positions':pos,'graph_scale':graph_scale,'zero_query':zero_query})
    return rows

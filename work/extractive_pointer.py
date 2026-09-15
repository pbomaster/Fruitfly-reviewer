"""Graph pointer LM with local token and full-sentence source representations."""
import torch
from torch import nn
from query_pointer import QueryPointer

class ExtractivePointer(QueryPointer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.sentence_key=nn.Linear(self.d,self.d,bias=False)
        self.boundary=nn.Embedding(2,self.d)
        read_mask=torch.ones(self.n,device=self.emb.weight.device)
        read_mask[self.in_index]=0
        self.register_buffer('read_mask',read_mask,persistent=False)
        # Evidence must traverse anatomical edges before reaching the decoder.
        self.read.register_forward_pre_hook(lambda module,args: (args[0]*self.read_mask,))
        self.to(self.emb.weight.device)

    def forward(self,tokens,source,source_mask,cache=None,encoded=None,graph_scale=1.,query_drive=None):
        if encoded is None: encoded=self.encode_source(source,source_mask)
        if cache is None:
            warm=torch.ones(tokens.shape[0],3,dtype=torch.long,device=tokens.device)
            _,cache,_,_=super().forward(warm,source,source_mask,None,encoded,graph_scale,query_drive)
        return super().forward(tokens,source,source_mask,cache,encoded,graph_scale,query_drive)

    def encode_source(self,source,mask):
        keys,values=super().encode_source(source,mask)
        # BOS marks sentence starts; all sentences remain available at inference.
        if not hasattr(self,'sentence_key'): return keys,values
        groups=(source==1).cumsum(1); count=int(groups.max())+1
        sums=torch.zeros(source.shape[0],count,self.d,device=source.device).scatter_add(1,groups[:,:,None].expand_as(values),values)
        sizes=torch.zeros(source.shape[0],count,device=source.device).scatter_add(1,groups,mask.float())
        sentence=(sums/sizes.clamp_min(1)[:,:,None]).gather(1,groups[:,:,None].expand_as(values))
        keys=keys+self.sentence_key(sentence)+self.boundary((source==1).long())
        return keys,values

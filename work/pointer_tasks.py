"""Controlled generated FACT data, unrelated to the evaluation paper.

These are ordinary templates, not AI-written reviews. The model must bind a
queried label to a random number amid distractors and generate/copy that number.
Exact equality is evaluated on unseen documents. Synthetic success is NOT a
scientific review success. No regex is used to generate model predictions.
"""
import random,re
import torch
from tokenizers import Tokenizer

TOKENIZER='work/science-corpus/tokenizer.json'
NAMES=['Aster','Birch','Cedar','Dahlia','Elm','Fern','Ginkgo','Hazel','Iris','Juniper','Kalmia','Lotus',
       'Maple','Nettle','Olive','Pine','Quince','Reed','Sage','Thyme','Umber','Violet','Willow','Yarrow']

class Tasks:
    def __init__(self): self.tok=Tokenizer.from_file(TOKENIZER)
    def example(self,seed,records=4,changed=False,style=0):
        rng=random.Random(seed); names=rng.sample(NAMES,records)
        numbers=rng.sample(range(1000,10000),records); index=rng.randrange(records)
        if changed: numbers[index]=1000+(numbers[index]-1000+1379)%9000
        parts=[]; spans=[]; offset=0
        for name,number in zip(names,numbers):
            if style==0: s=f'Sample {name} contains {number} cells. '
            else: s=f'The cell count for sample {name} is {number}. '
            start=offset+s.index(str(number)); spans.append((start,start+len(str(number))))
            parts.append(s); offset+=len(s)
        source=''.join(parts).strip()
        query=f'How many cells does sample {names[index]} contain? Answer:'
        enc=self.tok.encode(source)
        positions=[i for i,(a,b) in enumerate(enc.offsets) if a<spans[index][1] and b>spans[index][0]]
        answer_ids=[enc.ids[i] for i in positions]
        # Exact copied source token span may include a leading whitespace token.
        target=answer_ids+[2]
        prefix=[1]+self.tok.encode(query).ids
        return {'source':source,'query':query,'answer':str(numbers[index]),'source_ids':enc.ids,
                'prefix':prefix,'target':target,'align':positions+[-1],'seed':seed,'records':records,'style':style,
                'changed':changed,'queried_name':names[index]}
    def batch(self,examples):
        b=len(examples); s=max(len(x['source_ids']) for x in examples)
        length=max(len(x['prefix'])+len(x['target'])-1 for x in examples)
        source=torch.zeros(b,s,dtype=torch.long,device='cuda'); mask=torch.zeros_like(source,dtype=torch.bool)
        tokens=torch.zeros(b,length,dtype=torch.long,device='cuda'); y=torch.full_like(tokens,-100); align=torch.full_like(tokens,-100)
        for j,e in enumerate(examples):
            n=len(e['source_ids']); source[j,:n]=torch.tensor(e['source_ids'],device='cuda'); mask[j,:n]=True
            sequence=e['prefix']+e['target'][:-1]; tokens[j,:len(sequence)]=torch.tensor(sequence,device='cuda')
            start=len(e['prefix'])-1
            y[j,start:start+len(e['target'])]=torch.tensor(e['target'],device='cuda')
            align[j,start:start+len(e['target'])]=torch.tensor([a if a>=0 else -100 for a in e['align']],device='cuda')
        return tokens,source,mask,y,align
    def training_batch(self,step,batch,seed=44117):
        return [self.example(seed+step*batch+j,records=4) for j in range(batch)]

@torch.no_grad()
def predict(model,tasks,examples,graph_scale=1.):
    # Batched prefixes of equal lengths are not assumed: process one example at a time.
    model.eval(); results=[]
    for e in examples:
        source=torch.tensor([e['source_ids']],device='cuda'); mask=torch.ones_like(source,dtype=torch.bool)
        enc=model.encode_source(source,mask)
        x=torch.tensor([e['prefix']],device='cuda')
        output,cache,att,gates=model(x,source,mask,encoded=enc,graph_scale=graph_scale)
        ids=[]; positions=[]
        for j in range(14):
            token=output[0,-1].argmax().item(); ids.append(token)
            positions.append(att[0,-1].argmax().item())
            if token==2: break
            output,cache,att,gates=model(torch.tensor([[token]],device='cuda'),source,mask,cache,enc,graph_scale)
        raw=tasks.tok.decode(ids)
        results.append({**{k:e[k] for k in ['source','query','answer','seed','records','style','changed','queried_name']},
                        'generated_ids':ids,'raw_output':raw,'exact':raw.strip()==e['answer'],
                        'selected_source_positions':positions,'graph_scale':graph_scale})
    return results

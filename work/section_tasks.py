import json,random
from pathlib import Path
from pointer_tasks import Tasks
from extractive_tasks import PROMPT

PATH=Path('work/section-corpus.json')
def encode_sentences(tok,sentences):
    ids=[]; spans={}; texts=[]
    for item in sentences:
        ids += [1]+tok.encode('['+item['section']+'] ').ids
        start=len(ids); ids+=tok.encode(item['text']+' ').ids
        spans[item['text']]=list(range(start,len(ids)))
        texts.append('['+item['section']+'] '+item['text'])
    return ids,spans,'\n'.join(texts)

class SectionTasks(Tasks):
    def __init__(self):
        super().__init__(); records=json.loads(PATH.read_text(encoding='utf8'))
        self.data={s:[r for r in records if r['split']==s] for s in ['train','dev','test']}
    def example(self,seed,split='dev',distractors=22):
        rng=random.Random(seed); p=rng.choice(self.data[split])
        c=rng.choice(p['contributions']); l=rng.choice(p['limitations'])
        positives={x['text'] for k in ['contributions','limitations'] for x in p[k]}
        negatives=[x for x in p['sentences'] if x['text'] not in positives]
        choices=[c,l]+rng.sample(negatives,min(distractors,len(negatives))); rng.shuffle(choices)
        ids,spans,source=encode_sentences(self.tok,choices)
        align=spans[c['text']]+spans[l['text']]+[-1]; target=[ids[j] for j in align[:-1]]+[2]
        return {'source':source,'source_ids':ids,'prefix':[1]+self.tok.encode(PROMPT).ids,'query':PROMPT,
                'target':target,'align':align,'answer':self.tok.decode(target),'seed':seed,'queried_name':p['id'],
                'changed':False,'contribution':c['text'],'limitation':l['text']}
    def training_batch(self,step,batch,seed=794119):
        return [self.example(seed+step*batch+j,'train') for j in range(batch)]

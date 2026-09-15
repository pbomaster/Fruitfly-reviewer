import json,random
from pathlib import Path
from pointer_tasks import Tasks

PATH=Path('work/extractive-corpus.json')
PROMPT='Write a short paragraph with one main contribution and one limitation of this study, using the source sentences.'
class ExtractiveTasks(Tasks):
    def __init__(self):
        super().__init__(); records=json.loads(PATH.read_text(encoding='utf8'))
        self.data={s:[r for r in records if r['split']==s] for s in ['train','dev','test']}
    def example(self,seed,split='dev'):
        rng=random.Random(seed); p=rng.choice(self.data[split])
        c=rng.choice(p['contributions']); l=rng.choice(p['limitations'])
        positives={x['text'] for k in ['contributions','limitations'] for x in p[k]}
        negatives=[x for x in p['sentences'] if x['text'] not in positives]
        choices=[c,l]+rng.sample(negatives,min(6,len(negatives))); rng.shuffle(choices)
        ids=[]; spans={}; text=[]
        for sentence in choices:
            ids.append(1); start=len(ids); tokens=self.tok.encode(sentence['text']+' ').ids
            ids+=tokens; spans[sentence['text']]=list(range(start,len(ids))); text.append(sentence['text'])
        align=spans[c['text']]+spans[l['text']]+[-1]
        target=[ids[j] for j in align[:-1]]+[2]
        return {'source':'\n'.join(text),'source_ids':ids,'prefix':[1]+self.tok.encode(PROMPT).ids,
                'query':PROMPT,'target':target,'align':align,'answer':self.tok.decode(target),
                'seed':seed,'queried_name':p['id'],'changed':False,'contribution':c['text'],'limitation':l['text']}
    def training_batch(self,step,batch,seed=774119):
        return [self.example(seed+step*batch+j,'train') for j in range(batch)]

"""Human-authored scientific passages: reconstruct a masked numeric span.

Only the existing train/dev/test paper split is used. The evaluation PDF is not
read. Source sentences and answer spans come from publisher text, not another
AI. This tests passage matching/copying, not semantic reasoning or reviewing.
"""
import json,random,re
from pathlib import Path
from pointer_tasks import Tasks
from resumable_checkpoint import atomic_json,sha256

PATH=Path('work/science-pointer-corpus.json')
def prepare():
    result={'train':[],'dev':[],'test':[]}; papers={s:set() for s in result}
    for line in Path('work/science-corpus/papers.jsonl').read_text(encoding='utf8').splitlines():
        p=json.loads(line); candidates=[]
        for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z])',p['source']):
            sentence=re.sub(r'\s+',' ',sentence).strip()
            if not 15<=len(sentence.split())<=65: continue
            matches=list(re.finditer(r'(?<![\w.])\d{2,6}(?:\.\d+)?(?![\w.])',sentence))
            if 1<=len(matches)<=4:
                candidates.append({'text':sentence,'spans':[[m.start(),m.end()] for m in matches]})
        if len(candidates)<4: continue
        rng=random.Random(p['id']); rng.shuffle(candidates); candidates=candidates[:16]
        for j,target in enumerate(candidates):
            others=[x for k,x in enumerate(candidates) if k!=j]
            result[p['split']].append({'paper_id':p['id'],'target_sentence':target,'distractors':rng.sample(others,3)})
        papers[p['split']].add(p['id'])
    atomic_json(PATH,result)
    summary={'source_sha256':sha256('work/science-corpus/papers.jsonl'),'prepared_sha256':sha256(PATH),
             'split':{s:{'papers':len(papers[s]),'passage_sets':len(result[s])} for s in result},
             'scope':'Four human-authored scientific sentences. Recover an explicitly masked number using a query excerpt.',
             'target_pdf_used':False,'limitations':['May select years, figure-related values or measurement values; not all are core findings.',
                                                  'Query passage overlap makes this evidence matching, not free-form question understanding.']}
    atomic_json('outputs/science-pointer-corpus.json',summary); print(json.dumps(summary,indent=2))

class ScienceTasks(Tasks):
    def __init__(self):
        super().__init__(); self.data=json.loads(PATH.read_text(encoding='utf8'))
    def example(self,seed,split='dev',changed=False,records=4,style=0,retry=0):
        rng=random.Random(seed); entry=self.data[split][rng.randrange(len(self.data[split]))]
        target=dict(entry['target_sentence']); a,b=rng.choice(target['spans']); sentence=target['text']; answer=sentence[a:b]
        if changed:
            replacement=str(int(float(answer))+137)
            sentence=sentence[:a]+replacement+sentence[b:]; answer=replacement; b=a+len(replacement)
        before=sentence[max(0,a-65):a]; after=sentence[b:b+35]
        query='Complete the missing value in this passage: '+before+'[MASK]'+after+' Answer:'
        parts=[(sentence,True)]+[(x['text'],False) for x in entry['distractors']]
        rng.shuffle(parts); source=''; start=None
        for text,is_target in parts:
            if is_target: start=len(source)+a
            source+=text+' '
        source=source.strip(); enc=self.tok.encode(source)
        positions=[i for i,(x,y) in enumerate(enc.offsets) if x<start+len(answer) and y>start]
        # BPE spans may include punctuation along with the value. Such targets
        # are excluded to keep string exactness a valid measure of the number.
        ids=[enc.ids[i] for i in positions]
        decoded=self.tok.decode(ids).strip()
        if decoded!=answer:
            # Deterministic retry on another independent item, bounded externally
            # by a corpus with ordinary whitespace-delimited numeric values.
            if retry>=20: raise RuntimeError('Could not select an exact numeric token span after 21 attempts')
            return self.example(seed+1000003,split,changed,records,style,retry+1)
        return {'source':source,'query':query,'answer':answer,'source_ids':enc.ids,
                'prefix':[1]+self.tok.encode(query).ids,'target':ids+[2],'align':positions+[-1],
                'seed':seed,'records':4,'style':style,'changed':changed,'queried_name':entry['paper_id']}
    def training_batch(self,step,batch,seed=554117):
        return [self.example(seed+step*batch+j,split='train') for j in range(batch)]

if __name__=='__main__': prepare()

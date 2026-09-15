"""Distantly supervised contribution/limitation excerpts from publisher prose.

Targets are verbatim human sentences, selected by general cues/section metadata;
they are NOT expert-written reviews. Target PDF is never loaded. Preserve this
limitation when reporting. Existing paper-level splits remain unchanged.
"""
import json,re,xml.etree.ElementTree as ET,random,collections
from pathlib import Path
from tokenizers import Tokenizer
from resumable_checkpoint import atomic_json,sha256

OUT=Path('work/extractive-corpus.json')
def clean(s): return re.sub(r'\s+',' ',s).strip()
def text(e): return clean(' '.join(e.itertext())) if e is not None else ''
def sentences(s):
    return [x for x in re.split(r'(?<=[.!?])\s+(?=[A-Z])',clean(s)) if 12<=len(x.split())<=65]
CONTRIB=re.compile(r'\b(we (show|find|found|demonstrate|develop|present|report|reveal|identify|provide|establish)|here[,]? we|our (results|findings|study)|these (results|findings) (show|demonstrate|suggest|reveal))\b',re.I)
LIMIT=re.compile(r'\b(limitations?|caveats?|cannot|could not|unable to|remains unclear|remain unclear|further (studies|work|research)|future (studies|work|research)|will be (needed|necessary|required))\b',re.I)

def main():
    tok=Tokenizer.from_file('work/science-corpus/tokenizer.json'); records=[]; counts=collections.Counter()
    for line in Path('work/science-corpus/papers.jsonl').read_text(encoding='utf8').splitlines():
        p=json.loads(line); root=ET.parse(p['source_xml']['file']).getroot(); body=root.find('body')
        abstract=sentences(text(root.find('./front/article-meta/abstract')))
        contrib=[s for s in abstract if CONTRIB.search(s) and not LIMIT.search(s)]
        limits=[]; allsent=[]
        if body is None: continue
        for sec in body.findall('sec'):
            heading=text(sec.find('title')).lower()
            ss=sentences(text(sec)); allsent+=ss
            if any(k in heading for k in ['discuss','conclu','limitation']):
                limits += [s for s in ss if LIMIT.search(s)]
        # Broader body fallback uses the same generic cue; it is noisy supervision.
        if not limits: limits=[s for s in allsent if LIMIT.search(s)]
        if not contrib: contrib=[s for s in allsent if CONTRIB.search(s) and not LIMIT.search(s)][:8]
        if not contrib or not limits: counts['missing_pair']+=1; continue
        pool=list(dict.fromkeys(abstract+allsent))
        def encode(xs):
            return [{'text':s,'ids':tok.encode(s).ids} for s in xs if len(tok.encode(s).ids)<=160]
        c=encode(contrib); l=encode(limits)
        if not c or not l: counts['long_pair']+=1; continue
        pool=encode(pool)
        records.append({'id':p['id'],'split':p['split'],'title':p['title'],'contributions':c,'limitations':l,'sentences':pool})
    # Exact positive sentence leakage exclusion (do not modify held-out records).
    held={x['text'] for r in records if r['split']!='train' for k in ['contributions','limitations'] for x in r[k]}
    for r in records:
        if r['split']=='train':
            for k in ['contributions','limitations']: r[k]=[x for x in r[k] if x['text'] not in held]
    records=[r for r in records if r['contributions'] and r['limitations']]
    atomic_json(OUT,records)
    stats={'sha256':sha256(OUT),'source_sha256':sha256('work/science-corpus/papers.jsonl'),
           'splits':{s:{'papers':sum(r['split']==s for r in records),'positive_sentences':sum(len(r['contributions'])+len(r['limitations']) for r in records if r['split']==s)} for s in ['train','dev','test']},
           'excluded':dict(counts),'target_pdf_used':False,
           'scope':'Training targets concatenate verbatim publisher-authored contribution and limitation sentences. Weak labels from generic cues; not independent expert reviews.',
           'limitations':['Cue labels may describe prior work or biological limitations rather than study limitations.','Extractive summaries do not establish independent critical reasoning.','Not every paper has explicit author limitations.']}
    atomic_json('outputs/extractive-corpus.json',stats); print(json.dumps(stats,indent=2))

if __name__=='__main__': main()

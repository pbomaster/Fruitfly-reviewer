"""Conservative line-wrap repair using vocabulary from training papers only.

Preserves raw extraction and logs every change. Does not rewrite model outputs.
"""
import re,json,collections
from pathlib import Path
from resumable_checkpoint import atomic_json,sha256

def main():
    corpus=Path('work/science-corpus/papers.jsonl'); counts=collections.Counter()
    with corpus.open(encoding='utf8') as f:
        for line in f:
            paper=json.loads(line)
            if paper['split']=='train':
                counts.update(re.findall(r'[A-Za-z]+(?:-[A-Za-z]+)*',paper['source'].lower()))
    raw=Path('outputs/paper-pages.json'); pages=json.loads(raw.read_text(encoding='utf8'))
    changes=[]; repaired=[]
    pattern=re.compile(r'([A-Za-z][A-Za-z-]*)[ \t]*-[ \t]*\r?\n[ \t]*([a-z][A-Za-z-]*)')
    for page,s in enumerate(pages,1):
        def fix(m):
            first=m.group(1); second=m.group(2)
            candidates=[first+second,first+'-'+second]
            supported=[x for x in candidates if counts[x.lower()]>=3]
            if not supported: return m.group(0)
            chosen=max(supported,key=lambda x:counts[x.lower()])
            changes.append({'page':page,'start':m.start(),'original':m.group(0),'replacement':chosen,
                            'training_occurrences':counts[chosen.lower()]})
            return chosen
        repaired.append(pattern.sub(fix,s))
    atomic_json('outputs/paper-pages-normalized.json',repaired)
    atomic_json('outputs/paper-normalization-audit.json',{'raw_sha256':sha256(raw),'vocabulary_corpus_sha256':sha256(corpus),
                'dictionary_split':'train only','rule':'Repair wrapped hyphens only when supported at least three times by training-paper vocabulary; keep unknown words unchanged.',
                'pages':len(pages),'changes':changes,'model_outputs_modified':False})
    print(json.dumps({'pages':len(pages),'changes':len(changes),'raw_preserved':True}))
if __name__=='__main__': main()

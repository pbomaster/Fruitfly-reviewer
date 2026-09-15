"""Attach publisher abstract/body metadata to the separately refined corpus."""
import json,xml.etree.ElementTree as ET
from pathlib import Path
from resumable_checkpoint import atomic_json,sha256

def main():
    source=Path('work/extractive-specific-corpus.json')
    records=json.loads(source.read_text(encoding='utf8'))
    papers={p['id']:p for p in map(json.loads,Path('work/science-corpus/papers.jsonl').read_text(encoding='utf8').splitlines())}
    result=[]
    for r in records:
        root=ET.parse(papers[r['id']]['source_xml']['file']).getroot()
        a=root.find('./front/article-meta/abstract')
        abstract=' '.join(' '.join(a.itertext()).split()) if a is not None else ''
        contributions=[x for x in r['contributions'] if x['text'] in abstract]
        if not contributions: continue
        def tag(x): return {**x,'section':'abstract' if x['text'] in abstract else 'body'}
        result.append({**r,'contributions':[tag(x) for x in contributions],
                       'limitations':[tag(x) for x in r['limitations']], 'sentences':[tag(x) for x in r['sentences']]})
    out=Path('work/section-corpus.json'); atomic_json(out,result)
    report={'source_sha256':sha256(source),'sha256':sha256(out),'target_pdf_read':False,
            'contribution_rule':'Previously selected human contribution sentence must occur in publisher abstract.',
            'section_labels':'Publisher metadata only; no target-relevant topics or answers.',
            'splits':{s:sum(r['split']==s for r in result) for s in ['train','dev','test']},
            'scope':'Weak extractive supervision, not independently authored reviews or verified major-result labels.'}
    atomic_json('outputs/section-corpus.json',report); print(json.dumps(report,indent=2))
if __name__=='__main__': main()

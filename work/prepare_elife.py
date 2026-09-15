"""Build an auditable human-review corpus, excluding evaluation paper and replies."""
import collections, hashlib, json, re, xml.etree.ElementTree as ET
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

OUT=Path('work/science-corpus'); OUT.mkdir(exist_ok=True)
def clean(text): return re.sub(r'\s+',' ',text or '').strip()
def txt(e): return clean(' '.join(e.itertext())) if e is not None else ''
def sha(s): return hashlib.sha256(s.encode()).hexdigest()

def main():
    source=json.loads(Path('work/elife/manifest.json').read_text(encoding='utf8'))
    papers=[]; excluded=collections.Counter(); seen=set()
    for entry in source['files']:
        raw=Path(entry['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==entry['sha256']
        r=ET.fromstring(raw); meta=r.find('./front/article-meta')
        years=[int(x.text) for x in meta.findall('./pub-date/year') if x.text and x.text.isdigit()]
        if not years or max(years)>2022: excluded['date_after_2022_or_missing']+=1; continue
        permissions=meta.find('permissions'); license_xml=ET.tostring(permissions,encoding='unicode') if permissions is not None else ''
        if 'creativecommons.org/licenses/by/' not in license_xml and 'creativecommons.org/publicdomain/zero/' not in license_xml:
            excluded['license_not_recognised']+=1; continue
        title=txt(meta.find('./title-group/article-title'))
        doi=next((x.text for x in meta.findall('article-id') if x.get('pub-id-type')=='doi'),'')
        source_text=title+'\n'+txt(meta.find('abstract'))+'\n'+txt(r.find('body'))
        # Explicit target identity exclusion, in addition to the publication cutoff.
        if '10.1016/j.cell.2026.08.015' in source_text.lower() or 'sexual dimorphism in the complete drosophila male central nervous system connectome' in source_text.lower():
            excluded['target_match']+=1; continue
        key=sha(re.sub(r'\W','',title.lower()))
        if key in seen: excluded['duplicate_title']+=1; continue
        seen.add(key)
        if len(source_text.split())<500: excluded['short_source']+=1; continue
        paragraphs=[]
        for letter in r.findall("./sub-article[@article-type='decision-letter']"):
            body=letter.find('body')
            if body is None: continue
            for p in body.iter('p'):
                s=txt(p)
                if not 40<=len(s.split())<=240: continue
                if any(z in s.lower() for z in ['thank you for submitting','reviewers have opted','interests of transparency','reviewing editor has drafted','article has been reviewed','reviewers have discussed','reviewer identity']): continue
                if sha(s) not in {sha(x) for x in paragraphs}: paragraphs.append(s)
        if not paragraphs: excluded['no_substantive_review_paragraph']+=1; continue
        # Paper-level split; no review or article version can cross splits.
        bucket=int(sha(doi or title)[:8],16)%20
        split='test' if bucket==0 else 'dev' if bucket==1 else 'train'
        papers.append({'id':doi,'title':title,'year':max(years),'split':split,'source':source_text,
                       'targets':paragraphs,'source_xml':entry,'license_xml':license_xml})
    # Review paragraph deduplication across papers: keep train/test strictly disjoint.
    heldout={sha(t) for p in papers if p['split']!='train' for t in p['targets']}
    for p in papers:
        if p['split']=='train': p['targets']=[t for t in p['targets'] if sha(t) not in heldout]
    papers=[p for p in papers if p['targets']]
    tokenizer=Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder=decoders.ByteLevel()
    trainer=trainers.BpeTrainer(vocab_size=4096,special_tokens=['<pad>','<bos>','<eos>','<unk>'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator((s for p in papers if p['split']=='train' for s in [p['source']]+p['targets']),trainer=trainer)
    tokenizer.save(str(OUT/'tokenizer.json'))
    with (OUT/'papers.jsonl').open('w',encoding='utf8') as f:
        for p in papers: f.write(json.dumps(p,ensure_ascii=False)+'\n')
    encoded=[]
    for p in papers:
        encoded.append({'id':p['id'],'title':p['title'],'split':p['split'],
                        'source':tokenizer.encode(p['source']).ids,
                        'targets':[tokenizer.encode(t).ids+[2] for t in p['targets']]})
    (OUT/'encoded.json').write_text(json.dumps(encoded),encoding='utf8')
    stats={'publisher_revision':source['revision'],'excluded':dict(excluded),'vocabulary_size':tokenizer.get_vocab_size(),
           'split':{s:{'papers':sum(p['split']==s for p in papers),
                        'review_paragraphs':sum(len(p['targets']) for p in papers if p['split']==s),
                        'source_tokens':sum(len(p['source']) for p in encoded if p['split']==s),
                        'review_tokens':sum(sum(map(len,p['targets'])) for p in encoded if p['split']==s)} for s in ['train','dev','test']},
           'target_exclusion':'All publication dates <= 2022; target title and DOI also screened. No target text used to train vocabulary or parameters.',
           'limitations':['Decision letters are human editorial/reviewer text, not necessarily standalone balanced reviews.',
                          'Published revised articles can differ from the manuscript seen by reviewers; criticism may already be addressed.',
                          'Chunks preserve all token counts, but lose order within chunks and do not encode figures.'],
           'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}}
    Path('outputs/science-corpus.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
    print(json.dumps(stats,indent=2))

if __name__=='__main__': main()

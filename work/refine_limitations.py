"""Prepare a separate high-specificity weak-label candidate; no running data edits.

Rules reference research methods and evidential uncertainty, not target topics.
Author statements remain verbatim; labels are not expert review judgements.
"""
import json,re,random,collections,xml.etree.ElementTree as ET
from pathlib import Path
from resumable_checkpoint import atomic_json,sha256

RULES={
 'explicit_caveat':r'\b(?:one|a|another|the|several|some|these|important|major)\s+(?:(?:potential|important|major|possible)\s+)?caveats?\b',
 'study_limitation':r'\b(?:limitations? (?:of|in|to|with) (?:this|the present|our|the current)|(?:our (?:study|results|analysis|analyses|approach|method)|this study|the present study|the current study)\b.{0,85}\blimitations?\b)',
 'evidential_uncertainty':r'\bwe (?:cannot|could not|were unable to|are unable to)\s+(?:completely |fully |directly |definitively |entirely )?(?:exclude|discount|rule out|determine|establish|distinguish|assess|evaluate|measure|observe|reconstruct|segment|detect|obtain|resolve|test)\b',
 'method_scope':r'\b(?:our|this study|the present study|the current study)\b.{0,100}\b(?:cannot|could not|unable to)\s+(?:directly |explicitly |definitively |fully )?(?:rule out|exclude|determine|resolve|measure|distinguish|establish|identify|test)\b',
}
def main():
    source=Path('work/extractive-corpus.json'); records=json.loads(source.read_text(encoding='utf8'))
    patterns={k:re.compile(v,re.I) for k,v in RULES.items()}; refined=[]; audit=[]; counts=collections.Counter()
    # Preserve revision 1 outputs before improving title/sentence boundaries.
    for path in ['work/extractive-specific-corpus.json','outputs/limitation-refinement.json','outputs/limitation-refinement-train-sample.json']:
        p=Path(path); backup=p.with_name(p.stem+'-v1'+p.suffix)
        if p.exists() and not backup.exists(): backup.write_bytes(p.read_bytes())
    papers={p['id']:p for p in map(json.loads,Path('work/science-corpus/papers.jsonl').read_text(encoding='utf8').splitlines())}
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file('work/science-corpus/tokenizer.json')
    for row in records:
        root=ET.parse(papers[row['id']]['source_xml']['file']).getroot()
        titles=sorted({' '.join(' '.join(e.itertext()).split()) for e in root.findall('.//sec/title')},key=len,reverse=True)
        def repair(item):
            original=item['text']; sentence=original
            for title in titles:
                if title and sentence.startswith(title+' '): sentence=sentence[len(title):].strip()
            return {'text':sentence,'ids':tok.encode(sentence).ids,**({'original_with_heading':original} if sentence!=original else {})}
        row={**row,**{k:[repair(x) for x in row[k]] for k in ['limitations','contributions','sentences']}}
        kept=[]
        for item in row['limitations']:
            matches=[name for name,p in patterns.items() if p.search(item['text'])]
            if matches:
                kept.append(item)
                audit.append({'paper':row['id'],'split':row['split'],'text':item['text'],'rules':matches})
                counts.update(matches)
        if kept:
            refined.append({**row,'limitations':kept})
    out=Path('work/extractive-specific-corpus.json'); atomic_json(out,refined)
    train=[a for a in audit if a['split']=='train']
    sample=random.Random(722013).sample(train,min(24,len(train)))
    report={'source_sha256':sha256(source),'sha256':sha256(out),'rules':RULES,'rule_counts':dict(counts),
            'splits':{s:{'papers':sum(r['split']==s for r in refined),'limitation_sentences':sum(len(r['limitations']) for r in refined if r['split']==s)} for s in ['train','dev','test']},
            'target_pdf_used':False,'active_training_data_modified':False,
            'revision':2,'heading_prefixes_removed_from_xml_titles':True,
            'status':'Prepared candidate only; no training or held-out performance claim. Rules are still weak labels.',
            'limitations':['Sentence-only attribution can confuse prior methods with the current study.','Coverage drops: valid limitations without these phrases are omitted.','No independent expert review labels.']}
    atomic_json('outputs/limitation-refinement.json',report)
    atomic_json('outputs/limitation-refinement-train-sample.json',sample)
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()

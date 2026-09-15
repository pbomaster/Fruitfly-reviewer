"""Frozen-checkpoint evaluation, including one held-out full-PDF generation.

No target review is supplied to the model; no output repairs or replacement.
"""
import json,re,time,collections
from pathlib import Path
import torch
from coverage_pointer import CoveragePointer
from extractive_tasks import ExtractiveTasks,PROMPT
from train_coverage import generate,loss_batch
from resumable_checkpoint import Checkpoints,atomic_json,sha256

def overlap(a,b):
    a=collections.Counter(re.findall(r'\w+',a.lower())); b=collections.Counter(re.findall(r'\w+',b.lower()))
    n=sum((a&b).values()); return 2*n/max(1,sum(a.values())+sum(b.values()))

def main():
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    out=Path('outputs/coverage-v8'); m=CoveragePointer(); saved,info=Checkpoints(out/'best').load()
    m.load_state_dict(saved['model']); tasks=ExtractiveTasks(); start=time.monotonic()
    examples=[tasks.example(980000000+i,'test') for i in range(32)]
    for name,scale in [('heldout',1.),('heldout-graph-removed',0.)]:
        rows=[generate(m,tasks,e,graph_scale=scale) for e in examples]
        for row,e in zip(rows,examples):
            row['contribution_overlap']=overlap(row['raw_output'],e['contribution'])
            row['limitation_overlap']=overlap(row['raw_output'],e['limitation'])
        atomic_json(out/f'{name}-raw.json',rows)
    with torch.no_grad():
        scores=[loss_batch(m,tasks,examples[i:i+8]) for i in range(0,len(examples),8)]
    # No page/section or cue filtering. Every extracted page contributes all text.
    pages=json.loads(Path('outputs/paper-pages.json').read_text(encoding='utf8'))
    ids=[]; segments=[]
    for page,text in enumerate(pages,1):
        normalized=re.sub(r'\s+',' ',text).strip()
        for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z])',normalized):
            ids.append(1); start_id=len(ids); toks=tasks.tok.encode(sentence+' ').ids
            ids+=toks; segments.append({'page':page,'text':sentence,'start':start_id,'end':len(ids)})
    e={'source':'\n'.join(pages),'source_ids':ids,'prefix':[1]+tasks.tok.encode(PROMPT).ids,
       'query':PROMPT,'queried_name':'input/paper.pdf'}
    atomic_json(out/'full-paper-input.json',e); atomic_json(out/'full-paper-segments.json',segments)
    rows=[]
    for name,scale in [('full-paper',1.),('full-paper-graph-removed',0.)]:
        row=generate(m,tasks,e,graph_scale=scale); rows.append(row)
        atomic_json(out/f'{name}-raw.json',row)
        (out/f'{name}-raw.txt').write_text(row['raw_output'],encoding='utf8')
    summary={'checkpoint':info,'test_n':len(examples),'test_papers':len({e['queried_name'] for e in examples}),
       'test_token_nll':sum(r['token_nll']*r['tokens'] for r in scores)/sum(r['tokens'] for r in scores),
       'full_pdf_sha256':sha256('input/paper.pdf'),'full_pdf_pages':len(pages),'full_pdf_tokens_with_sentence_markers':len(ids),
       'raw_generation_preserved':True,'target_pdf_used_for_training':False,'raw_text_equals_decode':True,
       'elapsed_seconds':time.monotonic()-start,'factual_review_success':'requires manual source-grounded assessment; no automatic claim'}
    for name in ['heldout','heldout-graph-removed']:
        rr=json.loads((out/f'{name}-raw.json').read_text(encoding='utf8'))
        assert all(tasks.tok.decode(r['generated_ids'])==r['raw_output'] for r in rr)
        summary[name]={'exact':sum(r['exact'] for r in rr)/len(rr),'ended_eos':sum(r['ended_eos'] for r in rr)/len(rr),
                       'contribution_overlap':sum(r['contribution_overlap'] for r in rr)/len(rr),
                       'limitation_overlap':sum(r['limitation_overlap'] for r in rr)/len(rr)}
    assert all(tasks.tok.decode(r['generated_ids'])==r['raw_output'] for r in rows)
    atomic_json(out/'evaluation.json',summary); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()

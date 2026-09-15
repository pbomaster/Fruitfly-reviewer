"""Frozen v9 counterfactual input check. Changed counts are NOT paper facts."""
import json
from pathlib import Path
import torch
from coverage_pointer import CoveragePointer
from section_tasks import SectionTasks,encode_sentences
from section_document import document_sentences
from extractive_tasks import PROMPT
from train_sections import generate
from resumable_checkpoint import Checkpoints,atomic_json,sha256

def main():
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    out=Path('outputs/sections-v9'); m=CoveragePointer(); saved,info=Checkpoints(out/'best').load()
    m.load_state_dict(saved['model']); tasks=SectionTasks()
    pages=json.loads(Path('outputs/paper-pages-normalized.json').read_text(encoding='utf8'))
    replacements={'8,069':'8,123','138':'147','289':'302','71 female':'83 female'}
    changed=[]
    for text in pages:
        for old,new in replacements.items(): text=text.replace(old,new)
        changed.append(text)
    ids,_,source=encode_sentences(tasks.tok,document_sentences(changed))
    e={'source':source,'source_ids':ids,'query':PROMPT,'prefix':[1]+tasks.tok.encode(PROMPT).ids,
       'queried_name':'COUNTERFACTUAL test only; altered counts are not paper facts'}
    row=generate(m,tasks,e); atomic_json(out/'counterfactual-input.json',e)
    atomic_json(out/'counterfactual-raw.json',row)
    (out/'counterfactual-raw.txt').write_text(row['raw_output'],encoding='utf8')
    papers=[json.loads(line) for line in Path('work/science-corpus/papers.jsonl').read_text(encoding='utf8').splitlines()]
    target_doi='10.1016/j.cell.2026.08.015'
    training=[p for p in papers if p['split']=='train']
    summary={'checkpoint':info,'counterfactual_replacements':replacements,
             'altered_numbers_in_output':{s:s in row['raw_output'] for s in ['8,123','147','302','83']},
             'original_major_count_absent':'8,069' not in row['raw_output'],
             'training_max_year':max(p['year'] for p in training),'target_doi_in_corpus':any(target_doi in p['id'] for p in papers),
             'generated_raw_equals_decode':tasks.tok.decode(row['generated_ids'])==row['raw_output'],
             'paper_sha256':sha256('input/paper.pdf'),'scope':'Input sensitivity and source provenance checks, not a general reasoning benchmark.'}
    atomic_json(out/'grounding-check.json',summary); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()

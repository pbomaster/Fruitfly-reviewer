"""Fixed checkpoint, independent papers, full raw and normalized PDF inputs."""
import json
from pathlib import Path
import torch
from coverage_pointer import CoveragePointer
from section_tasks import SectionTasks,encode_sentences
from extractive_tasks import PROMPT
from section_document import document_sentences
from train_sections import generate
from resumable_checkpoint import Checkpoints,atomic_json,sha256

def main():
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    out=Path('outputs/sections-v9'); m=CoveragePointer(); saved,info=Checkpoints(out/'best').load()
    m.load_state_dict(saved['model']); tasks=SectionTasks()
    summary={'checkpoint':info,'target_pdf_trained':False,'test':{}}
    examples=[tasks.example(990000000+i,'test') for i in range(32)]
    for name,scale in [('heldout',1.),('heldout-graph-removed',0.)]:
        rows=[generate(m,tasks,e,graph_scale=scale) for e in examples]
        assert all(tasks.tok.decode(r['generated_ids'])==r['raw_output'] for r in rows)
        atomic_json(out/(name+'-raw.json'),rows)
        summary['test'][name]={'n':len(rows),'papers':len({r['queried_name'] for r in rows}),
          'exact':sum(r['exact'] for r in rows)/len(rows),'ended_eos':sum(r['ended_eos'] for r in rows)/len(rows)}
    for name,path in [('full-paper','outputs/paper-pages.json'),('normalized-paper','outputs/paper-pages-normalized.json')]:
        pages=json.loads(Path(path).read_text(encoding='utf8')); sentences=document_sentences(pages)
        ids,_,source=encode_sentences(tasks.tok,sentences)
        e={'source':source,'source_ids':ids,'query':PROMPT,'prefix':[1]+tasks.tok.encode(PROMPT).ids,'queried_name':'input/paper.pdf'}
        atomic_json(out/(name+'-input.json'),e); atomic_json(out/(name+'-segments.json'),sentences)
        row=generate(m,tasks,e)
        assert tasks.tok.decode(row['generated_ids'])==row['raw_output']
        atomic_json(out/(name+'-raw.json'),row); (out/(name+'-raw.txt')).write_text(row['raw_output'],encoding='utf8')
        summary[name]={'pages':len(pages),'tokens':len(ids),'input_sha256':sha256(path),
          'abstract_sentences':sum(s['section']=='abstract' for s in sentences),'all_text_preserved':True,'ended_eos':row['ended_eos']}
    summary['review_success']='Requires source-grounded assessment; no automatic success.'
    atomic_json(out/'evaluation.json',summary); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()

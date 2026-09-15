"""Separate input-normalization experiment; preserve the primary raw-PDF run."""
import json,re
from pathlib import Path
import torch
from sequential_pointer import SequentialPointer
from extractive_tasks import ExtractiveTasks,PROMPT
from train_transition import generate
from resumable_checkpoint import Checkpoints,atomic_json,sha256

def main():
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    out=Path('outputs/transition-v7'); m=SequentialPointer()
    saved,info=Checkpoints(out/'best').load(); m.load_state_dict(saved['model']); tasks=ExtractiveTasks()
    path=Path('outputs/paper-pages-normalized.json'); pages=json.loads(path.read_text(encoding='utf8'))
    ids=[]
    for page in pages:
        for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z])',re.sub(r'\s+',' ',page).strip()):
            ids.extend([1]+tasks.tok.encode(sentence+' ').ids)
    e={'source':'\n'.join(pages),'source_ids':ids,'prefix':[1]+tasks.tok.encode(PROMPT).ids,
       'query':PROMPT,'queried_name':'input/paper.pdf; conservative line-wrap repair'}
    atomic_json(out/'normalized-paper-input.json',e)
    row=generate(m,tasks,e)
    assert tasks.tok.decode(row['generated_ids'])==row['raw_output']
    atomic_json(out/'normalized-paper-raw.json',row)
    (out/'normalized-paper-raw.txt').write_text(row['raw_output'],encoding='utf8')
    atomic_json(out/'normalized-paper-evaluation.json',{'checkpoint':info,'pages':len(pages),'input_tokens':len(ids),
                'input_sha256':sha256(path),'output_rewritten':False,'scope':'Separate normalization comparison, not used to choose checkpoint.'})
    print(json.dumps({'input_tokens':len(ids),'raw_output':row['raw_output']}))
if __name__=='__main__': main()

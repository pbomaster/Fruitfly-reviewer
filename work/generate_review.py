"""Read a PDF and preserve the frozen MaleCNS model's unedited short paragraph."""
import argparse,collections,datetime,json,os,re
from pathlib import Path
from types import SimpleNamespace
import torch
from pypdf import PdfReader
from tokenizers import Tokenizer
from coverage_pointer import CoveragePointer
from section_tasks import encode_sentences
from section_document import document_sentences
from extractive_tasks import PROMPT
from train_sections import generate
from resumable_checkpoint import Checkpoints,atomic_json,sha256

def main():
    root=Path(__file__).resolve().parent.parent
    parser=argparse.ArgumentParser(); parser.add_argument('--pdf',default=str(root/'input/paper.pdf'))
    parser.add_argument('--out'); args=parser.parse_args(); os.chdir(root)
    pdf=Path(args.pdf).resolve()
    out=Path(args.out).resolve() if args.out else root/'outputs'/'review-runs'/datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError('This tested runner requires the configured CUDA environment.')
    extractor='pypdf'; extraction_warning=None
    try:
        pages=[(page.extract_text() or '').replace('\x00','') for page in PdfReader(pdf).pages]
    except Exception as error:
        from pypdf.errors import PdfReadError
        if not isinstance(error,PdfReadError): raise
        import fitz
        extractor='PyMuPDF'; extraction_warning=str(error)
        with fitz.open(pdf) as document: pages=[page.get_text().replace('\x00','') for page in document]
    atomic_json(out/'extracted-pages.json',pages)
    counts=json.loads(Path('work/training-word-counts.json').read_text(encoding='utf8')); changes=[]; repaired=[]
    pattern=re.compile(r'([A-Za-z][A-Za-z-]*)[ \t]*-[ \t]*\r?\n[ \t]*([a-z][A-Za-z-]*)')
    for page,s in enumerate(pages,1):
        def fix(m):
            candidates=[m.group(1)+m.group(2),m.group(1)+'-'+m.group(2)]
            supported=[v for v in candidates if counts.get(v.lower(),0)>=3]
            if not supported: return m.group(0)
            chosen=max(supported,key=lambda v:counts[v.lower()])
            changes.append({'page':page,'start':m.start(),'original':m.group(0),'replacement':chosen})
            return chosen
        repaired.append(pattern.sub(fix,s))
    atomic_json(out/'input-normalization.json',changes)
    sentences=document_sentences(repaired); tok=Tokenizer.from_file('work/science-corpus/tokenizer.json')
    ids,_,source=encode_sentences(tok,sentences)
    e={'source':source,'source_ids':ids,'query':PROMPT,'prefix':[1]+tok.encode(PROMPT).ids,'queried_name':str(pdf)}
    atomic_json(out/'model-input.json',e); atomic_json(out/'segments.json',sentences)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    model=CoveragePointer(); saved,info=Checkpoints('outputs/sections-v9/best').load(); model.load_state_dict(saved['model'])
    row=generate(model,SimpleNamespace(tok=tok),e)
    assert tok.decode(row['generated_ids'])==row['raw_output']
    atomic_json(out/'generation.json',row); (out/'review.txt').write_text(row['raw_output'],encoding='utf8')
    atomic_json(out/'run.json',{'pdf':str(pdf),'pdf_sha256':sha256(pdf),'pages':len(pages),'tokens':len(ids),
                 'extractor':extractor,'extraction_warning':extraction_warning,
                 'checkpoint':info,'tokenizer_sha256':sha256('work/science-corpus/tokenizer.json'),
                 'word_counts_sha256':sha256('work/training-word-counts.json'),'output_edited':False,
                 'scope':'Experimental extractive review paragraph. Reads extracted PDF text; figures are not interpreted.'})
    print(str(out)); print(json.dumps({'raw_output':row['raw_output'],'ended_eos':row['ended_eos']}))
if __name__=='__main__': main()

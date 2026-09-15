"""Preserve unedited autoregressive generations, ablations, and lexical diagnostics.

This evaluator never supplies a target review to the model and never declares
success from keyword hits or language-model loss.
"""
import argparse, hashlib, json, re, time
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from connectome_memory import ConnectomeMemoryLM
from train_connectome_memory import Corpus, evaluate, source_bag

def clean_pdf(s):
    s=re.sub(r'([A-Za-z])\s*-\s*\n\s*([a-z])',r'\1\2',s)
    return re.sub(r'\s+',' ',s).strip()

@torch.no_grad()
def generate(m,tok,source,sampling,seed=20260915,graph_scale=1.,limit=320):
    torch.manual_seed(seed)
    ids=tok.encode(source).ids
    bag,mask=source_bag(ids)
    mem=m.memory(torch.from_numpy(bag[None]).cuda())
    mask=torch.from_numpy(mask[None]).cuda()
    generated=[]; cache=None; current=torch.tensor([[1]],device='cuda')
    t=time.monotonic()
    for _ in range(limit):
        logits,cache=m(current,mem,mask,cache,graph_scale=graph_scale)
        logits=logits[0,-1].clone(); logits[0]=float('-inf'); logits[1]=float('-inf'); logits[3]=float('-inf')
        if sampling:
            logits/=0.8
            sorted_logits,index=logits.sort(descending=True)
            cum=sorted_logits.softmax(-1).cumsum(-1)
            remove=cum>0.9; remove[1:]=remove[:-1].clone(); remove[0]=False
            sorted_logits[remove]=float('-inf')
            token=index[torch.multinomial(sorted_logits.softmax(-1),1)].item()
        else: token=logits.argmax().item()
        generated.append(token)
        if token==2: break
        current=torch.tensor([[token]],device='cuda')
    return {'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'input_tokens':len(ids),
            'input_tokens_in_memory':len(ids),'memory_chunks':int(mask.sum().item()),
            'within_chunk_representation':'normalised learned token-embedding mean; no token order within chunk',
            'decode':'temperature_0.8_top_p_0.9' if sampling else 'greedy','seed':seed,
            'graph_scale':graph_scale,'max_new_tokens':limit,'generated_ids':generated,
            'stopped_on_eos':generated[-1]==2,'raw_output':tok.decode(generated),'seconds':time.monotonic()-t}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',default='outputs/connectome-memory-v1'); args=parser.parse_args()
    out=Path(args.run); tok=Tokenizer.from_file('work/science-corpus/tokenizer.json')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    m=ConnectomeMemoryLM(); ck=torch.load(out/'best.pt',weights_only=True); m.load_state_dict(ck['model']); m.eval()
    pages=json.loads(Path('outputs/paper-pages.json').read_text(encoding='utf8'))
    source='\n\n'.join(f'[Page {i+1}] '+clean_pdf(p) for i,p in enumerate(pages))
    paper2=clean_pdf(pages[1]); abstract=paper2[paper2.find('SUMMARY'):]
    # Use a held-out paper, not a human-made control review.
    heldout=next(json.loads(s) for s in Path('work/science-corpus/papers.jsonl').read_text(encoding='utf8').splitlines() if json.loads(s)['split']=='test')
    changed=re.sub(r'166\s*,?\s*700','120,000',source)
    assert changed!=source
    cases=[('full_pdf_greedy',source,False,1.),('full_pdf_sample',source,True,1.),
           ('abstract_greedy',abstract,False,1.),('empty_document','',False,1.),
           ('heldout_other_paper',heldout['source'],False,1.),
           ('changed_neuron_count',changed,False,1.),('graph_removed',source,False,0.)]
    trials=[]
    for name,text,sampling,scale in cases:
        r=generate(m,tok,text,sampling,graph_scale=scale); r['name']=name
        (out/(name+'-raw.txt')).write_text(r['raw_output'],encoding='utf8')
        (out/(name+'-source.txt')).write_text(text,encoding='utf8')
        trials.append(r)
        print(name,json.dumps(r,ensure_ascii=False),flush=True)
    # Exact-text overlap is a memorisation diagnostic, not a semantic metric.
    train_ngrams=set()
    for line in Path('work/science-corpus/papers.jsonl').read_text(encoding='utf8').splitlines():
        p=json.loads(line)
        if p['split']!='train': continue
        for t in p['targets']:
            words=re.findall(r'\w+',t.lower())
            train_ngrams.update(tuple(words[i:i+12]) for i in range(len(words)-11))
    for r in trials:
        words=re.findall(r'\w+',r['raw_output'].lower())
        grams=[tuple(words[i:i+12]) for i in range(len(words)-11)]
        r['training_review_12gram_matches']=sum(g in train_ngrams for g in grams)
        r['output_12grams']=len(grams)
    # Corrected document-swap ablation guarantees a different paper for each row.
    corpus=Corpus(); items=[]
    for i,p in enumerate(corpus.papers):
        if p['split']=='dev': items.extend((i,t) for t in p['targets'][:2])
    # Remove dev/test shared boilerplate from the FINAL test, retaining original
    # dataset and original training-run metrics so the correction is auditable.
    dev_targets={tuple(t) for p in corpus.papers if p['split']=='dev' for t in p['targets']}
    filtered_test=[]; removed=[]
    administrative=['as the editors have judged','changes in our revision policy',
                    'reviewers have discussed','thank you for submitting','reviewers have opted',
                    'reviewing editor has drafted','article has been reviewed','interests of transparency']
    for i,t in corpus.samples['test']:
        text=tok.decode(t).lower()
        reason='exact_dev_duplicate' if tuple(t) in dev_targets else 'administrative_boilerplate' if any(s in text for s in administrative) else None
        if reason: removed.append({'paper_id':corpus.papers[i]['id'],'reason':reason,'text':tok.decode(t)})
        else: filtered_test.append((i,t))
    clean_test=evaluate(m,corpus,filtered_test)
    normal=evaluate(m,corpus,items)
    paper_ids=sorted({i for i,_ in items}); original=dict(corpus.bags)
    for j,i in enumerate(paper_ids): corpus.bags[i]=original[paper_ids[(j+1)%len(paper_ids)]]
    swapped=evaluate(m,corpus,items)
    report={'checkpoint':str(out/'best.pt'),'checkpoint_sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),
            'selected_epoch':ck['epoch'],'pdf_sha256':hashlib.sha256(Path('input/paper.pdf').read_bytes()).hexdigest(),
            'pages_in_full_input':len(pages),'human_review_source':'Publisher eLife decision letters, publication <=2022',
            'heldout_other_paper':{'id':heldout['id'],'title':heldout['title']},
            'document_swap_ablation':{'correct':normal,'different_paper_for_every_row':swapped,'mismatched_fraction':1.0},
            'final_filtered_test':clean_test,'test_filter':{'original_paragraphs':len(corpus.samples['test']),'removed':removed},
            'trials':trials,'success':False,'status':'Needs factual human/assistant assessment; lexical change alone is insufficient.'}
    (out/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')

if __name__=='__main__': main()

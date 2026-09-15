"""Train from scratch on publisher human reviews; select only by heldout papers."""
import argparse, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from connectome_memory import ConnectomeMemoryLM, detach

def source_bag(ids,vocab=4096,chunks=64):
    # Adaptive chunks include ALL source tokens, including long PDFs.
    a=np.asarray(ids,dtype=np.int64)
    if len(a)==0: a=np.array([0])
    c=min(chunks,max(1,math.ceil(len(a)/256)))
    bins=np.minimum(np.arange(len(a))*c//len(a),c-1)
    bag=np.bincount(bins*vocab+a,minlength=chunks*vocab).reshape(chunks,vocab).astype(np.float32)
    count=bag.sum(-1,keepdims=True)
    bag/=np.maximum(count,1)
    return bag,(count[:,0]>0)

class Corpus:
    def __init__(self):
        self.papers=json.loads(Path('work/science-corpus/encoded.json').read_text())
        self.samples={s:[] for s in ['train','dev','test']}
        self.bags={}
        for i,p in enumerate(self.papers):
            self.bags[i]=source_bag(p['source'])
            for target in p['targets']: self.samples[p['split']].append((i,target))
    def batch(self,items,source_mode='correct'):
        ids=[i for i,_ in items]
        if source_mode=='shuffled': ids=ids[1:]+ids[:1]
        bags=torch.from_numpy(np.stack([self.bags[i][0] for i in ids])).cuda()
        mask=torch.from_numpy(np.stack([self.bags[i][1] for i in ids])).cuda()
        if source_mode=='empty': bags.zero_(); mask[:]=False; mask[:,0]=True
        length=max(len(t) for _,t in items)
        x=torch.zeros(len(items),length,dtype=torch.long,device='cuda')
        y=torch.full_like(x,-100)
        for j,(_,t) in enumerate(items):
            x[j,:len(t)]=torch.tensor([1]+t[:-1],device='cuda')
            y[j,:len(t)]=torch.tensor(t,device='cuda')
        return bags,mask,x,y

@torch.no_grad()
def evaluate(m,corpus,items,batch=32,source_mode='correct',graph_scale=1.):
    m.eval(); loss=0.; count=0
    # Stable length ordering bounds padding; source shuffle is across papers below.
    for j in range(0,len(items),batch):
        bags,mask,x,y=corpus.batch(items[j:j+batch],source_mode)
        mem=m.memory(bags); cache=None
        for k in range(0,x.shape[1],64):
            logits,cache=m(x[:,k:k+64],mem,mask,cache,graph_scale=graph_scale)
            target=y[:,k:k+64]
            loss+=F.cross_entropy(logits.flatten(0,1),target.flatten(),reduction='sum').item()
            count+=(target!=-100).sum().item()
    return {'nll':loss/count,'tokens':count,'paragraphs':len(items)}

def main():
    a=argparse.ArgumentParser(); a.add_argument('--epochs',type=int,default=4); a.add_argument('--batch',type=int,default=32)
    a.add_argument('--out',default='outputs/connectome-memory-v1'); a.add_argument('--resume'); args=a.parse_args()
    out=Path(args.out); out.mkdir(exist_ok=True)
    seed=20260915; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    gpu=json.loads(Path('outputs/gpu-check.json').read_text()); assert gpu['sparse_forward_ok'] and gpu['sparse_input_gradient_ok']
    start=time.monotonic(); c=Corpus(); m=ConnectomeMemoryLM()
    opt=torch.optim.AdamW(m.parameters(),lr=0.0005,weight_decay=.05)
    run={'seed':seed,'args':vars(args),'gpu':gpu,'initialisation':'All language parameters random; only graph topology, signed weights and sensory node indices copied.',
         'trainable_parameters':sum(p.numel() for p in m.parameters()),'epochs':[],'steps':0,
         'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('work/connectome_memory.py')]},
         'training_tokens':0,'success':False}
    first_epoch=1; best=float('inf'); stale=0
    if args.resume:
        ck=torch.load(args.resume,weights_only=True); m.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        first_epoch=ck['epoch']+1; best=ck['dev_nll']
    # Two independently held-out paragraphs per paper for checkpoint selection.
    dev=[]
    for i,p in enumerate(c.papers):
        if p['split']=='dev': dev.extend((i,t) for t in p['targets'][:2])
    baseline=evaluate(m,c,dev,args.batch); run['initial_dev']=baseline
    print('initial_dev',json.dumps(baseline),flush=True)
    for epoch in range(first_epoch,args.epochs+1):
        items=list(c.samples['train']); random.shuffle(items)
        # Bucket within random blocks; don't group all samples of one paper.
        ordered=[]
        for j in range(0,len(items),args.batch*30): ordered.extend(sorted(items[j:j+args.batch*30],key=lambda it:len(it[1])))
        epoch_loss=0.; epoch_tokens=0; epoch_start=time.monotonic()
        for j in range(0,len(ordered),args.batch):
            m.train(); bags,mask,x,y=c.batch(ordered[j:j+args.batch]); cache=None
            for k in range(0,x.shape[1],64):
                opt.zero_grad(set_to_none=True)
                mem=m.memory(bags)
                logits,cache=m(x[:,k:k+64],mem,mask,cache)
                target=y[:,k:k+64]; nt=(target!=-100).sum().item()
                if nt==0: break
                loss=F.cross_entropy(logits.flatten(0,1),target.flatten())
                if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
                loss.backward(); grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                if not torch.isfinite(grad): raise RuntimeError('non-finite gradient')
                opt.step(); cache=detach(cache)
                epoch_loss+=loss.item()*nt; epoch_tokens+=nt; run['steps']+=1; run['training_tokens']+=nt
            if j//args.batch%25==0:
                progress={'epoch':epoch,'paragraphs':min(j+args.batch,len(ordered)),'total_paragraphs':len(ordered),
                          'train_nll':epoch_loss/max(1,epoch_tokens),'steps':run['steps'],'training_tokens':run['training_tokens'],
                          'seconds':time.monotonic()-start}
                print(json.dumps(progress),flush=True)
                (out/'progress.json').write_text(json.dumps(progress,indent=2))
        dev_score=evaluate(m,c,dev,args.batch)
        improved=dev_score['nll']<best
        row={'epoch':epoch,'train_nll':epoch_loss/epoch_tokens,'dev':dev_score,'seconds':time.monotonic()-epoch_start,'selected':improved}
        run['epochs'].append(row); print('epoch',json.dumps(row),flush=True)
        ck={'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'dev_nll':dev_score['nll']}
        torch.save(ck,out/'last.pt')
        if improved:
            best=dev_score['nll']; stale=0; torch.save(ck,out/'best.pt')
        else: stale+=1
        run['seconds']=time.monotonic()-start
        (out/'training.json').write_text(json.dumps(run,indent=2))
        if stale>=2: break
    m.load_state_dict(torch.load(out/'best.pt',weights_only=True)['model'])
    # Full held-out test is used only after selection, never as a training objective.
    run['test']=evaluate(m,c,c.samples['test'],args.batch)
    run['dev_correct']=evaluate(m,c,dev,args.batch)
    run['dev_shuffled_document']=evaluate(m,c,dev,args.batch,source_mode='shuffled')
    run['dev_empty_document']=evaluate(m,c,dev,args.batch,source_mode='empty')
    run['dev_graph_removed']=evaluate(m,c,dev,args.batch,graph_scale=0.)
    run['seconds']=time.monotonic()-start
    run['interpretation']='NLL is language modelling evidence, not review success. Raw generation and factual assessment are required.'
    (out/'training.json').write_text(json.dumps(run,indent=2))
    print('complete',json.dumps(run),flush=True)

if __name__=='__main__': main()

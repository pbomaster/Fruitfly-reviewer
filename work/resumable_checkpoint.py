"""Two-slot atomic checkpoints with checksums and explicit RNG restoration.

Files are written beside their destination, flushed/fsynced, then replaced.
A manifest is committed last. Abrupt power loss may lose the newest interval;
neither Python fsync nor this scheme guarantees hardware write-cache durability.
"""
import hashlib, json, os, random, time
from pathlib import Path
import torch

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    path=Path(path); tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def rng_state():
    return {'python':random.getstate(),'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python']); torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']: torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])

class Checkpoints:
    def __init__(self,directory):
        self.root=Path(directory); self.root.mkdir(parents=True,exist_ok=True)
    def valid_entries(self):
        entries=[]
        for slot in ['a','b']:
            meta=self.root/f'checkpoint-{slot}.json'
            if not meta.exists(): continue
            try:
                info=json.loads(meta.read_text(encoding='utf8'))
                path=self.root/f'checkpoint-{slot}.pt'
                if path.is_file() and sha256(path)==info['sha256']: entries.append(info)
            except (OSError,ValueError,KeyError): pass
        return sorted(entries,key=lambda x:x['step'],reverse=True)
    def save(self,payload):
        entries=self.valid_entries()
        slot='b' if entries and entries[0]['slot']=='a' else 'a'
        path=self.root/f'checkpoint-{slot}.pt'; tmp=path.with_name(path.name+'.tmp')
        if torch.cuda.is_available(): torch.cuda.synchronize()
        with tmp.open('wb') as f:
            torch.save(payload,f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
        info={'slot':slot,'step':payload['step'],'sha256':sha256(path),'bytes':path.stat().st_size,
              'saved_unix':time.time(),'file':str(path)}
        atomic_json(self.root/f'checkpoint-{slot}.json',info)
        atomic_json(self.root/'latest.json',info)
        return info
    def load(self):
        failures=[]
        for info in self.valid_entries():
            try:
                p=torch.load(self.root/f"checkpoint-{info['slot']}.pt",map_location='cpu',weights_only=True)
                if p['step']!=info['step']: raise ValueError('step mismatch')
                return p,info
            except Exception as e: failures.append(str(e))
        raise RuntimeError('No intact checkpoint available. '+str(failures))

"""Fetch the pinned upstream artifact used ONLY for connectome tensors."""
import hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download

REVISION='65c677b3d566a2e9793d5f72999cdb441c6c0a9f'
EXPECTED='355f06c44d14e38af50e9c801f51839c37a0a56ac4ca016da2be9b3ae6f215ad'

def main():
    dest=Path(__file__).resolve().parent/'work/fly-model'
    path=Path(hf_hub_download(repo_id='ngxson/fly-llm-hf',filename='model.safetensors',
                            revision=REVISION,local_dir=dest))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=EXPECTED:
        raise RuntimeError(f'Graph artifact checksum mismatch: {digest}')
    print(f'Graph artifact verified: {path}')

if __name__=='__main__': main()


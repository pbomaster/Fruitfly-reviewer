"""Download publisher XML as data only; never execute repository content."""
import concurrent.futures, hashlib, json, re, time, urllib.request
from pathlib import Path

ROOT = Path('work/elife')
ROOT.mkdir(exist_ok=True)

def fetch(url):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'connectome-review-research'}), timeout=60).read()
        except Exception:
            if attempt == 3: raise
            time.sleep(attempt + 1)

def main():
    treefile = ROOT / 'tree.json'
    if not treefile.exists():
        treefile.write_bytes(fetch('https://api.github.com/repos/elifesciences/elife-article-xml/git/trees/0c5058c62b6b4985c9fa0ac3c773ba8107350a2e?recursive=1'))
    tree = json.loads(treefile.read_bytes())
    latest = {}
    for item in tree['tree']:
        m = re.fullmatch(r'articles/elife-(\d+)-v(\d+)\.xml', item['path'])
        if m and 40000 <= int(m[1]) <= 75000:
            key = m[1]
            if key not in latest or int(m[2]) > latest[key][0]:
                latest[key] = (int(m[2]), item['path'])
    paths = [v[1] for _,v in sorted(latest.items(), key=lambda kv:hashlib.sha256(kv[0].encode()).hexdigest())][:1600]
    print('revision',tree['sha'],'truncated',tree.get('truncated'),'selected',len(paths),flush=True)
    def download(path):
        url = f"https://raw.githubusercontent.com/elifesciences/elife-article-xml/{tree['sha']}/{path}"
        dest = ROOT/Path(path).name
        if not dest.exists(): dest.write_bytes(fetch(url))
        raw = dest.read_bytes()
        return {'path':path,'url':url,'file':str(dest),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    manifest=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for item in pool.map(download, paths):
            manifest.append(item)
            if len(manifest)%100==0: print('downloaded',len(manifest),flush=True)
    (ROOT/'manifest.json').write_text(json.dumps({'revision':tree['sha'],'files':manifest},indent=2),encoding='utf8')
    print('complete',len(manifest),flush=True)

if __name__=='__main__': main()

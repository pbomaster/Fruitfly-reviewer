"""Check document isolation, label origin, and full-length memory coverage."""
import hashlib,json,re
from pathlib import Path
import numpy as np
from train_connectome_memory import source_bag

papers=[json.loads(l) for l in Path('work/science-corpus/papers.jsonl').read_text(encoding='utf8').splitlines()]
by_split={s:{p['id'] for p in papers if p['split']==s} for s in ['train','dev','test']}
overlap={a+'_'+b:sorted(by_split[a]&by_split[b]) for a,b in [('train','dev'),('train','test'),('dev','test')]}
target='Sexual dimorphism in the complete Drosophila male central nervous system connectome'
matches=[p['id'] for p in papers if target.lower() in p['source'].lower() or '10.1016/j.cell.2026.08.015' in p['source'].lower()]
# The final token must still affect the last source chunk beyond 127,003 tokens.
long_source=[4]*130000+[4000]
bag,mask=source_bag(long_source)
coverage={'length':len(long_source),'chunks':int(mask.sum()),'last_token_mass':float(bag[-1,4000]),
          'last_token_present':bool(bag[-1,4000]>0),'all_nonempty_rows_normalised':bool(np.allclose(bag[mask].sum(-1),1.))}
assert coverage['last_token_present'] and coverage['all_nonempty_rows_normalised']
assert not any(overlap.values()) and not matches and max(p['year'] for p in papers)<=2022
train_hashes={hashlib.sha256(t.encode()).hexdigest() for p in papers if p['split']=='train' for t in p['targets']}
held_hashes={hashlib.sha256(t.encode()).hexdigest() for p in papers if p['split']!='train' for t in p['targets']}
assert not train_hashes&held_hashes
dev_hashes={hashlib.sha256(t.encode()).hexdigest() for p in papers if p['split']=='dev' for t in p['targets']}
test_hashes={hashlib.sha256(t.encode()).hexdigest() for p in papers if p['split']=='test' for t in p['targets']}
result={'paper_split_overlap':overlap,'target_title_or_doi_matches':matches,'max_publication_year':max(p['year'] for p in papers),
        'exact_review_paragraph_overlap_train_heldout':len(train_hashes&held_hashes),'long_document_coverage':coverage,
        'dev_test_exact_paragraph_overlap':len(dev_hashes&test_hashes),
        'qualification':'No train-heldout exact identity/text overlap found. Dev/test contain shared editorial boilerplate; final evaluator removes it. Near-duplicates are not comprehensively excluded.'}
Path('outputs/corpus-audit.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

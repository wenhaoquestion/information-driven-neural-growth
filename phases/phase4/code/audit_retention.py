"""Reconstruct saved paired risks from raw finite input-label counts."""
from pathlib import Path
import json, hashlib
import numpy as np
root=Path(__file__).resolve().parents[1]
a=np.load(root/'results/retention_counts.npz')
params=json.loads((root/'results/retention_execution.json').read_text())['analytic_checks']
lookup={x['seed']:x for x in params}
y=np.tile([0,1],4)
maxerr=0.; records=0
for par in params:
 seed=par['seed'];prev=np.zeros((1000,8),dtype='int64')
 for n in [128,512,2048,8192,32768,131072]:
  c=a[f'R1_fresh_{seed}_{n}'];assert np.all(c>=prev) and np.all(c.sum(axis=1)==n);prev=c
for line in (root/'results/retention_raw.jsonl').open():
 row=json.loads(line);records+=1
 seed,j,n=row['seed'],row['replicate'],row['n']
 if row['experiment']=='R1' and row['method']=='direct':
  h=np.array(lookup[seed]['h']);p=np.repeat((1-h)/2,2);q=np.repeat((1+h)/2,2)
  per=(y-p)**2-(y-q)**2
  calc=float(np.dot(a[f'R1_fresh_{seed}_{n}'][j],per)/n)
  maxerr=max(maxerr,abs(calc-row['estimate']))
 if row['experiment']=='R2':
  m=row['m'];hist=a[f'R2_hist_{seed}'][j]
  h=.4*np.where(hist[1::2]>=hist[::2],1.,-1.)
  assert abs(float(np.mean(h*0)))==0
  p=np.repeat((1-h)/2,2);q=np.repeat((1+h)/2,2);yt=np.tile([0,1],m)
  raw=a[f'R2_fresh_{seed}'][j] if row['method']=='fresh_audit' else hist
  calc=float(np.dot(raw,(yt-p)**2-(yt-q)**2)/n)
  maxerr=max(maxerr,abs(calc-row['estimate']))
 assert row['accepted']==(row['estimate']>row['radius'])
assert maxerr<1e-12
report={'records_checked':records,'max_direct_squared_loss_reconstruction_error':maxerr,
 'nested_counts_checked':len(params)*6,'all_acceptance_comparisons_reconstructed':True,
 'scope':'Every saved base record decision; all direct and R2 risk estimates independently reconstructed from squared losses and raw counts; not empirical coverage proof.',
 'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'results').glob('retention_*') if p.is_file()}}
(root/'results/retention_audit.json').write_text(json.dumps(report,indent=2))
print({k:v for k,v in report.items() if k!='files'})

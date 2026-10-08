"""Post hoc fair full-pool and noise-aware-projection comparators."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize_scalar
from retention_experiment import radius, wilson

root=Path(__file__).resolve().parents[1]
archive=np.load(root/'results/retention_counts.npz')
checks=json.loads((root/'results/retention_execution.json').read_text())['analytic_checks']
states=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
f1,f2=states.T
ys=np.tile([0,1],4)
records=[]
summary=[]
for check in checks:
    seed,z=check['seed'],check['z']
    old=archive[f'R1_hist_{seed}']
    N0=int(old[0].sum())
    h=np.array(check['h'])
    coeff=.2*np.sqrt(1-z)
    oldmean=old @ (np.repeat(f1,2)*(2*ys-1))/N0
    e=np.sqrt(2*np.log(2/.025)/N0)
    score=np.repeat(h,2)*(2*ys-1)
    for n in [128,512,2048,8192,32768,131072]:
        fresh=archive[f'R1_fresh_{seed}_{n}']
        def rad(a):
            r=h-a*f1
            return abs(a)*e+radius(n,float(np.mean(r*r)),float(np.max(np.abs(r))),.025)
        if coeff>0:
            opt=minimize_scalar(rad,bounds=(0,coeff),method='bounded')
            a=min([0.,coeff,float(opt.x)],key=rad)
        else:a=0.
        residual=h-a*f1
        estimates={
            'full_pool':((old+fresh)@score/(N0+n),radius(N0+n,.04,np.max(np.abs(h)),.05)),
            'noise_aware_moment':(a*oldmean+fresh@(np.repeat(residual,2)*(2*ys-1))/n,rad(a))}
        for method,(est,r) in estimates.items():
            accept=est>r;k=int(accept.sum());reps=len(est)
            summary.append(dict(experiment='R1',alternative=check['alternative'],z=z,n=n,method=method,
                replicates=reps,accepted=k,rate=k/reps,wilson95=wilson(k,reps),radius=float(r),
                unique_labels=N0+n,coefficient=float(a) if method=='noise_aware_moment' else None))
            for j in range(reps):records.append(dict(seed=seed,replicate=j,alternative=check['alternative'],z=z,
                n=n,method=method,estimate=float(est[j]),radius=float(r),accepted=bool(accept[j]),
                unique_labels=N0+n))
(root/'results/retention_supplement_summary.json').write_text(json.dumps(summary,indent=2))
(root/'results/retention_supplement_raw.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n')
print(json.dumps({'records':len(records),'groups':len(summary)}))

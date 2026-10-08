"""Regenerate auxiliary plots and numerical proof checks from saved runs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'results/complementarity_runs.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
colors=['#366899','#bd5e39']
for block,color in zip([1,2],colors):
    chosen=[r for r in rows if r['signal'] and r['epsilon']==0.1 and r['max_block']==block]
    risks=np.array([[e['risk'] for e in r['events']] for r in chosen])
    widths=np.array([[e['width'] for e in r['events']] for r in chosen])
    xx=np.arange(1,11)
    mean=risks.mean(0); err=t.ppf(.975,len(chosen)-1)*risks.std(0,ddof=1)/np.sqrt(len(chosen))
    axes[0].plot(xx,mean,color=color,label=f'At most {block} features/action')
    axes[0].fill_between(xx,mean-err,mean+err,color=color,alpha=.16)
    axes[1].plot(xx,widths.mean(0),color=color)
axes[0].axhline(.25,ls=':',color='gray',label='Bayes risk')
axes[0].set(ylabel='Exact population mean squared error',xlabel='Structural opportunity',title='Finite-sample heuristic gates')
axes[0].legend(fontsize=8)
axes[1].set(ylabel='Mean active feature count',xlabel='Structural opportunity',title='Repeated structural decisions')
eps=np.geomspace(.01,.8,250)
axes[2].plot(eps,eps**2/(1+eps**2),color=colors[0],label='Best singleton gain')
axes[2].plot(eps,np.ones_like(eps),color=colors[1],label='Pair gain')
axes[2].axhline(.03,color='gray',ls='--',label='Singleton capacity charge')
axes[2].axhline(.06,color='gray',ls=':',label='Pair capacity charge')
axes[2].set(xscale='log',yscale='log',xlabel='Mixing coefficient epsilon',ylabel='Population risk reduction',title='Analytic one-component diagnostic')
axes[2].legend(fontsize=8)
(ROOT/'figures').mkdir(exist_ok=True)
fig.savefig(ROOT/'figures/complementarity_diagnostic.png',dpi=180)
plt.close(fig)

checks=[]
for epsilon in [0.01,0.1,0.25,0.8]:
    gram=np.array([[1+epsilon**2,1-epsilon**2],[1-epsilon**2,1+epsilon**2]])
    c=np.array([epsilon,-epsilon])
    weights=np.linalg.solve(gram,c)
    pair_gain=float(c@weights)
    singleton=epsilon**2/(1+epsilon**2)
    checks.append(dict(epsilon=epsilon,singleton_gain=singleton,pair_gain=pair_gain,
        pair_l1=float(abs(weights).sum()),pair_l2_sq=float(weights@weights),
        normalized_min_eigenvalue=float(np.linalg.eigvalsh(gram/(1+epsilon**2))[0]),
        initial_submodularity_ratio=2*singleton))
    assert abs(pair_gain-1)<1e-10
    assert abs(weights@weights-1/(2*epsilon**2))<1e-7
    assert abs(checks[-1]['normalized_min_eigenvalue']-2*singleton)<1e-12
(ROOT/'results/complementarity_analytic_checks.json').write_text(json.dumps(checks,indent=2))
paired=[]
for epsilon in [0.1,0.25]:
    by={(r['seed'],r['max_block']):r for r in rows if r['signal'] and r['epsilon']==epsilon}
    d=np.array([by[s,2]['risk']-by[s,1]['risk'] for s in range(93000,93032)])
    half=float(t.ppf(.975,31)*d.std(ddof=1)/np.sqrt(32))
    paired.append(dict(epsilon=epsilon,pair_minus_single_mse=float(d.mean()),
        descriptive_95_t_interval=[float(d.mean()-half),float(d.mean()+half)]))
(ROOT/'results/complementarity_paired.json').write_text(json.dumps(paired,indent=2))
print(json.dumps(paired,indent=2))

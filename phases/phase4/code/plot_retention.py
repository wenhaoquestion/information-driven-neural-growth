"""Create figures directly from frozen and separately labelled post hoc results."""
from pathlib import Path
import os,json
root=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(root/'tmp/mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rows=json.loads((root/'results/retention_summary.json').read_text())
extra=json.loads((root/'results/retention_supplement_summary.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'savefig.bbox':'tight'})
colors={'direct':'#485b70','oracle_moment':'#15877d','finite_history':'#c57524',
        'full_pool':'#7651a8','noise_aware_moment':'#bf4b66'}
labels={'direct':'Fresh paired bound','oracle_moment':'Exact retained moment (oracle)',
        'finite_history':'Finite retained moment (32,768 labels)',
        'full_pool':'Full old + fresh pool (post hoc)',
        'noise_aware_moment':'Noise-aware coefficient (post hoc)'}
fig,ax=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
for method in colors:
    group=sorted([r for r in rows+extra if r['experiment']=='R1' and r['alternative']
                  and r['z']==1/64 and r['method']==method],key=lambda x:x['n'])
    x=np.array([r['n'] for r in group]); y=np.array([r['rate'] for r in group])
    ax[0].plot(x,y,'o-',ms=4,color=colors[method],label=labels[method],
               linestyle='--' if method in ['full_pool','noise_aware_moment'] else '-')
    ax[0].fill_between(x,[r['wilson95'][0] for r in group],[r['wilson95'][1] for r in group],
                       color=colors[method],alpha=.10)
ax[0].set(xscale='log',ylim=(-.02,1.02),xlabel='Fresh labels',ylabel='Acceptance probability',
          title='Fixed gain 0.003; residual movement = total / 64')
ax[0].legend(fontsize=7.7,loc='lower right')
for method in ['direct','oracle_moment','finite_history','full_pool','noise_aware_moment']:
    group=sorted([r for r in rows+extra if r['experiment']=='R1' and r['alternative']
                  and r['n']==8192 and r['method']==method],key=lambda x:x['z'])
    ax[1].plot([r['z'] for r in group],[r['rate'] for r in group],'o-',ms=4,
               color=colors[method],linestyle='--' if method in ['full_pool','noise_aware_moment'] else '-')
ax[1].set(xscale='log',ylim=(-.02,1.02),xlabel='Residual / total squared movement',
          ylabel='Acceptance probability',title='8,192 fresh labels; all gains = 0.003')
(root/'figures').mkdir(exist_ok=True)
fig.savefig(root/'figures/retention_geometry.pdf');fig.savefig(root/'figures/retention_geometry.png',dpi=160)
plt.close(fig)
fig,ax=plt.subplots(figsize=(7.5,4),layout='constrained')
for method,color,label in [('naive_reuse','#b34d42','Naive fixed-pair reuse'),
                          ('uniform_moments','#15877d','Simultaneous moment region'),
                          ('fresh_audit','#485b70','Fresh independent audit')]:
    g=sorted([r for r in rows if r['experiment']=='R2' and r['method']==method],key=lambda r:r['m'])
    x=np.array([r['m'] for r in g]); y=np.array([r['rate'] for r in g])
    lo=np.array([r['wilson95'][0] for r in g]);hi=np.array([r['wilson95'][1] for r in g])
    ax.errorbar(x,y,yerr=np.maximum(0,np.vstack([y-lo,hi-y])),fmt='o-',capsize=3,color=color,label=label)
ax.axhline(.05,color='black',ls=':',lw=1,label='Nominal single-decision level')
ax.set(xscale='log',ylim=(-.03,1.03),xlabel='Input states / adaptive sign choices',
       ylabel='False acceptance probability',title='Null label law; candidate selected from 4,096 old labels')
ax.legend(fontsize=9)
fig.savefig(root/'figures/adaptive_reuse.pdf');fig.savefig(root/'figures/adaptive_reuse.png',dpi=160)
plt.close(fig)
print('Created retention_geometry and adaptive_reuse PDF/PNG from saved results.')

"""Standalone research figures from saved CSV rows; no recomputation of costs."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'logs'/'matplotlib_cache')
os.environ['OMP_NUM_THREADS']='1'
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelcolor':'#253442','text.color':'#253442',
                     'axes.edgecolor':'#8A939A','grid.color':'#DDE2E5','grid.linewidth':.6})
blue='#1769A8'; orange='#C26428'; gray='#565D64';gold='#AA8825'

def read(name):
    with (ROOT/'results'/name).open() as f:
        return list(csv.DictReader(f))

def values(rows,col):return np.array([float(a[col]) for a in rows])

def finish(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=190,bbox_inches='tight')
    fig.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)

rows=read('endpoint_polynomial.csv'); r=values(rows,'curvature_degree')
fig,axs=plt.subplots(1,3,figsize=(13.6,4.4),constrained_layout=True)
ax=axs[0]
ax.loglog(r,values(rows,'rho_det'),'o-',color=blue,label='Deterministic optimum')
ax.loglog(r,values(rows,'stochastic_lower'),'s--',color=orange,label='Stochastic convex lower bound')
ax.fill_between(r,values(rows,'stochastic_lower'),values(rows,'rho_det'),color=gold,alpha=.18,label='Numerical lower–upper bracket')
ax.set(xlabel='Curvature degree r = 2n',ylabel='Excess-risk hierarchy price',title='Unequal source weights')
ax.legend(fontsize=8.6,loc='upper left');ax.grid(True,which='major')
ax=axs[1]
ax.semilogx(r,values(rows,'rho_times_delta'),'o-',color=blue,label='Deterministic')
ax.semilogx(r,values(rows,'stoch_lower_times_delta'),'s--',color=orange,label='Stochastic lower bound')
ax.axhline(.25,color=gray,lw=1,ls=':',label='1/4 reference')
ax.set(xlabel='Curvature degree r = 2n',ylabel='Price × spacing parameter δ',title='Finite-degree scaling check',ylim=(0,.29))
ax.legend(fontsize=8.6,loc='lower right');ax.grid(True)
ax=axs[2]
ax.loglog(r,values(rows,'opt2_normalized'),'o-',color=blue,label='Flat OPT₂')
ax.loglog(r,values(rows,'opt3_normalized'),'s--',color=orange,label='Flat OPT₃')
ax.set(xlabel='Curvature degree r = 2n',ylabel='Absolute excess distortion',title='Normalization: 0 < φ″ ≤ 1')
ax.legend(fontsize=8.6);ax.grid(True)
fig.suptitle('Endpoint polynomial construction: finite verification, not a neural training result',fontsize=13)
fig.text(.5,-.14,'Seven constructed instances; all 18 deterministic hierarchies enumerated. Bounds use numerical costs, not interval arithmetic.\nδ = (4 log n / n)²; points (0, 2δ, 1−2δ, 1); masses (1/2−δ², δ², δ², 1/2−δ²).',ha='center',fontsize=9,color=gray)
finish(fig,'endpoint_construction')

rows=read('finite_degree.csv')
fig,axs=plt.subplots(1,2,figsize=(11.4,4.8),constrained_layout=True)
labels={.125:('0.125 × εᵥ₁',gray,'v'),.25:('0.25 × εᵥ₁',orange,'s'),.5:('0.5 × εᵥ₁',gold,'^'),1.:('εᵥ₁ = 16 log n / n',blue,'o')}
for f,(label,color,marker) in labels.items():
    rr=[a for a in rows if abs(float(a['epsilon_fraction_of_v1'])-f)<1e-8]
    axs[0].loglog(values(rr,'curvature_degree'),values(rr,'rho_det'),marker=marker,color=color,label=label)
fixed=[a for a in rows if abs(float(a['epsilon'])-1/16)<1e-14]
axs[0].loglog(values(fixed,'curvature_degree'),values(fixed,'rho_det'),'x:',color='#747E31',label='Fixed ε = 1/16')
axs[0].set(xlabel='Curvature degree r = 4n',ylabel='Deterministic hierarchy price',title='Equal masses: width matters at finite degree')
axs[0].legend(fontsize=8.7);axs[0].grid(True,which='major')
rr=[a for a in rows if a['v1_sequence']=='True']
r=values(rr,'curvature_degree')
axs[1].plot(r,values(rr,'rho_det'),'o-',color=blue,label='Deterministic optimum')
axs[1].plot(r,values(rr,'stochastic_lower'),'s--',color=orange,label='Stochastic lower bound')
axs[1].plot(r,values(rr,'stochastic_feasible_upper'),'^--',color=gold,label='Feasible stochastic upper bound')
axs[1].plot(r,values(rr,'leading_n_over_32logn'),':',color=gray,label='Leading n / (32 log n)')
axs[1].set(xlabel='Curvature degree r = 4n',ylabel='Hierarchy price',title='Original v1 sequence: three admissible points',ylim=(0,32))
axs[1].legend(fontsize=8.7);axs[1].grid(True)
fig.text(.5,-.14,'28 specified degree/width combinations; only ε ≤ 1/16 shown. These sampled curves are not optimized degree envelopes.\nThe v1 sequence first meets ε ≤ 1/16 at n = 1938 (r = 7752). Smaller-width controls are distinct generator families.',ha='center',fontsize=9,color=gray)
finish(fig,'equal_mass_degree_width')
print('Wrote four standalone figure artifacts (PNG and PDF).')

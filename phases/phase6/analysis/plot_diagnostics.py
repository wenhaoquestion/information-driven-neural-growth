"""Post-analysis descriptive views of all primary seeds; no tests or selection."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

root=Path(__file__).resolve().parent/'confirmation'
read=lambda name:list(csv.DictReader((root/name).open()))
auc=read('seed_auc.csv'); points=read('point_contrasts.csv')
methods=['historical_moment','random_growth128','random_growth138','fixed7']
labels=['Historical moment','Random R128','Random R138','Fixed width 7']
colors=['#285f89','#d48b25','#b64f7e','#658240']
fig,ax=plt.subplots(1,2,figsize=(11.4,4.4),layout='constrained')
for method,label,color in zip(methods,labels,colors):
    vals=np.sort([float(r['auc']) for r in auc if r['group']=='indefinite_noisy_global' and r['method']==method])
    assert len(vals)==256
    ax[0].step(vals,np.arange(1,257)/256,where='post',label=label,color=color,lw=1.8)
ax[0].set(xscale='log',xlabel='Seed-level AUC (log scale; all 256 seeds)',ylabel='Empirical cumulative proportion',title='Typical outcomes and the fixed-width tail')
ax[0].legend(fontsize=8,loc='lower right');ax[0].grid(alpha=.2)
for comp,label,color in zip(methods[1:3],labels[1:3],colors[1:3]):
    r=sorted([r for r in points if r['group']=='indefinite_noisy_global' and r['comparator']==comp],key=lambda r:float(r['event_budget']))
    x=np.array([float(v['event_budget'])*7/1e6 for v in r]);y=np.array([float(v['mean']) for v in r])
    low=np.array([float(v['ci95_low']) for v in r]);high=np.array([float(v['ci95_high']) for v in r])
    ax[1].errorbar(x,y,yerr=[y-low,high-y],fmt='o-',capsize=3,color=color,label='H − '+label)
ax[1].axhline(0,color='#555b63',lw=1)
ax[1].set(xscale='log',xlabel='Allocated total work (millions; log axis)',ylabel='Paired final excess-risk difference',title='Random controls: pointwise 95% t intervals')
ax[1].set_xticks([22.4,35,44.8,56,70],['22.4','35','44.8','56','70']);ax[1].grid(alpha=.2);ax[1].legend(fontsize=8)
ax[1].xaxis.set_minor_formatter(NullFormatter())
for ext in ['pdf','png']:
    fig.savefig(root/'figures'/('primary_distribution_diagnostic.'+ext),dpi=180)
plt.close(fig)

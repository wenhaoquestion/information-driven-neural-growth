"""Render theory/population/finite-sample figures strictly from saved results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})

def save(fig,name):
    for ext in ['pdf','png','svg']: fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=180)
    plt.close(fig)

def main():
    rows=json.loads((ROOT/'results/high_order_population.json').read_text())['rows']
    fig,ax=plt.subplots(1,2,figsize=(10.5,3.7),layout='constrained')
    for method,color,label in [('balanced','#b44531','Balanced: risk increases'),('vanishing','#166b89','Vanishing weight: risk decreases')]:
        rr=sorted([r for r in rows if r['d']==6 and r['method']==method],key=lambda r:r['t'])
        x=np.array([r['t'] for r in rr]);y=np.array([abs(float(r['risk_change'])) for r in rr]);theory=np.array([abs(float(r['limit_coefficient']))*r['t']**r['predicted_power'] for r in rr])
        ax[0].loglog(x,y,'o-',color=color,ms=3,label=label)
        ax[0].loglog(x,theory,'--',color=color,alpha=.6)
    ax[0].set(xlabel='Incoming radius t',ylabel='Absolute population loss change (nats)',title='Same two-child width; opposite signs')
    ax[0].legend(fontsize=8,loc='upper left')
    for d,c in [(6,'#166b89'),(8,'#bd6b19'),(10,'#7656a2')]:
        rr=sorted([r for r in rows if r['d']==d and r['method']=='vanishing' and r['t']<=.02],key=lambda r:r['t'])
        ax[1].semilogx([r['t'] for r in rr],[float(r['normalized']) for r in rr],'o-',ms=4,c=c,label=f'd = {d}')
    ax[1].axhline(1,color='.4',ls='--',lw=1);ax[1].set(xlabel='Incoming radius t',ylabel='Change / signed leading term',title='Convergence of the proved coefficient')
    ax[1].legend(fontsize=9);save(fig,'high_order_limits')
    data=json.loads((ROOT/'results/score_sampling.json').read_text());fig,ax=plt.subplots(figsize=(8.5,4.2),layout='constrained')
    for t,color in zip([.1,.2,.4,.8],['#7656a2','#b44531','#166b89','#60843c']):
        rr=[r for r in data['rows'] if r['t']==t];n=np.array([r['n'] for r in rr]);p=np.array([r['raw_score_nonpositive_count']/r['reps'] for r in rr]);N=1000;z=1.95996398454
        cen=(p+z*z/(2*N))/(1+z*z/N);half=z*np.sqrt(p*(1-p)/N+z*z/(4*N*N))/(1+z*z/N)
        ax.errorbar(n,p,yerr=np.array([p-(cen-half),(cen+half)-p]),fmt='o',ms=4,c=color,capsize=2,label=f'Raw score, t = {t:g}')
        ax.semilogx(n,[r['normal_sign_error_prediction'] for r in rr],color=color,lw=1,alpha=.7)
    ax.axhline(0,color='.25',ls='--',label='Supplied parity feature: 0 / 1000 errors')
    ax.axhspan(0,z*z/(1000+z*z),color='.5',alpha=.2)
    ax.set(xscale='log',xlabel='Number of iid training observations',ylabel='Nonpositive-score fraction',ylim=(-.015,.59))
    ax.legend(fontsize=8,ncol=2,loc='lower left',bbox_to_anchor=(0,1.02),borderaxespad=0);save(fig,'score_sampling')
if __name__=='__main__':main()

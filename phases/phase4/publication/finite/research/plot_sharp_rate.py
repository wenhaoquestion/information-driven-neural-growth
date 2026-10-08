"""Plot theorem envelopes and the newly executed proof challenges."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'serif','font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def main():
    with (ROOT/'results/quartet_scaling.csv').open() as f:rows=list(csv.DictReader(f))
    L=np.array([float(x['L']) for x in rows]);det=np.array([float(x['total_price']) for x in rows]);stochlo=np.array([float(x['randomized_total_price']) for x in rows])
    grid=np.geomspace(np.log(4),2048,200)
    fig,ax=plt.subplots(1,2,figsize=(10.5,4.1))
    ax[0].loglog(grid,10*(2+grid),color='#888888',ls=':',label='Previous proved universal bound')
    ax[0].loglog(grid,2+2*np.sqrt(3+grid),color='#cc6f26',lw=2,label='New proved universal bound')
    ax[0].loglog(L,det,'o-',color='#175a72',markersize=4,label='Exact constructed-family price')
    ax[0].loglog(L,stochlo,'s--',color='#6a4381',markersize=3,label='Constructed stochastic lower bound')
    ax[0].set(xlabel=r'$L=\log(1/\gamma)$',ylabel='Simultaneous total log-loss factor')
    ax[0].legend(frameon=False,fontsize=8)
    data=[json.loads(x) for x in (ROOT/'research/entropy_lipschitz_raw.jsonl').read_text().splitlines()]
    vals=[x['lip_fraction'] for x in data if x['lip_fraction'] is not None]
    grad=[x['gradient_fraction'] for x in data if x['gradient_fraction'] is not None]
    ax[1].hist(vals,bins=np.linspace(0,1,31),alpha=.65,label='Squared Lipschitz ratio',color='#175a72')
    ax[1].hist(grad,bins=np.linspace(0,1,31),histtype='step',lw=1.5,label='Gradient ratio',color='#cc6f26')
    ax[1].axvline(1,color='black',ls='--',lw=1)
    ax[1].set(xlabel='Computed quantity / proved upper bound',ylabel='Generated mathematical test cases')
    ax[1].legend(frameon=False,fontsize=9)
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(ROOT/'figures'/f'sharp_source_mass.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    print('Generated universal-envelope/proof-challenge figure from saved numerical outputs.')
if __name__=='__main__':main()

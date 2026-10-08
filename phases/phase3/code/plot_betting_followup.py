"""Plot the paired gate follow-up from its saved summary."""
import json,os
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(root/'tmp/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
data=json.loads((root/'results/betting_followup_summary.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(12.3,3.7),constrained_layout=True)
tasks=['null','linear','additive','interaction']
for method,offset,color in [('residual',-.12,'#3d64a8'),('random',.12,'#bf6238')]:
    rows=[next(r for r in data['rows'] if r['task']==t and r['method']==method) for t in tasks]
    x=np.arange(4)+offset
    d=[r['final_risk_paired_bet_minus_eb'] for r in rows]
    axs[0].errorbar(x,[r['mean'] for r in d],yerr=[[r['mean']-r['lower'] for r in d],[r['upper']-r['mean'] for r in d]],fmt='o',capsize=3,color=color,label=method)
    axs[1].plot(x,[r['bet_final_width'] for r in rows],'o-',color=color,label=method+' betting')
    axs[1].plot(x,[r['eb_final_width'] for r in rows],'x--',color=color,label=method+' EB')
    axs[2].plot(x,[r['training_cost_ratio_mean'] for r in rows],'o-',color=color,label=method)
axs[0].axhline(0,color='.4',ls=':');axs[2].axhline(1,color='.4',ls=':')
for ax in axs:
    ax.set_xticks(np.arange(4),['No signal','Linear','Additive','Interaction'],rotation=20)
axs[0].set(ylabel='Final Brier: betting minus EB',title='Same audit budget; paired new seeds')
axs[1].set(ylabel='Mean retained hidden width',title='Growth decisions change capacity')
axs[2].set(ylabel='Training/probe update proxy ratio',title='Betting / EB controller cost')
axs[0].legend(fontsize=8);axs[1].legend(fontsize=7);axs[2].legend(fontsize=8)
fig.savefig(root/'figures/betting_followup.pdf');fig.savefig(root/'figures/betting_followup.png',dpi=180)
print(root/'figures/betting_followup.pdf')

from pathlib import Path
import json,os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t
B=Path(__file__).resolve().parents[1];rows=[json.loads(x) for x in (B/'results/moments/raw.jsonl').read_text().splitlines()]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(14,4.3))
methods=[('raw_risk','Pooled raw moment','#bb5538'),('fresh_risk','Fresh residual correction','#256eaa'),('rank_known_risk','Known-rank correction (privileged)','#32845b')]
for ax,sigma in zip(axes[:2],[0.,.1]):
 for key,label,color in methods+([('least_squares_risk','Lifted least squares (more state)','#666666')] if sigma else []):
  vals=np.array([[r[key] for r in rows if r['sigma']==sigma and r['stage']==stage] for stage in range(1,9)])
  mean=vals.mean(1);err=t.ppf(.975,63)*vals.std(1,ddof=1)/8
  ax.plot(np.arange(1,9)*512,mean,'o-',ms=3,label=label,color=color)
  ax.fill_between(np.arange(1,9)*512,np.maximum(mean-err,1e-18),mean+err,alpha=.12,color=color)
 ax.set_yscale('log');ax.set_xlabel('Unique labels seen');ax.set_ylabel('Excess squared prediction risk');ax.set_title(f'Noise SD {sigma:g}');ax.grid(alpha=.18)
 if not sigma:
  ls=[r['least_squares_risk'] for r in rows if r['sigma']==0 and r['stage']==8]
  ax.text(.04,.08,f'Lifted LS final mean: {np.mean(ls):.1e}\n(excluded from this log axis)',transform=ax.transAxes,fontsize=9)
ax=axes[2];vals=np.array([[r['false_birth_risk_increase'] for r in rows if r['sigma']==0 and r['stage']==stage] for stage in range(1,9)])
mean=vals.mean(1);err=t.ppf(.975,63)*vals.std(1,ddof=1)/8
ax.plot(np.arange(1,9)*512,mean,'o-',color='#984c90',label='True risk increase from stale birth')
ax.fill_between(np.arange(1,9)*512,mean-err,mean+err,alpha=.15,color='#984c90')
lower=[next(r['false_birth_lower_bound'] for r in rows if r['sigma']==0 and r['stage']==s) for s in range(1,9)]
ax.plot(np.arange(1,9)*512,lower,'--',color='black',label='Proved expectation lower bound')
ax.set_yscale('log');ax.set_xlabel('Unique labels used in stale summary');ax.set_ylabel('Excess risk added at exact fit');ax.set_title('Harmful proposal despite zero residual');ax.grid(alpha=.18)
handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,-.06),frameon=False,fontsize=9)
axes[2].legend(frameon=False,fontsize=8);fig.tight_layout(rect=[0,.1,1,1]);(B/'figures').mkdir(exist_ok=True)
fig.savefig(B/'figures/moment_diagnostics.pdf',bbox_inches='tight');fig.savefig(B/'figures/moment_diagnostics.png',dpi=170,bbox_inches='tight')
summary={'final':[], 'raw_formula':[], 'minimum_false_birth':min(r['false_birth_risk_increase'] for r in rows if r['sigma']==0)}
for sigma in [0.,.1]:
 for key,_,_ in methods+[('least_squares_risk','','')]:
  x=np.array([r[key] for r in rows if r['sigma']==sigma and r['stage']==8]);mean=x.mean();h=t.ppf(.975,63)*x.std(ddof=1)/8
  summary['final'].append(dict(sigma=sigma,method=key,mean=mean,ci95=[mean-h,mean+h]))
 for stage in [1,8]:
  subset=[r for r in rows if r['sigma']==sigma and r['stage']==stage]
  summary['raw_formula'].append(dict(sigma=sigma,stage=stage,empirical=np.mean([r['raw_risk'] for r in subset]),exact=subset[0]['predicted_raw_risk']))
(B/'results/moments/findings.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
table=[r'\begin{tabular}{lrr}',r'\toprule',r'Estimator & Noise SD $0$ & Noise SD $.1$\\\midrule']
for key,label in [('raw_risk','Pooled raw moment'),('fresh_risk','Fresh residual correction'),('rank_known_risk','Known-rank correction'),('least_squares_risk','Lifted least squares')]:
 values=[next(r['mean'] for r in summary['final'] if r['method']==key and r['sigma']==sigma) for sigma in [0.,.1]]
 def texnumber(x):
  if x<1e-5:
   mantissa,exponent=f'{x:.2e}'.split('e');return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'
  return f'${x:.6f}$'
 table.append(label+' & '+' & '.join(map(texnumber,values))+r'\\')
table.extend([r'\bottomrule',r'\end{tabular}'])
(B/'manuscript/moment_table.tex').write_text('\n'.join(table)+'\n')

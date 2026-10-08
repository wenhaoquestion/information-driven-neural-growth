"""Create all neural-study summary tables and figures from saved raw runs."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'
F=ROOT/'figures';F.mkdir(parents=True,exist_ok=True)

def stat(x):
 x=np.asarray(x,dtype=float);return {'mean':float(x.mean()),'t95_halfwidth':float(student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))),'n':len(x)}

def summarize_bridge(path):
 a=json.loads(path.read_text());groups={}
 for r in a['runs']:
  key=(r['regime'],r['ntrain'],r['poststeps']);groups.setdefault(key,[]).append(r)
 rows=[]
 for (regime,n,steps),rr in groups.items():
  row={'regime':regime,'ntrain':n,'poststeps':steps,'nseeds':len(rr)}
  for method in ['spectral','random','oracle']:
   for metric in ['post_risk','gated_risk','initial_gain','negative_axis_alignment','accepted']:
    st=stat([r['events'][method][metric] for r in rr])
    row[f'{method}_{metric}']=st['mean'];row[f'{method}_{metric}_95half']=st['t95_halfwidth']
  for method in ['continued_width1','fixed_width2_equal_steps','fixed_width2_approx_equal_compute']:
   st=stat([r[method] for r in rr]);row[method]=st['mean'];row[method+'_95half']=st['t95_halfwidth']
  for other in ['random','oracle']:
   st=stat([r['events']['spectral']['post_risk']-r['events'][other]['post_risk'] for r in rr])
   row[f'spectral_minus_{other}']=st['mean'];row[f'spectral_minus_{other}_95half']=st['t95_halfwidth']
   gst=stat([r['events']['spectral']['gated_risk']-r['events'][other]['gated_risk'] for r in rr])
   row[f'gated_spectral_minus_{other}']=gst['mean'];row[f'gated_spectral_minus_{other}_95half']=gst['t95_halfwidth']
  st=stat([r['events']['spectral']['gated_risk']-r['continued_width1'] for r in rr])
  row['gated_minus_continued']=st['mean'];row['gated_minus_continued_95half']=st['t95_halfwidth']
  rows.append(row)
 with (R/(path.stem+'_summary.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return rows,groups

def plot_bridge(rows,groups):
 plt.rcParams.update({'font.size':9,'axes.spines.right':False,'axes.spines.top':False})
 fig,ax=plt.subplots(1,3,figsize=(12.3,3.7),layout='constrained')
 names=['four_state','rotated_nuisance','no_signal']
 labels=['Four states','Rotated 8-dimensional inputs','No target signal']
 styles=[('spectral_post_risk','Spectral split','#2171b5'),('random_post_risk','Random split','#ef8a62'),
  ('continued_width1','Continue width 1','#777777'),('fixed_width2_approx_equal_compute','Fixed width 2','#238b45')]
 for axis,name,title in zip(ax,names,labels):
  rr=[r for r in rows if r['regime']==name and r['poststeps']==300]
  for key,label,c in styles:
   axis.errorbar([r['ntrain'] for r in rr],[r[key] for r in rr],
     yerr=[r[key+'_95half'] for r in rr],marker='o',color=c,label=label,capsize=3)
  axis.axhline(np.log(2),ls=':',lw=1,color='black',label='Constant population risk' if name=='four_state' else None)
  axis.set(xscale='log',xlabel='IID training examples (validation same size)',ylabel='Exact population log loss (nats)',title=title)
 fig.legend(*ax[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
 fig.savefig(F/'learning_curvature_risks.pdf');fig.savefig(F/'learning_curvature_risks.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(10.2,3.7),layout='constrained')
 rg=['four_state','rotated_nuisance','sample_pretrained','random_start','no_signal']
 labels=['4 states','Rotated','Pretrained','Random init.','No signal']
 xx=np.arange(len(rg))
 for delta,steps,c in [(-.1,30,'#2171b5'),(.1,300,'#ef8a62')]:
  rr=[next(r for r in rows if r['regime']==z and r['ntrain']==1024 and r['poststeps']==steps) for z in rg]
  ax[0].errorbar(xx+delta,[r['spectral_minus_random'] for r in rr],yerr=[r['spectral_minus_random_95half'] for r in rr],
   fmt='o',color=c,capsize=3,label=f'{steps} post-growth updates')
  ax[1].errorbar(xx+delta,[r['gated_minus_continued'] for r in rr],yerr=[r['gated_minus_continued_95half'] for r in rr],
   fmt='o',color=c,capsize=3)
 for a in ax:a.axhline(0,color='black',lw=1);a.set_xticks(xx,labels,rotation=15);a.set_ylabel('Paired population risk difference (nats)')
 ax[0].set_title('Direction advantage: spectral minus random')
 ax[1].set_title('Heuristic validation gate minus continuation')
 ax[0].legend(frameon=False)
 fig.savefig(F/'learning_curvature_differences.pdf');fig.savefig(F/'learning_curvature_differences.png',dpi=180);plt.close(fig)

def summarize_probe(filename='learning_neural_probe32.json', figure='learning_information_probe'):
 a=json.loads((R/filename).read_text());rows=[]
 for regime in ['teacher','localized','linear_null']:
  rr=[r for r in a['runs'] if r['regime']==regime]
  for method in list(rr[0]['selected'])+['continued_width2','fixed_width3_steps','fixed_width3_parameter_compute']:
   vals=np.array([r['selected'][method]['risk'] if method in r['selected'] else r[method] for r in rr])
   rand=np.array([r['selected']['random']['risk'] for r in rr]);st=stat(vals);diff=stat(vals-rand)
   rows.append({'regime':regime,'method':method,**st,'minus_random':diff['mean'],'paired95_halfwidth':diff['t95_halfwidth']})
 with (R/(Path(filename).stem+'_summary.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 fig,ax=plt.subplots(1,3,figsize=(12,3.8),layout='constrained',sharey=True)
 methods=['conditional_information','activation_entropy','trained_validation','continued_width2','fixed_width3_parameter_compute']
 labels=['Contrast CMI','Activation entropy','Trained validation','Continue width 2','Fixed width 3']
 if 'aligned_information' in a['runs'][0]['selected']:
  methods.insert(1,'aligned_information');labels.insert(1,'Aligned CMI')
 for a,regime in zip(ax,['teacher','localized','linear_null']):
  rr=[next(r for r in rows if r['method']==m and r['regime']==regime) for m in methods]
  a.errorbar([r['minus_random'] for r in rr],np.arange(len(rr)),xerr=[r['paired95_halfwidth'] for r in rr],fmt='o',capsize=3)
  a.xaxis.set_major_locator(plt.MaxNLocator(4));a.axvline(0,color='black',lw=1);a.set_yticks(np.arange(len(rr)),labels);a.set(title=regime.replace('_',' '),xlabel='Risk minus random growth (nats)');a.invert_yaxis()
 fig.savefig(F/(figure+'.pdf'));fig.savefig(F/(figure+'.png'),dpi=180)
 plt.close(fig)
 return rows

if __name__=='__main__':
 rows,groups=summarize_bridge(R/'learning_curvature_bridge64.json');plot_bridge(rows,groups);probe=summarize_probe()
 for r in rows:
  print(r['regime'],r['ntrain'],r['poststeps'],'risk spec/rand/fixed/continue',*[round(r[k],6) for k in ['spectral_post_risk','random_post_risk','fixed_width2_approx_equal_compute','continued_width1']],
  'diff95',round(r['spectral_minus_random'],6),round(r['spectral_minus_random_95half'],6),'gate',round(r['spectral_accepted'],3),'gatediff95',round(r['gated_minus_continued'],6),round(r['gated_minus_continued_95half'],6))

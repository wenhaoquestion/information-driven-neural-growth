"""Adaptive extension summaries: read raw output only, preserve every arm."""
from pathlib import Path
import csv,json
from collections import Counter
import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/'experiments/adaptive_confirmation/results';OUT=ROOT/'results/adaptive'
TASKS=['null','lowrank_noiseless','indefinite_noisy','nonlinear']
METHODS=['fresh_adaptive','random_adaptive','fixed2_gated','fixed7_gated','fixed7_ungated']
NAMES={'fresh_adaptive':'Fresh residual growth','random_adaptive':'Random growth','fixed2_gated':'Fixed 2 gated','fixed7_gated':'Fixed 7 gated','fixed7_ungated':'Fixed 7 ungated'}
COLORS=['#287f8e','#303d50','#7c65b3','#b35a35','#c29535']
def stat(x):
    x=np.asarray(x);mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)));h=float(student_t.ppf(.975,len(x)-1)*se)
    return dict(n=len(x),mean=mean,ci95_low=mean-h,ci95_high=mean+h)
def csvout(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    runs=[json.loads(p.read_text()) for p in sorted(IN.glob('*.json')) if p.name not in ['environment.json','run_index.json']]
    groups={t:[r for r in runs if r['task']==t] for t in TASKS};summary=[];paired=[];events=[];checks=[]
    assert all(len(rr)==12 for rr in groups.values())
    for task,rr in groups.items():
        for m in METHODS:
            vals=[r['methods'][m] for r in rr];actions=Counter(e['action'] for v in vals for e in v['events'][1:])
            row={'task':task,'method':m,**stat([v['final_risk'] for v in vals]),
                 'mean_width':float(np.mean([v['events'][-1]['width'] for v in vals])),
                 'min_width':min(v['events'][-1]['width'] for v in vals),'max_width':max(v['events'][-1]['width'] for v in vals),
                 'grow':actions['grow'],'continue':actions['continue'],'hold':actions['hold'],
                 'evaluator_risk_increases':sum(e['risk_after']>e['risk_before']+1e-12 for v in vals for e in v['events'][1:]),
                 'mean_total_proxy':float(np.mean([sum(e['total_event_proxy'] for e in v['events']) for v in vals])),
                 'mean_rejected_proxy':float(np.mean([sum(e['rejected_branch_proxy'] for e in v['events']) for v in vals]))}
            summary.append(row)
            for r in rr:
                for e in r['methods'][m]['events']:
                    assert e['total_event_proxy']<=r['config']['event_budget']
                    if e['action']=='hold':assert e['before_parameters']==e['after_parameters']
                    assert max(e['fit_pool_ids'])<e['validation_range'][0]
                    events.append({'task':task,'seed':r['seed'],'method':m,'event':e['t'],'action':e['action'],'width':e['width'],'risk_before':e['risk_before'],'risk_after':e['risk_after'],'total_proxy':e['total_event_proxy'],'rejected_proxy':e['rejected_branch_proxy']})
        for comp in METHODS[1:]:
            paired.append({'task':task,'first':'fresh_adaptive','second':comp,**stat([r['methods']['fresh_adaptive']['final_risk']-r['methods'][comp]['final_risk'] for r in rr])})
    report={'runs':len(runs),'trajectories':len(runs)*len(METHODS),'events':len(events),'summary':summary,'paired':paired,
            'wall_seconds':json.loads((IN/'run_index.json').read_text())['wall_seconds'],
            'checks':{'all_event_proxy_within_budget':True,'every_hold_exact':True,'current_validation_excluded_from_fit_pools':True}}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2));csvout(OUT/'method_summary.csv',summary);csvout(OUT/'paired_contrasts.csv',paired);csvout(OUT/'events.csv',events)
    table=['\\begin{tabular}{lrrrr}','\\toprule','Method & Null & Rank two & Indefinite & Nonlinear \\\\','\\midrule']
    for m in METHODS:
        cells=[]
        for task in TASKS:
            row=next(r for r in summary if r['task']==task and r['method']==m);cells.append(f"{row['mean']:.5f} ({row['mean_width']:.2f})")
        table.append(NAMES[m]+' & '+' & '.join(cells)+' \\\\')
    table+=['\\bottomrule','\\end{tabular}'];(ROOT/'manuscript/adaptive_table.tex').write_text('\n'.join(table)+'\n')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(2,4,figsize=(12,6))
    for col,task in enumerate(TASKS):
        rr=groups[task]
        for i,m in enumerate(METHODS):
            for row,key in [(0,'risk_after'),(1,'width')]:
                v=np.array([[e[key] for e in r['methods'][m]['events']] for r in rr]);mu=v.mean(0);se=v.std(0,ddof=1)/np.sqrt(len(rr));xx=np.arange(1,8)
                axes[row,col].plot(xx,mu,color=COLORS[i],label=NAMES[m]);axes[row,col].fill_between(xx,mu-se,mu+se,color=COLORS[i],alpha=.12)
                axes[row,col].set_xlabel('Arrival / decision event')
        axes[0,col].set_title(task.replace('_',' '));axes[0,col].set_ylabel('Population MSE');axes[1,col].set_ylabel('Committed width')
        if task=='lowrank_noiseless':axes[0,col].set_yscale('log')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.03),ncol=5,frameon=False);fig.tight_layout(rect=[0,.05,1,1])
    fig.savefig(ROOT/'figures/adaptive_trajectories.pdf');fig.savefig(ROOT/'figures/adaptive_trajectories.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

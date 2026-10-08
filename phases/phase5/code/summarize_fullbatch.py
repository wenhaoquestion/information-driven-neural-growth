"""Post-hoc full-minibatch follow-up; read only its separately frozen outputs."""
from pathlib import Path
import json,csv
import numpy as np
from scipy.stats import t as tdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/'experiments/fullbatch_confirmation/results';OUT=ROOT/'results/fullbatch'
TASKS=['null','lowrank_noiseless','indefinite_noisy','nonlinear']
METHODS=['historical_moment','random_replay128','random_matched138','fixed7_replay138']
LABELS={'historical_moment':'Historical moment','random_replay128':'Random R128','random_matched138':'Random R138','fixed7_replay138':'Fixed width 7'}
def stat(x):
    x=np.asarray(x);m=float(x.mean());h=float(tdist.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return {'n':len(x),'mean':m,'ci95_low':m-h,'ci95_high':m+h}
def csvout(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    OUT.mkdir(parents=True,exist_ok=True);runs=[json.loads(p.read_text()) for p in sorted(IN.glob('*.json')) if p.name not in ['environment.json','run_index.json']]
    summary=[];paired=[];groups={task:[r for r in runs if r['task']==task] for task in TASKS};counts=[]
    for task,rr in groups.items():
        assert len(rr)==12
        for method in METHODS:
            summary.append({'task':task,'method':method,**stat([r['methods'][method]['final_risk'] for r in rr]),
                            'fit_accesses':rr[0]['methods'][method]['ledger']['label_accesses']['fit'],
                            'optimizer_steps':sum(e['fit']['steps'] for e in rr[0]['methods'][method]['events'])})
        for first,second in [('random_replay128','random_matched138'),('historical_moment','random_replay128'),('historical_moment','random_matched138'),('historical_moment','fixed7_replay138')]:
            paired.append({'task':task,'first':first,'second':second,**stat([r['methods'][first]['final_risk']-r['methods'][second]['final_risk'] for r in rr])})
        for r in rr:
            a=r['methods']['random_replay128'];b=r['methods']['random_matched138']
            assert a['ledger']['label_accesses']['fit']==b['ledger']['label_accesses']['fit']
            assert [e['fit']['steps'] for e in a['events']]==[e['fit']['steps'] for e in b['events']]
            counts.append({'task':task,'seed':r['seed'],'fit_accesses':a['ledger']['label_accesses']['fit'],'optimizer_steps':sum(e['fit']['steps'] for e in a['events'])})
    report={'runs':len(runs),'trajectories':len(runs)*4,'events':len(runs)*4*7,'summary':summary,'paired':paired,
            'matched_random_steps_and_accesses':True,'random_count_checks':counts,'wall_seconds':json.loads((IN/'run_index.json').read_text())['wall_seconds']}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2));csvout('method_summary.csv',summary);csvout('paired_contrasts.csv',paired);csvout('random_count_checks.csv',counts)
    table=['\\begin{tabular}{lrrrr}','\\toprule','Method & Null & Rank two & Indefinite & Nonlinear \\\\','\\midrule']
    for m in METHODS:
        cells=[f"{next(r['mean'] for r in summary if r['task']==task and r['method']==m):.6f}" for task in TASKS]
        table.append(LABELS[m]+' & '+' & '.join(cells)+' \\\\')
    table+=['\\bottomrule','\\end{tabular}'];(ROOT/'manuscript/fullbatch_table.tex').write_text('\n'.join(table)+'\n')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(1,4,figsize=(11,3.2));pairs=[('random_replay128','random_matched138'),('historical_moment','random_replay128'),('historical_moment','random_matched138')]
    for a,task in zip(axes,TASKS):
        rows=[next(r for r in paired if r['task']==task and (r['first'],r['second'])==pair) for pair in pairs]
        m=np.array([r['mean'] for r in rows]);lo=np.array([r['ci95_low'] for r in rows]);hi=np.array([r['ci95_high'] for r in rows])
        a.errorbar(m,np.arange(3),xerr=np.array([m-lo,hi-m]),fmt='o',capsize=3,color='#287f8e');a.axvline(0,color='#888',lw=.7)
        a.set_yticks(np.arange(3),['R128 − R138','Moment − R128','Moment − R138']);a.set_title(task.replace('_',' '));a.set_xlabel('Paired final MSE difference');a.invert_yaxis()
    fig.tight_layout();fig.savefig(ROOT/'figures/fullbatch_followup.pdf');fig.savefig(ROOT/'figures/fullbatch_followup.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

"""Generate descriptive paired summaries/figures from every completed run."""
from pathlib import Path
import argparse,csv,json,os
import numpy as np
from scipy.stats import t as student
BASE=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(BASE/'tmp/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

METHODS=['withheld_residual','released_residual','withheld_random','released_random','fixed2_online','fixed26_online','fixed26_gated']
LABELS=['Withheld residual','Released residual','Withheld random','Released random','Fixed 2 online','Fixed 26 online','Fixed 26 gated']
COLORS=['#b94c4c','#b94c4c','#247b9c','#247b9c','#aaa59e','#272d34','#7b62a3']
STYLES=['--','-','--','-',':','-',':']
TASKS=['null','additive','interaction','radial']

def interval(values):
    a=np.array(values,dtype=float);n=len(a);mean=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(n)) if n>1 else float('nan')
    h=float(student.ppf(.975,n-1)*se) if n>1 else float('nan')
    return dict(n=n,mean=mean,sd=float(a.std(ddof=1)),se=se,ci95_low=mean-h,ci95_high=mean+h)
def writecsv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default=str(BASE/'results/confirmation'));args=ap.parse_args()
    files=[p for p in Path(args.results).glob('*.json') if p.stem.startswith(tuple(TASKS))]
    runs=[json.loads(p.read_text()) for p in sorted(files)];assert runs,'No completed runs'
    expected={(t,seed) for t in TASKS for seed in json.loads((BASE/'config.json').read_text())['seeds']}
    if {(x['task'],x['seed']) for x in runs} != expected:
        raise SystemExit('Full historical raw run set is required. Restore the raw manifest files, or use plot_saved_summaries.py to plot the complete distributed CSV summaries.')
    out=BASE/'results';fig=BASE/'figures';fig.mkdir(exist_ok=True);perrun=[];eventrows=[]
    for rr in runs:
        for method,m in rr['methods'].items():
            ee=m['events'];last=ee[-1];ac=m['pre_tail_access'];parts=ac['parts']
            row=dict(task=rr['task'],seed=rr['seed'],method=method,initial_risk=m['initial_risk'],final_risk=m['final_risk'],
                event12_risk=ee[11]['retained_risk'] if len(ee)>=12 else '',last6_mean_risk=float(np.mean([e['retained_risk'] for e in ee[-6:]])),
                mean_event_risk=float(np.mean([e['retained_risk'] for e in ee])),tail_risk=m['tail_risk'],
                final_width=last['width'],final_parameters=last['parameters'],growth_count=sum(e['action']=='grow' for e in ee),
                continue_count=sum(e['action']=='continue' for e in ee),hold_count=sum(e['action']=='hold' for e in ee),
                unique_observed_labels=last['unique_observed_labels'],unique_fitted_labels=ac['unique_fitted_labels'],
                repeated_fit_label_accesses=ac['repeated_fit_label_accesses'],training_parameter_examples=m['training_parameter_examples'],
                candidate_parameter_examples=parts.get('proposal_fit',{}).get('parameter_examples',0),
                proposal_prediction_parameter_examples=sum(e['proposal']['prediction_parameter_examples'] if e['proposal'] else 0 for e in ee),
                rejected_branch_parameter_examples=last['cumulative_rejected_branch_parameter_examples'],
                audit_parameter_examples=last['audit_parameter_examples'],audit_prediction_evaluations=last['cumulative_audit_prediction_evaluations'],
                betting_factor_evaluations=last['betting_factor_evaluations'],tail_parameter_examples=m['tail_cost']['parameter_examples'],
                tail_unique_fitted_labels=m['all_access']['unique_fitted_labels'],tail_repeated_fit_label_accesses=m['all_access']['repeated_fit_label_accesses'],
                evaluator_checkpoint_increases=sum(e['retained_risk']>e['evaluator_risks']['incumbent']+1e-12 for e in ee),
                elapsed_seconds=m['elapsed_seconds'])
            perrun.append(row)
            for e in ee:
                eventrows.append(dict(task=rr['task'],seed=rr['seed'],method=method,event=e['t'],action=e['action'],retained_risk=e['retained_risk'],
                    incumbent_risk=e['evaluator_risks']['incumbent'],continuation_risk=e['evaluator_risks']['continuation'],
                    growth_risk=e['evaluator_risks'].get('growth',''),width=e['width'],parameters=e['parameters'],
                    unique_observed_labels=e['unique_observed_labels'],fit_available_before=e['fit_available_before'],
                    fit_available_after_release=e['fit_available_after_release'],unique_fitted_labels=e['unique_fitted_labels'],
                    event_repeated_fit_accesses=e['event_repeated_fit_accesses'],training_parameter_examples=e['training_parameter_examples'],
                    rejected_branch_parameter_examples=e['cumulative_rejected_branch_parameter_examples'],audit_parameter_examples=e['audit_parameter_examples']))
    writecsv(out/'run_summary.csv',perrun);writecsv(out/'events.csv',eventrows)
    summary=[];contrasts=[]
    for task in TASKS:
        for method in METHODS:
            rows=[a for a in perrun if a['task']==task and a['method']==method]
            if not rows:continue
            row=dict(task=task,method=method,n=len(rows))
            for metric in ['initial_risk','final_risk','tail_risk','final_width','growth_count','mean_event_risk','unique_fitted_labels',
                           'repeated_fit_label_accesses','training_parameter_examples','rejected_branch_parameter_examples','audit_parameter_examples']:
                ci=interval([a[metric] for a in rows]);row[metric+'_mean']=ci['mean'];row[metric+'_ci_low']=ci['ci95_low'];row[metric+'_ci_high']=ci['ci95_high']
            summary.append(row)
        pairs=[('release_residual','released_residual','withheld_residual'),('release_random','released_random','withheld_random'),
               ('residual_minus_random_released','released_residual','released_random'),
               ('released_residual_minus_fixed2','released_residual','fixed2_online'),
               ('released_residual_minus_fixed26','released_residual','fixed26_online'),
               ('released_residual_minus_fixed26_gated','released_residual','fixed26_gated'),
               ('fixed26_gated_minus_online','fixed26_gated','fixed26_online')]
        trows={}
        for a in perrun:
            if a['task']==task:trows.setdefault(a['seed'],{})[a['method']]=a
        for metric in ['final_risk','tail_risk','mean_event_risk','final_width']:
            for name,a,b in pairs:
                contrasts.append(dict(task=task,metric=metric,contrast=name,**interval([v[a][metric]-v[b][metric] for v in trows.values()])))
            vals=[(v['released_residual'][metric]-v['withheld_residual'][metric])-(v['released_random'][metric]-v['withheld_random'][metric]) for v in trows.values()]
            contrasts.append(dict(task=task,metric=metric,contrast='release_by_proposal_interaction',**interval(vals)))
    writecsv(out/'method_summary.csv',summary);writecsv(out/'paired_contrasts.csv',contrasts)
    (out/'summary.json').write_text(json.dumps(dict(runs=len(runs),method_trajectories=len(perrun),events=len(eventrows),
        uncertainty='Descriptive 95% paired Student-t intervals; no multiplicity adjustment; independent seeds are replicates.',
        summaries=summary,contrasts=contrasts),indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42})
    fig1,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,task in zip(axs.flat,TASKS):
        rs=[a for a in runs if a['task']==task]
        if not rs:continue
        for method,label,color,ls in zip(METHODS,LABELS,COLORS,STYLES):
            vals=np.array([[r0['methods'][method]['initial_risk']]+[e['retained_risk'] for e in r0['methods'][method]['events']] for r0 in rs])
            xx=np.arange(vals.shape[1]);mean=vals.mean(0);se=vals.std(0,ddof=1)/np.sqrt(len(vals))
            ax.plot(xx,mean,label=label,color=color,linestyle=ls,lw=1.6);ax.fill_between(xx,mean-se,mean+se,color=color,alpha=.06)
        ax.set_title(task.capitalize());ax.set_xlabel('Completed audit decisions');ax.set_ylabel('Committed Brier risk')
    handles,labels=axs[0,0].get_legend_handles_labels();fig1.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig1.savefig(fig/'risk_trajectories.pdf');fig1.savefig(fig/'risk_trajectories.png',dpi=160);plt.close(fig1)
    fig2,axs=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
    names=['release_residual','release_random','residual_minus_random_released','release_by_proposal_interaction']
    labels=['Release: residual','Release: random','Residual − random (released)','Release × proposal']
    for ax,task in zip(axs.flat,TASKS):
        cs=[next(c for c in contrasts if c['task']==task and c['metric']=='final_risk' and c['contrast']==name) for name in names]
        means=1000*np.array([c['mean'] for c in cs]);lo=1000*np.array([c['ci95_low'] for c in cs]);hi=1000*np.array([c['ci95_high'] for c in cs])
        ax.errorbar(means,np.arange(4),xerr=[means-lo,hi-means],fmt='o',color='#284b63',capsize=3);ax.axvline(0,color='#555',lw=.8)
        ax.set_yticks(np.arange(4),labels);ax.invert_yaxis();ax.set_title(task.capitalize());ax.set_xlabel('Paired final Brier difference (×10⁻³; 95% CI)')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    fig2.savefig(fig/'paired_diagnostics.pdf');fig2.savefig(fig/'paired_diagnostics.png',dpi=160);plt.close(fig2)
    fig3,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    fig3.suptitle('Committed checkpoints and post-training diagnostic (×)',fontsize=10)
    for ax,task in zip(axs.flat,TASKS):
        for method,label,color,ls in zip(METHODS,LABELS,COLORS,STYLES):
            s=next(a for a in summary if a['task']==task and a['method']==method)
            ax.scatter(s['final_width_mean'],s['final_risk_mean'],label=label,color=color,marker='o' if ls=='-' else ('s' if ls=='--' else '^'),s=45)
            ax.plot([s['final_width_mean']]*2,[s['final_risk_ci_low'],s['final_risk_ci_high']],color=color,alpha=.7)
            ax.scatter(s['final_width_mean'],s['tail_risk_mean'],color=color,marker='x',s=32)
        ax.set_title(task.capitalize());ax.set_xlabel('Mean retained hidden width');ax.set_ylabel('Brier risk')
    handles,labels=axs[0,0].get_legend_handles_labels();fig3.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig3.savefig(fig/'risk_capacity_tail.pdf');fig3.savefig(fig/'risk_capacity_tail.png',dpi=160);plt.close(fig3)
    tex=['% Generated from actual confirmation run_summary.csv.','\\begin{tabular}{llrrr}','\\toprule','Task & Method & Final risk & Tail risk & Width \\\\','\\midrule']
    for s in summary:
        tex.append(f"{s['task'].capitalize()} & {LABELS[METHODS.index(s['method'])]} & {s['final_risk_mean']:.5f} & {s['tail_risk_mean']:.5f} & {s['final_width_mean']:.2f} \\\\")
    tex+=['\\bottomrule','\\end{tabular}'];(out/'results_table.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(dict(runs=len(runs),trajectories=len(perrun),events=len(eventrows),figures=3),indent=2))

if __name__=='__main__':main()

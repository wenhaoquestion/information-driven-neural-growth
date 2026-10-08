"""Rebuild every numerical summary and figure from frozen confirmation JSON."""
from pathlib import Path
import csv, json
import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/'experiments/confirmation/results'; OUT=ROOT/'results/quadratic'; FIG=ROOT/'figures'
METHODS=['historical_moment','reservoir_residual','fresh_residual','learned_probe','random_replay128','random_matched138','fixed7_replay138']
TASKS=['null','lowrank_noiseless','indefinite_noisy','nonlinear']
LABELS={'historical_moment':'Historical moment','reservoir_residual':'Replay residual','fresh_residual':'Fresh residual','learned_probe':'Learned probe','random_replay128':'Random R128','random_matched138':'Random R138','fixed7_replay138':'Fixed width 7'}
COLORS=['#b35a35','#7c65b3','#287f8e','#688a45','#959595','#232d3b','#c29535']
def stat(v):
    v=np.asarray(v,dtype=float); mean=float(v.mean()); se=float(v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.
    h=float(student_t.ppf(.975,len(v)-1)*se) if len(v)>1 else 0.
    return dict(n=len(v),mean=mean,se=se,ci95_low=mean-h,ci95_high=mean+h)
def csvout(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True,parents=True);FIG.mkdir(exist_ok=True)
    allrows=[json.loads(p.read_text()) for p in sorted(IN.glob('*.json')) if p.name not in ['environment.json','run_index.json']]
    groups={t:sorted([r for r in allrows if r['task']==t],key=lambda r:r['seed']) for t in TASKS}
    assert all(len(v)==12 for v in groups.values())
    summary=[]; contrasts=[]; events=[]; checkpoint=[]; trajectory={}
    comparator=['fresh_residual','reservoir_residual','learned_probe','random_matched138','fixed7_replay138']
    for task,rr in groups.items():
        trajectory[task]={}
        for method in METHODS:
            vals=[r['methods'][method] for r in rr]
            row={'task':task,'method':method,**stat([r['final_risk'] for r in vals])}
            row.update(mean_total_proxy=float(np.mean([sum(v['ledger']['proxy'].values())+v['ledger']['extra']['eigendecomposition'] for v in vals])),
                       mean_fit_accesses=float(np.mean([v['ledger']['label_accesses']['fit'] for v in vals])),
                       mean_proposal_accesses=float(np.mean([v['ledger']['label_accesses']['proposal'] for v in vals])),
                       final_width=vals[0]['events'][-1]['width'],
                       persistent_words=vals[0]['events'][-1]['persistent_retention_words']+vals[0]['events'][-1]['persistent_model_optimizer_words']+6)
            summary.append(row)
            bytime=np.array([[e['risk_after'] for e in v['events']] for v in vals]);trajectory[task][method]={'mean':bytime.mean(axis=0).tolist(),'se':(bytime.std(axis=0,ddof=1)/np.sqrt(len(rr))).tolist()}
            for run in rr:
                for e in run['methods'][method]['events']:
                    events.append({'task':task,'seed':run['seed'],'method':method,'event':e['t'],'risk':e['risk_after'],'width':e['width'],'proxy':e['total_event_proxy'],
                                   'fit_accesses':e['event_label_accesses']['fit'],'proposal_accesses':e['event_label_accesses']['proposal'],'moment_accesses':e['event_label_accesses']['moment']})
        for comp in comparator:
            dif=[r['methods']['historical_moment']['final_risk']-r['methods'][comp]['final_risk'] for r in rr]
            contrasts.append({'task':task,'first':'historical_moment','second':comp,'metric':'final_risk_first_minus_second',**stat(dif)})
        contrasts.append({'task':task,'first':'random_replay128','second':'random_matched138','metric':'final_risk_first_minus_second',**stat([r['methods']['random_replay128']['final_risk']-r['methods']['random_matched138']['final_risk'] for r in rr])})
        contrasts.append({'task':task,'first':'historical_moment','second':'random_replay128','metric':'exploratory_attribution_control_final_risk_first_minus_second',**stat([r['methods']['historical_moment']['final_risk']-r['methods']['random_replay128']['final_risk'] for r in rr])})
        for comp in ['fresh_residual','reservoir_residual','learned_probe','random_matched138']:
            for metric in ['risk_after_identical_refit','population_best_gain_along_direction']:
                if task=='nonlinear' and metric.startswith('population'):continue
                dif=[]
                for run in rr:
                    pair=run['paired_checkpoint_diagnostics']
                    dif.append(np.mean([e['choices']['historical_moment'][metric]-e['choices'][comp][metric] for e in pair]))
                checkpoint.append({'task':task,'first':'historical_moment','second':comp,'metric':metric+'_first_minus_second_seed_mean_over_six_events',**stat(dif)})
    csvout('method_summary.csv',summary);csvout('paired_contrasts.csv',contrasts);csvout('events.csv',events);csvout('checkpoint_contrasts.csv',checkpoint)
    index=json.loads((IN/'run_index.json').read_text())
    report={'runs':len(allrows),'trajectories':sum(len(r['methods']) for r in allrows),'events':len(events),'same_checkpoint_comparisons':sum(len(e['choices']) for r in allrows for e in r['paired_checkpoint_diagnostics']),
            'wall_seconds':index['wall_seconds'],'summary':summary,'primary_contrasts':contrasts,'checkpoint_contrasts':checkpoint}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2))
    table=['\\begin{tabular}{lrrrr}','\\toprule','Method & Null & Rank two & Indefinite & Nonlinear \\\\','\\midrule']
    for method in METHODS:
        nums=[next(r['mean'] for r in summary if r['task']==task and r['method']==method) for task in TASKS]
        table.append(LABELS[method]+' & '+' & '.join(f'{v:.6f}' for v in nums)+' \\\\')
    table+=['\\bottomrule','\\end{tabular}']
    (ROOT/'manuscript/quadratic_table.tex').write_text('\n'.join(table)+'\n')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    fig,ax=plt.subplots(2,2,figsize=(10,7))
    for a,task in zip(ax.flat,TASKS):
        for i,m in enumerate(METHODS):
            v=trajectory[task][m];xx=np.arange(1,8)*1024;mu=np.array(v['mean']);se=np.array(v['se'])
            a.plot(xx,mu,color=COLORS[i],label=LABELS[m],lw=1.6);a.fill_between(xx,mu-se,mu+se,color=COLORS[i],alpha=.10)
        a.set_title(task.replace('_',' '));a.set_xlabel('Unique labels arrived');a.set_ylabel('Population MSE')
        if task=='lowrank_noiseless':a.set_yscale('log')
    handles,labels=ax[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.02),frameon=False)
    fig.tight_layout(rect=[0,.085,1,1]);fig.savefig(FIG/'quadratic_trajectories.pdf');fig.savefig(FIG/'quadratic_trajectories.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(1,4,figsize=(12,3.7))
    comps=['fresh_residual','reservoir_residual','learned_probe','random_matched138','fixed7_replay138']
    for a,task in zip(ax,TASKS):
        rows=[next(z for z in contrasts if z['task']==task and z['second']==c and z['first']=='historical_moment') for c in comps]
        means=np.array([r['mean'] for r in rows]);lo=np.array([r['ci95_low'] for r in rows]);hi=np.array([r['ci95_high'] for r in rows])
        a.errorbar(means,np.arange(5),xerr=np.array([means-lo,hi-means]),fmt='o',color=COLORS[0],capsize=3)
        a.axvline(0,color='#888',lw=.7);a.set_yticks(np.arange(5),[LABELS[c] for c in comps]);a.set_title(task.replace('_',' '));a.set_xlabel('Historical minus comparator MSE');a.invert_yaxis()
    fig.tight_layout();fig.savefig(FIG/'quadratic_paired.pdf');fig.savefig(FIG/'quadratic_paired.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11,6.5))
    for col,task in enumerate(TASKS[:3]):
        rr=groups[task]
        for i,m in enumerate(['historical_moment','reservoir_residual','fresh_residual','learned_probe','random_matched138']):
            direction=np.array([[e['choices'][m]['population_best_gain_along_direction'] for e in r['paired_checkpoint_diagnostics']] for r in rr])
            fitrisk=np.array([[e['choices'][m]['risk_after_identical_refit'] for e in r['paired_checkpoint_diagnostics']] for r in rr])
            for a,v in [(axes[0,col],direction),(axes[1,col],fitrisk)]:
                mu=v.mean(0);se=v.std(0,ddof=1)/np.sqrt(len(rr));a.plot(np.arange(2,8),mu,color=COLORS[METHODS.index(m)],label=LABELS[m]);a.fill_between(np.arange(2,8),mu-se,mu+se,color=COLORS[METHODS.index(m)],alpha=.12)
                a.set_xlabel('Arrival / growth event')
        axes[0,col].set_title(task.replace('_',' '));axes[0,col].set_ylabel('Best risk gain along direction');axes[1,col].set_ylabel('MSE after identical refit')
        if task=='lowrank_noiseless':axes[1,col].set_yscale('log')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.03),ncol=5,frameon=False)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(FIG/'quadratic_checkpoint.pdf');fig.savefig(FIG/'quadratic_checkpoint.png',dpi=180);plt.close(fig)
    print(json.dumps({'runs':report['runs'],'trajectories':report['trajectories'],'events':report['events'],'same_checkpoint_comparisons':report['same_checkpoint_comparisons'],'wall_seconds':report['wall_seconds']},indent=2))
    for r in contrasts:print(r)

if __name__=='__main__':main()

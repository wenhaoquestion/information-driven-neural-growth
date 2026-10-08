"""Generate every main empirical table/figure from saved independent runs."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import t as student
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
TASKS=['null','linear','additive','interaction']
METHODS=['residual','shuffled','random','covariance','ssd']
COLORS=dict(zip(METHODS,['#136F63','#AE493A','#3B67A0','#976EA2','#9A741F']))
def interval(v):
    a=np.asarray(v);m=float(a.mean());h=float(student.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else 0.
    return m,h
def savefig(name):
    plt.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight')
    plt.savefig(ROOT/'figures'/f'{name}.png',dpi=170,bbox_inches='tight');plt.close()
def writecsv(name,rows):
    with (ROOT/'results'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    data={k:json.loads((ROOT/'results'/f'confirm_{k}.json').read_text()) for k in ['eb4096','eb16384','mean16384']}
    rows=[];pairs=[];events=[]
    for label,d in data.items():
        for task in TASKS:
            rs=[r for r in d['runs'] if r['task']==task]
            for m in d['config']['methods']:
                vals=[r['methods'][m] for r in rs]
                avg,half=interval([v['final_risk'] for v in vals])
                row=dict(protocol=label,task=task,method=m,n=len(rs),risk=avg,risk_ci_half=half,
                         width=np.mean([v['final_width'] for v in vals]),
                         no_growth_fraction=np.mean([v['final_width']==2 for v in vals]),
                         multi_growth_fraction=np.mean([v['final_width']>=4 for v in vals]),
                         tail_risk=np.mean([v['tail_risk'] for v in vals]),
                         fixed_matched_risk=np.mean([v['fixed_matched_risk'] for v in vals]),
                         fixed_matched_tail_risk=np.mean([v['fixed_matched_tail_risk'] for v in vals]),
                         training_parameter_examples=np.mean([v['training_parameter_examples'] for v in vals]),
                         audit_parameter_examples=np.mean([v['audit_parameter_examples'] for v in vals]),
                         elapsed_seconds=np.mean([v['elapsed_seconds'] for v in vals]),audit_labels=vals[0]['audit_labels'])
                rows.append(row)
            for other in ['shuffled','random','covariance','ssd','fixed_matched','fixed_matched_tail','small','small_tail','large','large_tail','small_pool','large_pool']:
                if other in METHODS and other not in d['config']['methods']:continue
                diffs=[]
                for r in rs:
                    v=r['methods']['residual']
                    if other in METHODS:a,b=v['final_risk'],r['methods'][other]['final_risk']
                    elif other=='fixed_matched':a,b=v['final_risk'],v['fixed_matched_risk']
                    elif other=='fixed_matched_tail':a,b=v['tail_risk'],v['fixed_matched_tail_risk']
                    elif other in ['small_pool','large_pool']:a,b=v['final_risk'],r['baselines'][other+'_matched']['risk']
                    elif other.endswith('tail'):a,b=v['tail_risk'],r['baselines'][other+'_risk']
                    else:a,b=v['final_risk'],r['baselines'][other+'_risk']
                    diffs.append(a-b)
                mean,half=interval(diffs);pairs.append(dict(protocol=label,task=task,contrast='residual - '+other,n=len(rs),mean=mean,ci_half=half))
        for r in d['runs']:
            for m,v in r['methods'].items():
                for e in v['events']:
                    g=e['growth_vs_continuation'];gi=e['growth_vs_incumbent']
                    truegc=e['continuation_risk']-e['growth_risk'];truegi=e['incumbent_risk']-e['growth_risk']
                    events.append(dict(protocol=label,task=r['task'],seed=r['seed'],method=m,event=e['t'],action=e['action'],width=e['width'],
                        risk=e['retained_risk'],training_parameter_examples=e['training_parameter_examples'],audit_labels=e['audit_labels'],
                        gain_vs_continuation=truegc,gain_vs_incumbent=truegi,empirical_gain=g['mean'],movement=g['prediction_movement'],
                        variance=g['variance'],radius=g['radius'],lcb=g['lcb'],
                        positive_both=bool(min(truegc,truegi)>d['config']['tau']),
                        positive_declined=bool(min(truegc,truegi)>d['config']['tau'] and e['action']!='grow'),
                        retained_increase=e['retained_risk']-e['incumbent_risk'],
                        additive_radius=14*np.log(2/g['delta'])/(3*(g['n']-1))))
    writecsv('summary.csv',rows);writecsv('paired_effects.csv',pairs);writecsv('events.csv',events)
    # Main table in Brier loss units, no ratios.
    lines=['\\begin{tabular}{llrrrr}','\\toprule','Task & Proposal & $n_A=4096$ & $n_A=16384$ & Width & Grow $\\ge2$ \\\\','\\midrule']
    for task in TASKS:
        for m in METHODS:
            a=next(v for v in rows if v['protocol']=='eb4096' and v['task']==task and v['method']==m)
            b=next(v for v in rows if v['protocol']=='eb16384' and v['task']==task and v['method']==m)
            lines.append(f"{task.title() if m==METHODS[0] else ''} & {m.title()} & {a['risk']:.4f} & {b['risk']:.4f} & {b['width']:.2f} & {b['multi_growth_fraction']:.2f} \\\\")
        lines.append('\\addlinespace')
    lines+=['\\bottomrule','\\end{tabular}']
    (ROOT/'manuscript'/'results_table.tex').write_text('\n'.join(lines))
    lines=['\\begin{tabular}{lrrrr}','\\toprule','Comparator & Null & Linear & Additive & Interaction \\\\','\\midrule']
    for other in ['shuffled','random','covariance','ssd','fixed_matched','fixed_matched_tail','small','large','large_pool']:
        vs=[next(p for p in pairs if p['protocol']=='eb16384' and p['task']==t and p['contrast']=='residual - '+other) for t in TASKS]
        lines.append(other.replace('_',' ')+' & '+' & '.join(f"${p['mean']:+.4f}\\pm{p['ci_half']:.4f}$" for p in vs)+' \\\\')
    lines+=['\\bottomrule','\\end{tabular}']
    (ROOT/'manuscript'/'paired_table.tex').write_text('\n'.join(lines))
    # All-run trajectory means and descriptive intervals, all methods.
    fig,axs=plt.subplots(2,4,figsize=(12,5.1),sharex=True)
    for j,task in enumerate(TASKS):
        rs=[r for r in data['eb16384']['runs'] if r['task']==task]
        for m in METHODS:
            for row,key in enumerate(['retained_risk','width']):
                z=np.array([[r['initial_risk'] if row==0 else 2]+[e[key] for e in r['methods'][m]['events']] for r in rs])
                mean=z.mean(0);half=student.ppf(.975,len(rs)-1)*z.std(0,ddof=1)/np.sqrt(len(rs))
                axs[row,j].plot(range(9),mean,label=m,color=COLORS[m]);axs[row,j].fill_between(range(9),mean-half,mean+half,color=COLORS[m],alpha=.09)
        axs[0,j].set_title(task.title());axs[1,j].set_xlabel('Decision opportunity')
        axs[1,j].set_xticks([0,2,4,6,8])
    axs[0,0].set_ylabel('Held-out Brier risk');axs[1,0].set_ylabel('Retained hidden units')
    axs[0,0].legend(fontsize=7);plt.tight_layout();savefig('learning_trajectories')
    # Individual repeated decisions of primary method, all 16 seeds.
    fig,axs=plt.subplots(1,4,figsize=(11.3,3.8),sharey=True)
    from matplotlib.colors import ListedColormap
    for j,task in enumerate(TASKS):
        rs=[r for r in data['eb16384']['runs'] if r['task']==task]
        z=np.array([[{'hold':0,'continue':1,'grow':2}[e['action']] for e in r['methods']['residual']['events']] for r in rs])
        axs[j].imshow(z,aspect='auto',interpolation='none',cmap=ListedColormap(['#eeeeee','#e0bd79','#136f63']),vmin=0,vmax=2)
        axs[j].set_title(task.title());axs[j].set_xticks(range(8),range(1,9));axs[j].set_xlabel('Opportunity');axs[j].set_yticks(range(0,len(rs),3),[r['seed'] for r in rs[::3]])
    axs[0].set_ylabel('Independent seed');plt.tight_layout();savefig('decision_histories')
    # Paired attribution, both short and long-horizon controls.
    contrasts=['shuffled','random','fixed_matched','fixed_matched_tail','large_pool']
    fig,axs=plt.subplots(1,4,figsize=(12,3.5),sharey=True)
    for j,task in enumerate(TASKS):
        for i,c in enumerate(contrasts):
            p=next(v for v in pairs if v['protocol']=='eb16384' and v['task']==task and v['contrast']=='residual - '+c)
            axs[j].errorbar(p['mean'],i,xerr=p['ci_half'],fmt='o',color='#136f63',capsize=3)
        axs[j].axvline(0,color='.5',lw=.8);axs[j].set_title(task.title());axs[j].set_xlabel('Residual minus comparator risk')
    axs[0].set_yticks(range(len(contrasts)),['Shuffled','Random','Fixed final width','Both +600 steps','Fixed 10, label pool']);axs[0].invert_yaxis()
    plt.tight_layout();savefig('attribution')
    # Gate margin / movement: show every primary nonlinear proposal, rejected too.
    fig,axs=plt.subplots(1,3,figsize=(11,3.3))
    for label,color in [('eb4096','#3B67A0'),('eb16384','#136F63'),('mean16384','#AE493A')]:
        vals=[e for e in events if e['protocol']==label and e['method']=='residual' and e['task'] in ['additive','interaction']]
        axs[0].scatter([e['gain_vs_continuation'] for e in vals],[e['radius'] for e in vals],s=11,alpha=.45,color=color,label=label)
    axs[0].set_xlabel('Evaluator gain versus continuation');axs[0].set_ylabel('Empirical-Bernstein radius');axs[0].legend(fontsize=7)
    vals=[e for e in events if e['protocol']=='eb16384' and e['method']=='residual']
    for action,color in [('grow','#136f63'),('hold','#ae493a'),('continue','#9a741f')]:
        es=[e for e in vals if e['action']==action and e['gain_vs_continuation']>0]
        axs[1].scatter([e['gain_vs_continuation'] for e in es],[e['movement'] for e in es],s=12,alpha=.5,color=color,label=action)
    axs[1].plot([1e-6,.1],[1e-6,.1],'--',color='.6',lw=1);axs[1].set_xscale('log');axs[1].set_yscale('log');axs[1].set_xlabel('Positive evaluator gain');axs[1].set_ylabel('Audit prediction movement');axs[1].legend(fontsize=7)
    for i,label in enumerate(['eb4096','eb16384','mean16384']):
        for j,task in enumerate(TASKS):
            a=next(r for r in rows if r['protocol']==label and r['task']==task and r['method']=='residual')
            axs[2].scatter(j+(i-1)*.15,a['risk'],color=['#3b67a0','#136f63','#ae493a'][i],marker=['o','s','^'][i])
    axs[2].set_xticks(range(4),TASKS,rotation=25);axs[2].set_ylabel('Final retained Brier risk')
    plt.tight_layout();savefig('evidence_diagnostics')
    # Resource trajectories (training+probe proxy; audit work separately in CSV).
    fig,axs=plt.subplots(1,2,figsize=(9,3.5))
    for j,task in enumerate(['additive','interaction']):
        rs=[r for r in data['eb16384']['runs'] if r['task']==task]
        for m in METHODS:
            costs=np.array([[e['training_parameter_examples']/1e6 for e in r['methods'][m]['events']] for r in rs])
            risks=np.array([[e['retained_risk'] for e in r['methods'][m]['events']] for r in rs])
            axs[j].plot(costs.mean(0),risks.mean(0),'-o',ms=3,color=COLORS[m],label=m)
        cost=np.mean([r['methods']['residual']['training_parameter_examples']/1e6 for r in rs])
        axs[j].scatter(cost,np.mean([r['baselines']['large_pool_matched']['risk'] for r in rs]),marker='*',s=90,color='black',label='Fixed 10, full label pool')
        axs[j].set_title(task.title());axs[j].set_xlabel('Training/probe parameter-examples (millions)');axs[j].set_ylabel('Held-out Brier risk')
    axs[1].legend(fontsize=7);plt.tight_layout();savefig('resource_trajectories')
    diagnostics={}
    for label in data:
        vs=[e for e in events if e['protocol']==label and e['method']=='residual']
        diagnostics[label]={'events':len(vs),'growth_events':sum(e['action']=='grow' for e in vs),
            'positive_available':sum(e['positive_both'] for e in vs),'positive_declined':sum(e['positive_declined'] for e in vs),
            'retained_increases_over_1e3':sum(e['retained_increase']>.001 for e in vs),
            'median_additive_fraction_of_radius':float(np.median([e['additive_radius']/e['radius'] for e in vs]))}
    (ROOT/'results'/'diagnostics_summary.json').write_text(json.dumps(diagnostics,indent=2))
    print(json.dumps(diagnostics,indent=2))
if __name__=='__main__':main()

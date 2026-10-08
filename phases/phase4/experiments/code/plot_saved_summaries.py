"""Plot complete saved summaries without raw trajectories, random draws, or training."""
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
    def load(name):
        with (BASE/'results'/name).open() as f:
            rows=list(csv.DictReader(f))
        for row in rows:
            for key,value in row.items():
                if key not in {'task','method','metric','contrast','action'} and value != '':
                    row[key]=float(value)
        return rows
    perrun=load('run_summary.csv');eventrows=load('events.csv')
    summary=load('method_summary.csv');contrasts=load('paired_contrasts.csv')
    assert len(perrun)==336 and len(eventrows)==8064 and len(contrasts)==128
    index={}
    for row in perrun:
        rr=index.setdefault((row['task'],row['seed']),dict(task=row['task'],seed=row['seed'],methods={}))
        rr['methods'][row['method']]={'initial_risk':row['initial_risk'],'events':[]}
    for row in sorted(eventrows,key=lambda r:(r['task'],r['seed'],r['method'],r['event'])):
        index[row['task'],row['seed']]['methods'][row['method']]['events'].append({'retained_risk':row['retained_risk']})
    runs=list(index.values());out=BASE/'results';fig=BASE/'figures';fig.mkdir(exist_ok=True)
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

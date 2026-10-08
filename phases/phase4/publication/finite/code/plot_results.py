"""Generate all manuscript figures from saved numerical outputs (vector PDF/SVG)."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'serif','font.size':13,'axes.labelsize':14,
 'legend.fontsize':11,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
 'axes.spines.right':False,'figure.dpi':150,'savefig.bbox':'tight'})
COL=['#1b566b','#d48432','#693e83','#558856']
def read(path):
    with open(path) as f: return list(csv.DictReader(f))
def arr(rows,k):return np.array([float(r[k]) for r in rows])
def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(ROOT/'figures'/f'{name}.{ext}')
    plt.close(fig)

def main():
    (ROOT/'figures').mkdir(exist_ok=True)
    rows=read(ROOT/'results/quartet_scaling.csv'); L=arr(rows,'L')
    lines=[r'\begin{table}[ht]',r'\centering',r'\begin{tabular}{rrrrrr}',r'\toprule',
           r'$L$ & $\log_{10}w$ & $\rho_{\rm det}$ & Stochastic lower & Feasible upper & $B_0/D_3^*$\\',r'\midrule']
    for row in rows:
        if int(row['L']) in [16,64,256,1024,2048]:
            vals=[row['L'],f"{float(row['log10_rare_mass']):.1f}"]+[f"{float(row[k]):.5f}" for k in
                 ['total_price','randomized_total_price','stochastic_upper','baseline_over_opt3']]
            lines.append(' & '.join(vals)+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',
              r'\caption{Selected computed values. The deterministic column uses total risk. The stochastic columns are a variational lower bound evaluated numerically and the risk of the explicit encoder, respectively. The underlying values are saved with sixty significant decimal digits.}',
              r'\label{tab:results}',r'\end{table}']
    (ROOT/'manuscript/numerical_table.tex').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(1,2,figsize=(10,4.0))
    ax[0].loglog(L,arr(rows,'total_price'),'o-',color=COL[0],label='Best deterministic hierarchy')
    ax[0].loglog(L,arr(rows,'randomized_total_price'),'s-',color=COL[2],label='Stochastic lower bound')
    ax[0].loglog(L,arr(rows,'stochastic_upper'),'^--',color=COL[1],label='Feasible stochastic encoder')
    ax[0].loglog(L,arr(rows,'asymptotic'),':',color='black',label=r'$\sqrt{L}/(2\sqrt{\log 2})$')
    ax[0].set(xlabel=r'Rarity parameter $L=-\log w$',ylabel='Worst-level total log-loss ratio')
    ax[0].legend(loc='upper left',frameon=False)
    ax[1].semilogx(L,arr(rows,'total_price')/arr(rows,'asymptotic'),'o-',color=COL[0],label='Deterministic / predicted limit')
    ax[1].semilogx(L,2*arr(rows,'randomized_total_price')/arr(rows,'asymptotic'),'s-',color=COL[2],label='Stochastic lower / predicted limit')
    ax[1].semilogx(L,2*arr(rows,'stochastic_upper')/arr(rows,'asymptotic'),'^--',color=COL[1],label='Stochastic upper / predicted limit')
    ax[1].axhline(1,color='black',linestyle=':',linewidth=1)
    ax[1].set(xlabel=r'Rarity parameter $L$',ylabel='Ratio divided by asymptotic expression')
    ax[1].legend(loc='best',frameon=False)
    fig.tight_layout(w_pad=2); save(fig,'hierarchy_scaling')
    sens=read(ROOT/'results/quartet_sensitivity.csv'); t=arr(sens,'multiplier')
    fig,ax=plt.subplots(1,2,figsize=(10,4.0))
    ax[0].plot(t,arr(sens,'near3_ratio'),'o-',color=COL[0],label='Near-pair hierarchy')
    ax[0].plot(t,arr(sens,'rare2_ratio'),'s-',color=COL[1],label='Rare-pair hierarchy')
    ax[0].plot(t,arr(sens,'price'),'k--',label='Best hierarchy')
    ax[0].set(xlabel=r'$\delta/\sqrt{\log(2)/L}$, at $L=256$',ylabel='Worst-level excess-loss ratio')
    ax[0].legend(frameon=False,fontsize=11)
    ax[1].semilogx(L,arr(rows,'total_price'),'o-',color=COL[0],label='Logarithmic loss: total risk')
    ax[1].semilogx(L,arr(rows,'brier_price'),'s-',color=COL[3],label='Squared loss: excess risk')
    ax[1].set(xlabel=r'Rarity parameter $L$',ylabel='Best deterministic hierarchy ratio')
    ax[1].legend(frameon=False,loc='upper left')
    fig.tight_layout(w_pad=2);save(fig,'sensitivity_and_loss')
    # Schematic, explicitly separate from computed quantities.
    fig,ax=plt.subplots(figsize=(8,3.2));ax.axis('off')
    xs=[.08,.35,.65,.92]
    for y in [.25,.72]:
        for x,n in zip(xs,['A','C','D','B']):
            ax.text(x,y,n,ha='center',va='center',fontsize=13,
                    bbox=dict(boxstyle='circle,pad=0.35',fc='white',ec=COL[0],lw=1.5))
    import matplotlib.patches as patches
    for lo,hi in [(.02,.41),(.59,.98)]:
        ax.add_patch(patches.FancyBboxPatch((lo,.59),hi-lo,.26,boxstyle='round,pad=0.01',ec=COL[0],fc=COL[0],alpha=.12))
    for lo,hi in [(.02,.14),(.29,.71),(.86,.98)]:
        ax.add_patch(patches.FancyBboxPatch((lo,.12),hi-lo,.26,boxstyle='round,pad=0.01',ec=COL[1],fc=COL[1],alpha=.15))
    ax.text(.5,.96,'Optimal at 2 states: AC | DB',ha='center',fontsize=11)
    ax.text(.5,.03,'Optimal at 3 states: A | CD | B',ha='center',fontsize=11)
    ax.annotate('',xy=(.65,.4),xytext=(.65,.56),arrowprops=dict(arrowstyle='->',color=COL[1],lw=2))
    ax.text(.82,.48,'reassign mass $w$',ha='center',fontsize=10,color=COL[1])
    save(fig,'crossing_partitions_schematic')
    print('Generated 3 figures in PDF, SVG, PNG from saved CSVs (one labeled schematic).')
if __name__=='__main__':main()

"""Numerical checks for new_research/evidence/manuscript/retained_theory.tex.

Run: python phases/phase4/research/independent_checks.py
No network; no fitting/evaluation oracle used as purported finite-label data.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
BASE_SEED = 91527000


def lp_checks():
    rng = np.random.default_rng(BASE_SEED)
    rows = []
    for trial in range(100):
        m, d = 12, 4
        mu = rng.dirichlet(np.ones(m))
        B = rng.normal(size=(m, d))
        h = rng.uniform(-1, 1, m)
        e = np.zeros(d) if trial % 2 == 0 else rng.uniform(0.01, .15, d)
        A = B.T * mu
        primal = linprog(-mu*h, A_ub=np.r_[A, -A], b_ub=np.r_[e, e],
                         bounds=[(-1,1)]*m, method='highs')
        # Variables are a_plus, a_minus, residual absolute values.
        ident = np.eye(m)
        dual_A = np.r_[np.c_[B,-B,-ident], np.c_[-B,B,-ident]]
        dual = linprog(np.r_[e,e,mu], A_ub=dual_A, b_ub=np.r_[h,-h],
                       bounds=[(0,None)]*(2*d+m), method='highs')
        assert primal.success and dual.success
        gap = abs(-primal.fun - dual.fun)
        assert gap < 2e-8, (trial,gap)
        gram = B.T @ (mu[:,None]*B)
        a = np.linalg.solve(gram, B.T@(mu*h))
        r = h-B@a
        S = float(mu@r**2)
        M = float(np.max(abs(r)))
        gamma = .4*S/M
        g = gamma*r/S
        eta = (1+g)/2
        KL = float(mu@(eta*np.log(2*eta)+(1-eta)*np.log(2*(1-eta))))
        moment_error = float(np.max(abs(B.T@(mu*g))))
        gain_error = abs(float(mu@(h*g))-gamma)
        assert KL <= gamma**2/S+1e-12
        assert moment_error < 1e-10 and gain_error < 1e-10
        p, q = (1-h)/2, (1+h)/2
        risk_p = mu@(eta*(1-p)**2+(1-eta)*p**2)
        risk_q = mu@(eta*(1-q)**2+(1-eta)*q**2)
        assert abs((risk_p-risk_q)-gamma) < 1e-12
        rows.append(dict(trial=trial,primal=-primal.fun,dual=dual.fun,gap=gap,
                         residual_second_moment=S,residual_supnorm=M,gamma=gamma,
                         KL=KL,KL_bound=gamma**2/S,moment_error=moment_error,
                         gain_error=gain_error))
    (HERE/'independent_math_checks.json').write_text(json.dumps(rows,indent=2)+'\n')
    return dict(instances=100,max_lp_duality_gap=max(r['gap'] for r in rows),
                max_moment_error=max(r['moment_error'] for r in rows),
                max_gain_error=max(r['gain_error'] for r in rows))


def radius(S,M,n,alpha):
    u = math.log(1/alpha)
    return math.sqrt(2*S*u/n)+4*M*u/(3*n)


def run_simulation():
    a = .4
    b_values = [.05,.1,.2,.4]
    ns = [256,1024,4096,16384]
    M_hist = 16384
    reps = 1000
    states = np.array(list(itertools.product([-1,1], repeat=3)))
    U,V,Z = states.T
    summary = {}
    records = []
    raw_path = HERE/'independent_retention_raw.jsonl'
    out_path = HERE/'independent_retention_outcomes.csv'
    with raw_path.open('w') as raw, out_path.open('w',newline='') as out:
        fields = ['theta','rep','seed','b','fresh_n','historical_n','method',
                  'estimate','lower','true_gain','accepted','unique_labels']
        writer = csv.DictWriter(out,fieldnames=fields)
        writer.writeheader()
        for condition,theta in enumerate([0.,.2,-.2]):
            probabilities = (1+theta*V*Z)/8
            for rep in range(reps):
                seed = BASE_SEED+1000+condition*10000+rep
                rng = np.random.default_rng(seed)
                old = rng.multinomial(M_hist, probabilities)
                fresh = np.zeros(8,dtype=int)
                old_U = float(old@(U*Z))/M_hist
                raw_record = dict(theta=theta,rep=rep,seed=seed,
                                  category_order=states.tolist(),historical_counts=old.tolist(),
                                  inspections=[])
                last = 0
                for n in ns:
                    fresh += rng.multinomial(n-last, probabilities)
                    last=n
                    raw_record['inspections'].append(dict(n=n,counts=fresh.tolist()))
                    for b in b_values:
                        h = a*U+b*V
                        y_raw = float(fresh@(h*Z))/n
                        y_res = float(fresh@(b*V*Z))/n
                        y_all = float((fresh+old)@(h*Z))/(n+M_hist)
                        hist_rad = a*math.sqrt(2*math.log(2/.025)/M_hist)
                        methods = {
                            'raw_fresh':(y_raw,radius(a*a+b*b,a+b,n,.05),n),
                            'oracle_moment':(y_res,radius(b*b,b,n,.05),n),
                            'finite_history_moment':(a*old_U+y_res,hist_rad+radius(b*b,b,n,.025),n+M_hist),
                            'full_history_raw':(y_all,radius(a*a+b*b,a+b,n+M_hist,.05),n+M_hist),
                        }
                        for method,(estimate,rad,unique) in methods.items():
                            row=dict(theta=theta,rep=rep,seed=seed,b=b,fresh_n=n,
                                     historical_n=(M_hist if method in ('finite_history_moment','full_history_raw') else 0),
                                     method=method,estimate=estimate,lower=estimate-rad,
                                     true_gain=b*theta,accepted=int(estimate-rad>0),unique_labels=unique)
                            writer.writerow(row)
                            records.append(row)
                raw.write(json.dumps(raw_record,separators=(',',':'))+'\n')
    for theta in [0.,.2,-.2]:
        for b in b_values:
            for n in ns:
                for method in ['raw_fresh','oracle_moment','finite_history_moment','full_history_raw']:
                    group=[r for r in records if r['theta']==theta and r['b']==b and r['fresh_n']==n and r['method']==method]
                    est=np.array([r['estimate'] for r in group])
                    k=sum(r['accepted'] for r in group)
                    # Wilson 95% pointwise intervals, 1000 independent runs.
                    z=1.959963984540054
                    ph=k/reps
                    center=(ph+z*z/(2*reps))/(1+z*z/reps)
                    half=z*math.sqrt(ph*(1-ph)/reps+z*z/(4*reps**2))/(1+z*z/reps)
                    theoretical_variance=(a*a+b*b-(b*theta)**2)/n if method=='raw_fresh' else None
                    if method=='oracle_moment': theoretical_variance=b*b*(1-theta**2)/n
                    if method=='finite_history_moment': theoretical_variance=a*a/M_hist+b*b*(1-theta**2)/n
                    if method=='full_history_raw': theoretical_variance=(a*a+b*b-(b*theta)**2)/(M_hist+n)
                    summary[f'{theta}/{b}/{n}/{method}']=dict(theta=theta,b=b,fresh_n=n,method=method,
                        replications=reps,acceptance_rate=ph,wilson_low=center-half,wilson_high=center+half,
                        estimate_mean=float(est.mean()),estimate_variance=float(est.var(ddof=1)),
                        theoretical_variance=theoretical_variance,unique_labels=group[0]['unique_labels'])
    artifact=dict(protocol='independent_research_protocol.md',base_seed=BASE_SEED,
       replicates_per_condition=reps,total_independent_paths=3*reps,total_outcome_rows=len(records),
       oracle_side_information='E[U(2Y-1)] = 0, exact; acquisition cost undefined; no total-label savings claim',
       inference='Each inspection has its own fixed-size 0.05 guarantee. No across-inspections familywise claim.',
       summary=list(summary.values()))
    (HERE/'independent_retention_summary.json').write_text(json.dumps(artifact,indent=2)+'\n')
    return artifact


def plot(artifact):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=artifact['summary']
    fig,axs=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    colors={'raw_fresh':'#b44c42','oracle_moment':'#267a99','finite_history_moment':'#a77b19','full_history_raw':'#555b6e'}
    labels={'raw_fresh':'Fresh raw','oracle_moment':'Exact moment (oracle)',
            'finite_history_moment':'Finite retained moment','full_history_raw':'Full old + fresh data'}
    for method in colors:
        g=sorted([r for r in rows if r['theta']==.2 and r['fresh_n']==4096 and r['method']==method],key=lambda x:x['b'])
        axs[0].plot([r['b'] for r in g],[r['acceptance_rate'] for r in g],'o-',color=colors[method],label=labels[method])
        g=sorted([r for r in rows if r['theta']==.2 and r['b']==.05 and r['method']==method],key=lambda x:x['fresh_n'])
        axs[1].plot([r['fresh_n'] for r in g],[r['estimate_variance'] for r in g],'o-',color=colors[method])
        axs[1].plot([r['fresh_n'] for r in g],[r['theoretical_variance'] for r in g],'--',color=colors[method],alpha=.65)
        g=sorted([r for r in rows if r['theta']==.2 and r['b']==.05 and r['method']==method],key=lambda x:x['fresh_n'])
        axs[2].plot([r['fresh_n'] for r in g],[r['acceptance_rate'] for r in g],'o-',color=colors[method])
    axs[0].set(xlabel='Residual amplitude b',ylabel='Acceptance rate',title='4096 fresh labels; gain = 0.2 b',ylim=(-.03,1.03))
    axs[1].set(xlabel='Fresh labels',ylabel='Estimator variance',title='b = 0.05; dashed = exact variance',xscale='log',yscale='log')
    axs[2].set(xlabel='Fresh labels',ylabel='Acceptance rate',title='b = 0.05; old pool = 16384',xscale='log',ylim=(-.03,1.03))
    for ax in axs: ax.grid(alpha=.2)
    fig.legend(handles=axs[0].get_lines(),labels=list(labels.values()),loc='outside lower center',ncol=4,fontsize=9)
    fig.savefig(HERE/'independent_retention_diagnostic.pdf')
    fig.savefig(HERE/'independent_retention_diagnostic.png',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-unexecuted-simulation',action='store_true',
                        help='Execute the preserved, previously unexecuted duplicate Monte Carlo plan.')
    args=parser.parse_args()
    checks=lp_checks()
    artifact=None
    if args.run_unexecuted_simulation:
        artifact=run_simulation()
        plot(artifact)
    inventory={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
               for p in HERE.glob('independent_*') if p.is_file() and p.name!='independent_manifest.json'}
    (HERE/'independent_manifest.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(checks=checks,simulation_executed=artifact is not None,
                         paths=artifact['total_independent_paths'] if artifact else 0,
                         outcome_rows=artifact['total_outcome_rows'] if artifact else 0),indent=2))


if __name__=='__main__': main()

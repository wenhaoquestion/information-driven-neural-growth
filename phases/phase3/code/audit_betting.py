"""Independent review of the post-confirmation bounded-product betting gate.

Checks unchanged inherited training functions, exact mixture computation,
finite Bernoulli-null expectations/tails, data separation, and every saved
betting event against independently reconstructed fresh audit data.
"""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import binom
import betting_growth as betting
import sequential_growth as inherited
from audit_results import check_file

ROOT=Path(__file__).resolve().parents[1]
TASK_INDEX={'null':0,'linear':1,'additive':2,'interaction':3}


def direct_log_e(d,tau):
    lambdas=np.array([.05,.1,.2,.5,1/(1+tau)])
    # Independently sum log factors per betting stream, then log-average.
    products=np.array([np.log1p(lam*(d-tau)).sum() for lam in lambdas])
    return float(np.logaddexp.reduce(products)-np.log(5)),products


def sample_independent(rng,n,task):
    x=rng.normal(size=(n,6))
    if task=='null':g=np.zeros(n)
    elif task=='linear':g=2*x[:,0]
    elif task=='additive':g=2.5*np.tanh(2*x[:,0]+.6)-2*np.tanh(2*x[:,1]-.5)+1.7*np.tanh(2*x[:,2]+.3)-1.5*np.tanh(2*x[:,3])
    else:g=4*np.tanh(2*x[:,0])*np.tanh(2*x[:,1])+2*np.tanh(2*x[:,2])*np.tanh(2*x[:,3])
    return x,rng.binomial(1,.05+.9*expit(g))


def losses_independent(x,y,p):
    a={k:np.array(v) for k,v in p.items()}
    pp=expit(np.tanh(x@a['w']+a['b'])@a['a']+a['c'])
    return (pp-y)**2


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='betting_followup4096.json')
    ap.add_argument('--require-complete',action='store_true');args=ap.parse_args()
    unchanged=[]
    for name in ['copy','params','pack','unpack','init','forward','losses','risk','gradient','train','train_pool','add','split','sample','proposal','comparison']:
        assert inspect.getsource(getattr(betting,name))==inspect.getsource(getattr(inherited,name))
        unchanged.append(name)
    rng=np.random.default_rng(298122);delta=.05/24
    examples=[]
    for tau in [0.,.0005]:
        a=rng.random(113);b=rng.random(113)
        observed=betting.betting_comparison(a,b,delta,tau)
        direct,streams=direct_log_e(a-b,tau)
        err=abs(observed['log_e']-direct)
        assert err<1e-12
        assert np.max(np.abs(np.array(observed['log_products'])-streams))<1e-12
        examples.append(dict(tau=tau,direct_log_e=direct,maximum_log_e_error=err))
    null=[]
    # D=+1 or -1 is realizable by Brier predictions 0 and 1. At
    # P(D=1)=(1+tau)/2, its mean equals the tested null boundary tau.
    for n in [8,64,1024]:
        for tau in [0.,.0005]:
            k=np.arange(n+1);p=(1+tau)/2;lambdas=np.array([.05,.1,.2,.5,1/(1+tau)])
            logstreams=[]
            for lam in lambdas:
                low=1+lam*(-1-tau);high=1+lam*(1-tau)
                if low<=1e-15:
                    lp=np.full(n+1,-np.inf);lp[-1]=n*np.log(high)
                else:lp=k*np.log(high)+(n-k)*np.log(low)
                logstreams.append(lp)
            logs=np.logaddexp.reduce(logstreams,axis=0)-np.log(5)
            logp=binom.logpmf(k,n,p)
            logexpect=float(logsumexp(logp+logs))
            tail=float(np.exp(logsumexp(logp[logs>=-np.log(delta)]))) if np.any(logs>=-np.log(delta)) else 0.
            assert abs(logexpect)<1e-10 and tail<=delta+1e-12
            null.append(dict(n=n,tau=tau,log_expected_mixture=logexpect,false_acceptance_probability=tail,delta=delta))
    cfg=dict(ntrain=64,naudit=512,ntest=127,events=3,presteps=5,branch_steps=4,
             probe_steps=2,tail_steps=3,candidates=4,alpha=.05,tau=.0005,
             gate='betting',methods=['residual','random'])
    one=betting.run(913115,'interaction',cfg);two=betting.run(913115,'interaction',{**cfg,'ntest':251})
    for m in cfg['methods']:
        for a,b in zip(one['methods'][m]['events'],two['methods'][m]['events']):
            assert a['action']==b['action'] and a['retained_parameters']==b['retained_parameters']
    source=ROOT/'results'/args.input;data=json.loads(source.read_text());rcfg=data['config']
    generic=check_file(source);maxerr=0.;events_checked=0;test_comparisons=0
    for r in data['runs']:
        ss=np.random.SeedSequence([r['seed'],TASK_INDEX[r['task']]]).spawn(5)
        arng=np.random.default_rng(ss[2]);audits=[sample_independent(arng,rcfg['naudit'],r['task']) for _ in range(rcfg['events'])]
        for m in r['methods'].values():
            for event,(xa,ya) in zip(m['events'],audits):
                events_checked+=1
                li,lc,lg=[losses_independent(xa,ya,event[k+'_parameters']) for k in ['incumbent','continuation','growth']]
                checks=[('continuation_vs_incumbent',li-lc,0.),('growth_vs_incumbent',li-lg,rcfg['tau']),('growth_vs_continuation',lc-lg,rcfg['tau'])]
                for name,d,tau in checks:
                    saved=event[name];test_comparisons+=1
                    direct,streams=direct_log_e(d,tau)
                    maxerr=max(maxerr,abs(direct-saved['log_e']))
                    assert abs(d.mean()-saved['mean'])<1e-12
                    assert np.max(np.abs(streams-np.array(saved['log_products'])))<1e-9
                    assert saved['tested_tau']==tau and abs(saved['log_threshold']+np.log(saved['delta']))<1e-12
                    assert saved['betting_accept']==bool(direct>=-np.log(saved['delta']))
                    assert saved['betting_factor_evaluations']==5*rcfg['naudit']
    assert maxerr<1e-9
    out=dict(status='complete' if generic['complete'] else 'partial',source_file=args.input,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),code_sha256=hashlib.sha256((ROOT/'code/betting_growth.py').read_bytes()).hexdigest(),
        inherited_functions_identical=unchanged,independent_mixture_checks=examples,
        exact_Bernoulli_null_checks=null,evaluator_input_count_change_leaves_actions_parameters_unchanged=True,
        events_reconstructed=events_checked,comparisons_reconstructed=test_comparisons,
        maximum_reconstructed_log_e_error=maxerr,generic_history_cost_and_risk_audit=generic,
        scope='Computational checks support the elementary e-value proof; null enumeration is not a universal validity proof.')
    (ROOT/'results/audit_betting.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:out[k] for k in ['status','events_reconstructed','comparisons_reconstructed','maximum_reconstructed_log_e_error']},indent=2))
    if args.require_complete:assert generic['complete'],'follow-up runs incomplete'


if __name__=='__main__':main()

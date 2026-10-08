"""Independent, bounded audits of the Phase III sequential learner.

Checks derivatives, single-probe Adam ascent, realizable additions/splits,
fresh-audit decisions from regenerated data, evaluator separation, and costs.
Outputs phase3/results/audit_sequential.json; does not edit main code.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.special import expit
from sequential_growth import (gradient, init, proposal, add, split, comparison,
                               params, run, sample, unpack, train_pool, train)

ROOT=Path(__file__).resolve().parents[1]


def direct_predictions(x,p):
    return 1/(1+np.exp(-(np.tanh(x@p['w']+p['b'])@p['a']+p['c'])))


def direct_loss(x,y,p):
    return (direct_predictions(x,p)-y)**2


def finite_difference(f,arrays,epsilon=1e-5):
    out={k:np.zeros_like(v) for k,v in arrays.items()}
    for k in arrays:
        for idx in np.ndindex(arrays[k].shape):
            a={kk:vv.copy() for kk,vv in arrays.items()}
            b={kk:vv.copy() for kk,vv in arrays.items()}
            a[k][idx]+=epsilon;b[k][idx]-=epsilon
            out[k][idx]=(f(a)-f(b))/(2*epsilon)
    return out


def main():
    rng=np.random.default_rng(619221)
    x=rng.normal(size=(37,6));y=rng.integers(2,size=len(x));p=init(rng,6,2)
    g=gradient(x,y,p)
    fd=finite_difference(lambda a:direct_loss(x,y,a).mean(),p)
    gradient_error={k:float(np.max(np.abs(g[k]-fd[k]))) for k in p}
    assert max(gradient_error.values())<1e-9
    d=rng.normal(size=6);d/=np.linalg.norm(d)
    added=add(p,d,.4);split0=split(p,0,d,radius=0)
    preservation={name:float(np.max(np.abs(direct_predictions(x,a)-direct_predictions(x,p))))
                  for name,a in [('dormant_addition',added),('zero_radius_split',split0)]}
    assert max(preservation.values())<1e-14
    assert params(added)==params(p)+8==params(split0)
    pooled,account=train_pool(x,y,p,3*len(y)*params(p),np.random.default_rng(712),
                             batch=len(y),lr=.015)
    ordinary=train(x,y,p,3,lr=.015)
    pool_error=max(float(np.max(np.abs(pooled[k]-ordinary[k]))) for k in p)
    assert pool_error<1e-13 and account['unique_labels']==len(y)
    _,partial=train_pool(x,y,p,75*params(p)+params(p)-1,np.random.default_rng(713),batch=7)
    assert partial['parameter_examples']==75*params(p)
    assert partial['unique_labels']==len(y)
    probe_checks=[]
    for kind in ['residual','covariance','shuffled']:
        prng=np.random.default_rng(6104)
        w=prng.normal(0,.7,(6,4));b=prng.normal(0,.2,4)
        z=direct_predictions(x,p);r=y-z
        if kind=='shuffled':r=r[prng.permutation(len(y))]
        fac=np.ones(len(y)) if kind=='covariance' else z*(1-z)
        def objective(ab):
            v=fac[:,None]*np.tanh(x@ab['w']+ab['b'])
            if kind=='covariance':v=v-v.mean(0)
            c=(r[:,None]*v).mean(0)
            den=1 if kind=='covariance' else (v*v).mean(0)+1e-3
            return float(np.sum(c*c/den))
        fd=finite_difference(objective,{'w':w,'b':b})
        expected={k:a+.04*fd[k]/(np.abs(fd[k])+1e-8) for k,a in [('w',w),('b',b)]}
        _,meta=proposal(x,y,p,np.random.default_rng(6104),kind,k=4,steps=1,lr=.04)
        err=max(float(np.max(np.abs(np.array(meta[k])-expected[k]))) for k in ['w','b'])
        assert err<1e-7
        probe_checks.append({'kind':kind,'independent_objective_before':objective({'w':w,'b':b}),
                            'independent_objective_after':objective({k:np.array(meta[k]) for k in ['w','b']}),
                            'finite_difference_one_step_parameter_max_error':err})
    # Directly check the SSD Taylor coefficient on a nonstationary network.
    pred=direct_predictions(x,p);h=np.tanh(x@p['w']+p['b'])
    coeff=np.mean(2*(pred-y)*pred*(1-pred)*p['a'][0]*(-2*h[:,0]*(1-h[:,0]**2))*(x@d)**2)/2
    eps=.0001
    actual=float(direct_loss(x,y,split(p,0,d,radius=eps)).mean()-direct_loss(x,y,p).mean())
    assert abs(actual-eps**2*coeff)<1e-13
    delta=.05/24
    a=rng.random(100);b=rng.random(100);D=a-b
    direct_radius=(2*D.var(ddof=1)*np.log(2/delta)/len(D))**.5+14*np.log(2/delta)/(3*(len(D)-1))
    eb=comparison(a,b,delta)
    assert abs(eb['lcb']-(D.mean()-direct_radius))<1e-14
    cfg=dict(ntrain=64,naudit=96,ntest=127,events=3,presteps=5,branch_steps=4,
             probe_steps=2,tail_steps=3,candidates=4,alpha=.05,tau=.0005,
             gate='eb',methods=['residual','shuffled','random','covariance','ssd'])
    seed=880125;task='interaction'
    one=run(seed,task,cfg)
    two=run(seed,task,{**cfg,'ntest':251})
    ss=np.random.SeedSequence([seed,3]).spawn(5)
    arng=np.random.default_rng(ss[2])
    audits=[sample(arng,cfg['naudit'],task)[:2] for _ in range(cfg['events'])]
    sequence=[]
    for method in cfg['methods']:
        ee=one['methods'][method]['events'];ee2=two['methods'][method]['events']
        for e,f,(xa,ya) in zip(ee,ee2,audits):
            assert e['action']==f['action']
            assert e['retained_parameters']==f['retained_parameters']
            inc,cont,grown=[unpack(e[k+'_parameters']) for k in ['incumbent','continuation','growth']]
            li,lc,lg=[direct_loss(xa,ya,p0) for p0 in [inc,cont,grown]]
            bounds=[]
            for aa,bb in [(li,lc),(li,lg),(lc,lg)]:
                diff=aa-bb;uu=np.log(2/(cfg['alpha']/(3*cfg['events'])))
                bounds.append(diff.mean()-(2*diff.var(ddof=1)*uu/len(diff))**.5-14*uu/(3*(len(diff)-1)))
            expected='grow' if bounds[1]>cfg['tau'] and bounds[2]>cfg['tau'] else ('continue' if bounds[0]>0 else 'hold')
            assert e['action']==expected
            assert e['audit_labels']==e['t']*cfg['naudit']
            assert e['parameters']==params(unpack(e['retained_parameters']))
        sequence.append({'method':method,'actions':[e['action'] for e in ee],
                         'evaluation_input_count_change_has_no_effect_on_actions_or_parameters':True,
                         'fresh_audit_direct_reconstruction':'all match',
                         'final_parameter_count':ee[-1]['parameters']})
    # Force genuinely changing histories to check dimensions and evaluator isolation;
    # this diagnostic deliberately makes no safe-acceptance claim.
    fcfg={**cfg,'gate':'forced','methods':['residual','ssd']}
    forced=run(seed,task,fcfg);forced2=run(seed,task,{**fcfg,'ntest':251})
    for method in fcfg['methods']:
        for e,f in zip(forced['methods'][method]['events'],forced2['methods'][method]['events']):
            assert e['action']=='grow' and e['width']==2+e['t']
            assert e['retained_parameters']==f['retained_parameters']
            assert e['parameters']==1+8*e['width']
        sequence.append({'method':method,'gate':'forced_diagnostic',
                         'widths':[e['width'] for e in forced['methods'][method]['events']],
                         'evaluation_input_count_change_has_no_effect_on_growing_history':True})
    output={'status':'all assertions passed','code_sha256':hashlib.sha256((ROOT/'code/sequential_growth.py').read_bytes()).hexdigest(),
            'brier_gradient_max_errors':gradient_error,'function_preservation_max_errors':preservation,
            'persistent_pool_checks':{'full_batch_vs_train_max_parameter_error':pool_error,
                                     'full_batch_account':account,'partial_budget_account':partial},
            'probe_single_step_checks':probe_checks,'ssd_nonstationary_Taylor_check':{'epsilon':eps,
                 'actual_loss_change':actual,'quadratic_prediction':eps**2*coeff},
            'empirical_Bernstein_direct_radius':direct_radius,'sequence_checks':sequence,
            'scope':'implementation checks, not a proof of concentration or a practical performance claim'}
    (ROOT/'results/audit_sequential.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()

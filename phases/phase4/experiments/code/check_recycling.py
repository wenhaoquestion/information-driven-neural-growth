"""Independent numerical and data-access audit of saved recycling experiments."""
from pathlib import Path
import argparse, copy, hashlib, json, sys, time
import numpy as np
from scipy.special import expit
import recycling as r

def direct_prediction(x,p):
    w=np.array(p['w']);b=np.array(p['b']);a=np.array(p['a']);c=float(p['c'])
    return 1/(1+np.exp(-(np.tanh(np.matmul(x,w)+b)@a+c)))

def implementation_checks():
    rng=np.random.default_rng(421090);s=r.init(rng,3);x=rng.normal(size=(13,6));y=rng.integers(2,size=13)
    g=r.grad(x,y,s['p']);maxerr=0.
    for k in s['p']:
        for ix in np.ndindex(s['p'][k].shape):
            pp=copy.deepcopy(s['p']);pm=copy.deepcopy(s['p']);eps=1e-5
            pp[k][ix]+=eps;pm[k][ix]-=eps
            fd=(np.mean((r.predict(x,pp)-y)**2)-np.mean((r.predict(x,pm)-y)**2))/(2*eps)
            maxerr=max(maxerr,abs(fd-g[k][ix]))
    assert maxerr<1e-8,maxerr
    grown=r.grow(s,rng.normal(size=6),.3)
    assert np.allclose(r.predict(x,s['p']),r.predict(x,grown['p']),rtol=0,atol=1e-15)
    cfg=dict(ntrain=32,naudit=64,ntest=77,events=3,pre_budget=20000,event_budget=40000,tail_budget=10000,
        batch=16,lr=.01,probe_lr=.04,probe_steps=4,candidates=4,alpha=.05,tau=.0005)
    xx,yy,_,_=r.generate(741001,'interaction',cfg)
    first,_,_=r.learner(741001,'interaction','released_residual',cfg,xx,yy)
    altered=yy.copy();altered[cfg['ntrain']:]=1-altered[cfg['ntrain']:]
    second,_,_=r.learner(741001,'interaction','released_residual',cfg,xx,altered)
    assert first['events'][0]['branch_parameters']==second['events'][0]['branch_parameters']
    assert first['events'][0]['proposal']==second['events'][0]['proposal']
    future=yy.copy();future[cfg['ntrain']+cfg['naudit']:]=1-future[cfg['ntrain']+cfg['naudit']:]
    third,_,_=r.learner(741001,'interaction','released_residual',cfg,xx,future)
    for k in ['branch_parameters','proposal','tests','action','committed_hash']:
        assert first['events'][0][k]==third['events'][0][k],k
    for method in ['withheld_residual','released_residual','withheld_random','released_random','fixed2_online','fixed26_online','fixed26_gated']:
        rr,_,_=r.learner(741002,'null',method,cfg,xx,yy)
        for e in rr['events']:
            assert all(v==0 for v in e['fit_accesses_by_arrival_block'][e['t']:])
    return dict(gradient_maximum_absolute_error=maxerr,function_preserving_addition=True,
        flipping_current_and_future_audit_labels_leaves_current_candidates_identical=True,
        flipping_future_audit_labels_leaves_earlier_decision_identical=True,
        all_seven_methods_data_access_checked=True)

def audit_file(path):
    z=json.loads(path.read_text());cfg=z['config'];base=r.BASE
    data=base/z['data_file'];countsfile=base/z['access_file']
    assert r.sha(data.read_bytes())==z['data_sha256'];assert r.sha(countsfile.read_bytes())==z['access_sha256']
    assert r.jsonhash(cfg)==z['config_sha256']
    dat=np.load(data);counts=np.load(countsfile);x=dat['x'];y=dat['y'];xt=dat['evaluator_x'];q=dat['evaluator_q']
    x2,y2,xt2,q2=r.generate(z['seed'],z['task'],cfg)
    assert np.array_equal(x,x2) and np.array_equal(y,y2) and np.array_equal(xt,xt2) and np.array_equal(q,q2)
    maxerror=0.;comparisons=0;event_count=0
    for name,m in z['methods'].items():
        previous=m['initial_parameters'];previous_state=None;training=m['initial_cost']['parameter_examples'];aw=0;ae=0;bf=0;rejected=0
        for i,e in enumerate(m['events']):
            event_count+=1;start=cfg['ntrain']+i*cfg['naudit'];stop=start+cfg['naudit'];release=not name.startswith('withheld')
            assert e['audit_id_range']==[start,stop] and e['unique_observed_labels']==stop
            available=start if release else cfg['ntrain']
            assert e['fit_available_before']==available and e['fit_available_after_release']==(stop if release else cfg['ntrain'])
            assert e['release_after_decision']==release
            assert all(v==0 for v in e['fit_accesses_by_arrival_block'][i+1:])
            if not release:assert all(v==0 for v in e['fit_accesses_by_arrival_block'][1:])
            assert sum(e['fit_accesses_by_arrival_block'])==e['event_repeated_fit_accesses']
            branches=e['branch_parameters'];assert branches['incumbent']==previous
            assert e['incumbent_hash']==r.jsonhash(branches['incumbent'])
            if previous_state is not None:assert e['incumbent_state_hash']==previous_state
            assert e['frozen_hashes']=={k:r.jsonhash(v) for k,v in branches.items()}
            ls={k:(direct_prediction(x[start:stop],p)-y[start:stop])**2 for k,p in branches.items()}
            fixed=name.startswith('fixed');gated=not fixed or name.endswith('gated')
            if gated:
                pairs={'ci':('incumbent','continuation',0.)}
                if not fixed:pairs.update(gi=('incumbent','growth',cfg['tau']),gc=('continuation','growth',cfg['tau']))
                decisions={}
                for key,(a,b,tau) in pairs.items():
                    comparisons+=1;d=ls[a]-ls[b];lams=[.05,.1,.2,.5,1/(1+tau)]
                    # Independent log-average calculation, separate scalar streams.
                    lp=np.asarray([np.log1p(l*(d-tau)).sum() for l in lams])
                    le=float(np.logaddexp.reduce(lp)-np.log(5));saved=e['tests'][key]
                    alpha=cfg['alpha']/((1 if fixed else 3)*cfg['events'])
                    maxerror=max(maxerror,abs(le-saved['log_e']))
                    assert abs(le-saved['log_e'])<1e-8 and abs(float(d.mean())-saved['mean'])<1e-12
                    assert saved['alpha']==alpha and saved['tau']==tau
                    decisions[key]=le>=-np.log(alpha);assert decisions[key]==saved['accepted']
                action='grow' if not fixed and decisions['gi'] and decisions['gc'] else ('continue' if decisions['ci'] else 'hold')
                aw+=cfg['naudit']*sum(sum(np.asarray(v).size for v in p.values()) for p in branches.values())
                ae+=cfg['naudit']*len(branches);bf+=5*cfg['naudit']*len(pairs)
            else:action='continue';assert not e['tests']
            assert e['action']==action
            chosen='growth' if action=='grow' else 'continuation' if action=='continue' else 'incumbent'
            assert e['committed_branch']==chosen and e['committed_hash']==r.jsonhash(branches[chosen])
            previous=branches[chosen];previous_state=e['committed_state_hash']
            # Work counts include every fitted branch whether it was retained or not.
            cost=sum(v['parameter_examples'] for v in e['branch_cost'].values())+(e['proposal']['parameter_examples'] if e['proposal'] else 0)
            assert cost==e['event_training_parameter_examples'] and 0<=cfg['event_budget']-cost<500
            training+=cost;assert training==e['training_parameter_examples']
            rejected+=sum(v['parameter_examples'] for k,v in e['branch_cost'].items() if k!=chosen)
            assert rejected==e['cumulative_rejected_branch_parameter_examples']
            assert aw==e['audit_parameter_examples'] and ae==e['cumulative_audit_prediction_evaluations'] and bf==e['betting_factor_evaluations']
            for k,p in branches.items():
                risk=float(np.mean((direct_prediction(xt,p)-q)**2+q*(1-q)))
                assert abs(risk-e['evaluator_risks'][k])<1e-12
            assert e['retained_risk']==e['evaluator_risks'][chosen]
        assert previous==m['final_parameters'] and training==m['training_parameter_examples']
        for key,pkey in [('initial_risk','initial_parameters'),('final_risk','final_parameters'),('tail_risk','tail_parameters')]:
            risk=float(np.mean((direct_prediction(xt,m[pkey])-q)**2+q*(1-q)))
            assert abs(risk-m[key])<1e-12
        for stage,akey in [('pre_tail','pre_tail_access'),('all','all_access')]:
            cc=counts[name+'_'+stage];assert int(cc.sum())==m[akey]['repeated_fit_label_accesses']
            assert int((cc>0).sum())==m[akey]['unique_fitted_labels']
        assert m['tail_is_ungated'] is True
    return dict(file=path.name,events=event_count,comparisons=comparisons,maximum_log_e_error=maxerror,passed=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results');ap.add_argument('--output',required=True);args=ap.parse_args()
    out=dict(implementation_checks=implementation_checks(),files=[])
    if args.results:
        for p in sorted(Path(args.results).glob('*.json')):
            if p.stem.startswith(tuple(r.TASKS)):out['files'].append(audit_file(p))
    out['status']='passed';out['total_events']=sum(a['events'] for a in out['files']);out['total_comparisons']=sum(a['comparisons'] for a in out['files'])
    Path(args.output).write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='files'},indent=2))

if __name__=='__main__':main()

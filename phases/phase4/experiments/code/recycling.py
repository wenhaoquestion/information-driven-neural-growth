"""Frozen online audit-release diagnostic, NumPy CPU, no external data/services.

The learner receives only X,Y and label IDs. Evaluator probabilities are loaded
only by evaluate_method after the full action sequence has been committed.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, platform, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.special import expit, logsumexp

BASE=Path(__file__).resolve().parents[1]
TASKS={'null':0,'additive':2,'interaction':3,'radial':4}
METHODS=['withheld_residual','released_residual','withheld_random','released_random',
         'fixed2_online','fixed26_online','fixed26_gated']

def sha(b):return hashlib.sha256(b).hexdigest()
def jsonhash(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
def pack(p):return {k:v.tolist() for k,v in p.items()}
def unpack(p):return {k:np.asarray(v,dtype=float) for k,v in p.items()}
def nparams(p):return sum(v.size for v in p.values())
def phash(p):return jsonhash(pack(p))
def shash(s):return jsonhash({'p':pack(s['p']),'m':pack(s['m']),'v':pack(s['v']),'step':s['step']})
def init(rng,width):
    p=dict(w=rng.normal(0,.7,(6,width)),b=rng.normal(0,.15,width),
           a=rng.normal(0,.35,width),c=np.asarray(0.))
    return dict(p=p,m={k:np.zeros_like(v) for k,v in p.items()},
                v={k:np.zeros_like(v) for k,v in p.items()},step=0)
def predict(x,p):return expit(np.tanh(x@p['w']+p['b'])@p['a']+p['c'])
def grad(x,y,p):
    h=np.tanh(x@p['w']+p['b']);z=expit(h@p['a']+p['c'])
    r=2*(z-y)*z*(1-z)/len(y);dh=r[:,None]*p['a']*(1-h*h)
    return dict(w=x.T@dh,b=dh.sum(0),a=h.T@r,c=np.asarray(r.sum()))
def adam(s,g,lr):
    s['step']+=1;t=s['step']
    for k in s['p']:
        s['m'][k]=.9*s['m'][k]+.1*g[k];s['v'][k]=.999*s['v'][k]+.001*g[k]**2
        s['p'][k]-=lr*s['m'][k]/(1-.9**t)/(np.sqrt(s['v'][k]/(1-.999**t))+1e-8)

class Access:
    def __init__(self,n,n0,na):self.counts=np.zeros(n,dtype=np.int64);self.n0=n0;self.na=na;self.parts={}
    def add(self,ids,kind,parameters=0):
        np.add.at(self.counts,ids,1)
        d=self.parts.setdefault(kind,{'label_accesses':0,'parameter_examples':0})
        d['label_accesses']+=int(len(ids));d['parameter_examples']+=int(len(ids)*parameters)
    def blocks(self,c=None):
        c=self.counts if c is None else c
        return [int(c[:self.n0].sum())]+[int(c[i:i+self.na].sum()) for i in range(self.n0,len(c),self.na)]
    def report(self):return dict(parts=copy.deepcopy(self.parts),unique_fitted_labels=int((self.counts>0).sum()),
                                 repeated_fit_label_accesses=int(self.counts.sum()),fit_accesses_by_arrival_block=self.blocks())

def fit(s,x,y,available,budget,rng,access,kind,lr,batch):
    out=copy.deepcopy(s);pcount=nparams(out['p']);spent=0;steps=0
    order=rng.permutation(available);pos=0
    while budget-spent>=pcount:
        if pos==len(order):order=rng.permutation(available);pos=0
        take=min(batch,len(order)-pos,(budget-spent)//pcount)
        ix=order[pos:pos+take];pos+=take
        access.add(ix,kind,pcount);adam(out,grad(x[ix],y[ix],out['p']),lr)
        spent+=take*pcount;steps+=1
    return out,dict(parameter_examples=int(spent),steps=steps,parameter_count=pcount)

def grow(s,w,b):
    out=copy.deepcopy(s)
    for group in ['p','m','v']:
        a=out[group];a['w']=np.column_stack((a['w'],w if group=='p' else np.zeros(6)))
        a['b']=np.r_[a['b'],b if group=='p' else 0.];a['a']=np.r_[a['a'],0.]
    return out

def proposal(s,x,y,available,rng,kind,cfg,access):
    k=cfg['candidates'];w=rng.normal(0,.7,(6,k));b=rng.normal(0,.2,k)
    w0=w.copy();b0=b.copy();steps=cfg['probe_steps'] if kind=='residual' else 0
    m=[np.zeros_like(w),np.zeros_like(b)];v=[a.copy() for a in m];spent=0;forwards=0
    for t in range(1,steps+1):
        ix=rng.choice(available,min(cfg['batch'],len(available)),replace=False)
        access.add(ix,'proposal_fit',7*k);spent+=len(ix)*7*k
        z=predict(x[ix],s['p']);forwards+=len(ix)*nparams(s['p']);r=y[ix]-z;factor=z*(1-z)
        h=np.tanh(x[ix]@w+b);feat=factor[:,None]*h
        cov=np.mean(r[:,None]*feat,0);den=np.mean(feat**2,0)+1e-3
        df=(2*cov/den)*r[:,None]/len(ix)-2*cov**2/den**2*feat/len(ix)
        dh=df*factor[:,None]*(1-h*h)
        for j,(a,g) in enumerate(zip([w,b],[x[ix].T@dh,dh.sum(0)])):
            m[j]=.9*m[j]+.1*g;v[j]=.999*v[j]+.001*g*g
            a+=cfg['probe_lr']*m[j]/(1-.9**t)/(np.sqrt(v[j]/(1-.999**t))+1e-8)
    if kind=='random':chosen=int(rng.integers(k));scores=None
    else:
        ix=rng.choice(available,min(4096,len(available)),replace=False)
        access.add(ix,'proposal_selection',0);z=predict(x[ix],s['p']);forwards+=len(ix)*(nparams(s['p'])+7*k)
        feat=(z*(1-z))[:,None]*np.tanh(x[ix]@w+b)
        scores=(np.mean((y[ix]-z)[:,None]*feat,0)**2/(np.mean(feat**2,0)+1e-3)).tolist();chosen=int(np.argmax(scores))
    return grow(s,w[:,chosen],b[chosen]),dict(kind=kind,chosen=chosen,scores=scores,
        initial_w=w0.tolist(),initial_b=b0.tolist(),w=w.tolist(),b=b.tolist(),
        candidate_count=k,candidate_steps=steps,parameter_examples=int(spent),prediction_parameter_examples=int(forwards))

def gate(d,alpha,tau):
    lambdas=np.asarray([.05,.1,.2,.5,1/(1+tau)])
    factors=1+np.outer(d-tau,lambdas);assert factors.min()>=-1e-14
    with np.errstate(divide='ignore'):lp=np.log(np.maximum(factors,0)).sum(0)
    le=float(logsumexp(lp)-np.log(5))
    return dict(mean=float(d.mean()),variance=float(d.var(ddof=1)),tau=tau,alpha=alpha,
        log_e=le,log_products=lp.tolist(),log_threshold=float(-np.log(alpha)),accepted=bool(le>=-np.log(alpha)))

def generate(seed,task,cfg):
    ss=np.random.SeedSequence([seed,TASKS[task],404]).spawn(2)
    def sample(rng,n):
        x=rng.normal(size=(n,6))
        if task=='null':g=np.zeros(n)
        elif task=='additive':g=2.5*np.tanh(2*x[:,0]+.6)-2*np.tanh(2*x[:,1]-.5)+1.7*np.tanh(2*x[:,2]+.3)-1.5*np.tanh(2*x[:,3])
        elif task=='interaction':g=4*np.tanh(2*x[:,0])*np.tanh(2*x[:,1])+2*np.tanh(2*x[:,2])*np.tanh(2*x[:,3])
        elif task=='radial':g=3*(x[:,0]**2+x[:,1]**2-1.4)
        else:raise ValueError(task)
        q=.05+.9*expit(g);return x,rng.binomial(1,q),q
    x,y,_=sample(np.random.default_rng(ss[0]),cfg['ntrain']+cfg['events']*cfg['naudit'])
    xt,_,qt=sample(np.random.default_rng(ss[1]),cfg['ntest'])
    return x,y,xt,qt

def learner(seed,task,method,cfg,x,y):
    # No evaluator arguments, q, future decisions or teacher latent variables.
    started=time.perf_counter();n0=cfg['ntrain'];na=cfg['naudit'];T=cfg['events']
    fixed=method.startswith('fixed');width=(26 if '26' in method else 2)
    released=not method.startswith('withheld');gated=(not fixed or method.endswith('gated'))
    s=init(np.random.default_rng(np.random.SeedSequence([seed,TASKS[task],901,width])),width)
    access=Access(len(y),n0,na);available=np.arange(n0)
    s,pre=fit(s,x,y,available,cfg['pre_budget'],np.random.default_rng([seed,988,width]),access,'warmup',cfg['lr'],cfg['batch'])
    initial=pack(s['p']);events=[];training=pre['parameter_examples'];audit_work=0;evals=0;factor_evals=0;rejected=0
    for t in range(T):
        before_s=shash(s);before_p=phash(s['p']);before_counts=access.counts.copy()
        rng=np.random.default_rng(np.random.SeedSequence([seed,TASKS[task],1201,t]))
        if fixed:
            c,cost=fit(s,x,y,available,cfg['event_budget'],rng,access,'continuation_fit',cfg['lr'],cfg['batch'])
            branches={'incumbent':s,'continuation':c};proposal_meta=None;branch_cost={'continuation':cost};spent=cost['parameter_examples']
        else:
            g0,proposal_meta=proposal(s,x,y,available,rng,'residual' if method.endswith('residual') else 'random',cfg,access)
            remaining=cfg['event_budget']-proposal_meta['parameter_examples'];assert remaining>0
            c,cc=fit(s,x,y,available,remaining//2,rng,access,'continuation_fit',cfg['lr'],cfg['batch'])
            g,gc=fit(g0,x,y,available,remaining-remaining//2,rng,access,'growth_fit',cfg['lr'],cfg['batch'])
            branches={'incumbent':s,'continuation':c,'growth':g};branch_cost={'continuation':cc,'growth':gc}
            spent=proposal_meta['parameter_examples']+cc['parameter_examples']+gc['parameter_examples']
        assert shash(s)==before_s,'Branch fitting silently mutated committed predictor/state'
        # Everything compared is frozen before current audit access.
        frozen={k:phash(v['p']) for k,v in branches.items()}
        audit_start=n0+t*na;audit_ids=np.arange(audit_start,audit_start+na)
        accesses=access.counts-before_counts
        assert not np.any(accesses[audit_start:]),'Current/future labels reached candidate fitting'
        assert (accesses[len(available):]==0).all(),'Disallowed fitting access'
        tests={}
        if gated:
            ls={k:(predict(x[audit_ids],v['p'])-y[audit_ids])**2 for k,v in branches.items()}
            audit_work+=na*sum(nparams(v['p']) for v in branches.values());evals+=na*len(branches)
            alpha=cfg['alpha']/((1 if fixed else 3)*T)
            tests['ci']=gate(ls['incumbent']-ls['continuation'],alpha,0.)
            if not fixed:
                tests['gi']=gate(ls['incumbent']-ls['growth'],alpha,cfg['tau'])
                tests['gc']=gate(ls['continuation']-ls['growth'],alpha,cfg['tau'])
            factor_evals+=na*5*len(tests)
            action='grow' if not fixed and tests['gi']['accepted'] and tests['gc']['accepted'] else ('continue' if tests['ci']['accepted'] else 'hold')
        else:action='continue'
        assert {k:phash(v['p']) for k,v in branches.items()}==frozen,'Audit changed candidates'
        chosen='growth' if action=='grow' else 'continuation' if action=='continue' else 'incumbent'
        rejected_event=sum(v['parameter_examples'] for k,v in branch_cost.items() if k!=chosen)
        rejected+=rejected_event;training+=spent
        s=copy.deepcopy(branches[chosen]);committed=phash(s['p']);committed_state=shash(s)
        event=dict(t=t+1,action=action,committed_branch=chosen,width=len(s['p']['a']),parameters=nparams(s['p']),
            fit_available_before=len(available),fit_available_before_range=[0,len(available)],
            audit_id_range=[audit_start,audit_start+na],unique_observed_labels=audit_start+na,
            fit_accesses_by_arrival_block=access.blocks(accesses),event_repeated_fit_accesses=int(accesses.sum()),
            unique_fitted_labels=int((access.counts>0).sum()),training_parameter_examples=training,
            event_training_parameter_examples=spent,rejected_branch_parameter_examples=rejected_event,
            cumulative_rejected_branch_parameter_examples=rejected,audit_parameter_examples=audit_work,
            cumulative_audit_prediction_evaluations=evals,betting_factor_evaluations=factor_evals,
            branch_cost=branch_cost,proposal=proposal_meta,tests=tests,
            incumbent_state_hash=before_s,incumbent_hash=before_p,frozen_hashes=frozen,
            committed_hash=committed,committed_state_hash=committed_state,
            branch_parameters={k:pack(v['p']) for k,v in branches.items()},elapsed_seconds=time.perf_counter()-started)
        # Release occurs strictly after action and parameter commitment.
        if released:available=np.arange(audit_start+na)
        event['fit_available_after_release']=len(available)
        event['release_after_decision']=released
        assert shash(s)==committed_state,'Audit release modified the committed predictor'
        events.append(event)
    committed_final=pack(s['p']);pre_tail=access.report();counts_before_tail=access.counts.copy()
    tail,tail_cost=fit(s,x,y,available,cfg['tail_budget'],np.random.default_rng([seed,TASKS[task],773]),access,'diagnostic_tail',cfg['lr'],cfg['batch'])
    assert pack(s['p'])==committed_final,'Diagnostic tail changed the protected final checkpoint'
    return dict(method=method,seed=seed,task=task,initial_parameters=initial,events=events,
        final_parameters=committed_final,tail_parameters=pack(tail['p']),tail_is_ungated=True,
        initial_cost=pre,tail_cost=tail_cost,pre_tail_access=pre_tail,all_access=access.report(),
        training_parameter_examples=training,tail_total_parameter_examples=training+tail_cost['parameter_examples'],
        elapsed_seconds=time.perf_counter()-started),counts_before_tail,access.counts

def evaluate_method(r,xt,qt):
    # Only called after all actions for this method have been fixed.
    def risk(p):return float(np.mean((predict(xt,unpack(p))-qt)**2+qt*(1-qt)))
    r['initial_risk']=risk(r['initial_parameters']);r['final_risk']=risk(r['final_parameters']);r['tail_risk']=risk(r['tail_parameters'])
    for e in r['events']:
        e['evaluator_risks']={k:risk(p) for k,p in e['branch_parameters'].items()}
        e['retained_risk']=e['evaluator_risks'][e['committed_branch']]
    return r

def run_one(seed,task,cfg,outdir):
    outdir=Path(outdir).resolve();begin=time.perf_counter();x,y,xt,qt=generate(seed,task,cfg)
    datafile=outdir.parent/'data'/f'{task}_{seed}.npz';datafile.parent.mkdir(exist_ok=True)
    np.savez_compressed(datafile,x=x,y=y,evaluator_x=xt,evaluator_q=qt)
    results={};counts={}
    for method in cfg['methods']:
        r,pre,allc=learner(seed,task,method,cfg,x,y)
        results[method]=evaluate_method(r,xt,qt);counts[method+'_pre_tail']=pre;counts[method+'_all']=allc
    countfile=outdir.parent/'data'/f'{task}_{seed}_fit_access_counts.npz';np.savez_compressed(countfile,**counts)
    result=dict(seed=seed,task=task,config=cfg,methods=results,
        data_file=str(datafile.relative_to(BASE)),data_sha256=sha(datafile.read_bytes()),
        access_file=str(countfile.relative_to(BASE)),access_sha256=sha(countfile.read_bytes()),
        runner_sha256=sha(Path(__file__).read_bytes()),config_sha256=jsonhash(cfg),
        wall_seconds=time.perf_counter()-begin)
    outfile=outdir/f'{task}_{seed}.json';outfile.write_text(json.dumps(result,indent=2))
    return dict(task=task,seed=seed,wall_seconds=result['wall_seconds'],risks={k:round(v['final_risk'],6) for k,v in results.items()},file=outfile.name,sha256=sha(outfile.read_bytes()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text());out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    import scipy
    environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),cpu_count=os.cpu_count(),
        numpy=np.__version__,scipy=scipy.__version__,blas_threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},
        command=sys.argv,runner_sha256=sha(Path(__file__).read_bytes()),config_sha256=jsonhash(cfg))
    (out/'environment.json').write_text(json.dumps(environment,indent=2));summaries=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_one,s,t,cfg,out) for t in cfg['tasks'] for s in cfg['seeds']]
        for f in as_completed(futures):
            r=f.result();summaries.append(r);print(json.dumps(r),flush=True)
            (out/'run_index.json').write_text(json.dumps(dict(runs=summaries,completed=len(summaries),planned=len(futures),wall_seconds=time.perf_counter()-start),indent=2))
    print('COMPLETE',len(summaries),'wall_seconds',time.perf_counter()-start,flush=True)

if __name__=='__main__':main()

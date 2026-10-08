"""Repeated trainable growth with honest fresh-data decisions; NumPy CPU.

All conditional teacher probabilities are evaluator-only. Pilot/confirmation
are separate files. The proposal rules are established residual-growth ideas,
not claimed as new algorithms. Squared probability error is bounded in [0,1].
"""
from __future__ import annotations
import argparse, hashlib, json, time, sys
from pathlib import Path
import numpy as np
from scipy.special import expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]

def copy(p): return {k:v.copy() for k,v in p.items()}
def params(p): return sum(v.size for v in p.values())
def pack(p): return {k:v.tolist() for k,v in p.items()}
def unpack(p): return {k:np.asarray(v,dtype=float) for k,v in p.items()}
def init(rng,d,k):
    return dict(w=rng.normal(0,.7,(d,k)),b=rng.normal(0,.15,k),
                a=rng.normal(0,.35,k),c=np.asarray(0.))
def forward(x,p):
    h=np.tanh(x@p['w']+p['b']); return expit(h@p['a']+p['c']),h
def losses(x,y,p): return (forward(x,p)[0]-y)**2
def risk(x,q,p):
    z=forward(x,p)[0]
    return float(np.mean((z-q)**2+q*(1-q)))
def gradient(x,y,p):
    z,h=forward(x,p); r=2*(z-y)*z*(1-z)/len(y)
    dh=r[:,None]*p['a']*(1-h*h)
    return dict(w=x.T@dh,b=dh.sum(0),a=h.T@r,c=np.asarray(r.sum()))
def train(x,y,p,steps,lr=.035):
    p=copy(p); m={k:np.zeros_like(v) for k,v in p.items()}; v=copy(m)
    for t in range(1,steps+1):
        g=gradient(x,y,p)
        for k in p:
            m[k]=.9*m[k]+.1*g[k];v[k]=.999*v[k]+.001*g[k]**2
            p[k]-=lr*m[k]/(1-.9**t)/(np.sqrt(v[k]/(1-.999**t))+1e-8)
    return p
def train_pool(x,y,p,budget,rng,batch=256,lr=.015):
    """Persistent Adam; cycle fresh permutations of the entire available pool.

    Budget counts parameter-example updates. The last small batch is charged
    exactly. No claim of matched wall time or of mandatory complete pool use.
    """
    p=copy(p);m={k:np.zeros_like(v) for k,v in p.items()};v=copy(m)
    seen=np.zeros(len(y),dtype=bool);order=rng.permutation(len(y));position=0;t=0;spent=0
    while budget-spent>=params(p):
        if position==len(y):order=rng.permutation(len(y));position=0
        take=min(batch,len(y)-position,(budget-spent)//params(p))
        ix=order[position:position+take];position+=take;seen[ix]=True;t+=1
        g=gradient(x[ix],y[ix],p)
        for k in p:
            m[k]=.9*m[k]+.1*g[k];v[k]=.999*v[k]+.001*g[k]**2
            p[k]-=lr*m[k]/(1-.9**t)/(np.sqrt(v[k]/(1-.999**t))+1e-8)
        spent+=take*params(p)
    return p,dict(parameter_examples=int(spent),unique_labels=int(seen.sum()),available_labels=len(y),steps=t,batch=batch,lr=lr)
def add(p,w,b):
    p=copy(p); p['w']=np.column_stack((p['w'],w));p['b']=np.r_[p['b'],b]
    p['a']=np.r_[p['a'],0.];return p
def split(p,j,v,radius=.5):
    p=copy(p); p['w']=np.column_stack((p['w'],p['w'][:,j]));p['b']=np.r_[p['b'],p['b'][j]]
    p['a']=np.r_[p['a'],p['a'][j]/2];p['a'][j]/=2
    p['w'][:,j]+=radius*v;p['w'][:,-1]-=radius*v;return p

def sample(rng,n,task,d=6):
    x=rng.normal(size=(n,d))
    if task=='null': g=np.zeros(n)
    elif task=='linear':g=2*x[:,0]
    elif task=='additive':
        g=2.5*np.tanh(2*x[:,0]+.6)-2*np.tanh(2*x[:,1]-.5)+1.7*np.tanh(2*x[:,2]+.3)-1.5*np.tanh(2*x[:,3])
    elif task=='interaction':g=4*np.tanh(2*x[:,0])*np.tanh(2*x[:,1])+2*np.tanh(2*x[:,2])*np.tanh(2*x[:,3])
    else:raise ValueError(task)
    q=.05+.9*expit(g); y=rng.binomial(1,q)
    return x,y,q

def proposal(x,y,p,rng,kind,k=4,steps=60,lr=.04):
    """Learn K single-unit probes jointly; choose on training data only.

    'residual' uses a Gauss-Newton proxy q^2/(v+1e-3) for a logit addition.
    'covariance' maximizes squared covariance with y-p (Cascade-inspired,
    not a full Cascade-Correlation reproduction). 'shuffled' spends the same
    candidate optimization on permuted residuals. 'random' does no fitting.
    """
    z,_=forward(x,p); residual=y-z
    if kind=='ssd':
        h=forward(x,p)[1];r=2*(z-y)*z*(1-z)
        eig=[]
        for j in range(len(p['a'])):
            weights=r*p['a'][j]*(-2*h[:,j]*(1-h[:,j]**2))
            S=(x.T*weights)@x/len(y);vals,vecs=np.linalg.eigh(S)
            eig.append((float(vals[0]),j,vecs[:,0]))
        score,j,direction=min(eig,key=lambda a:a[0])
        return split(p,j,direction),{'rule':'ssd','score':score,'unit':j,'direction':direction.tolist(),'candidate_count':len(eig),'candidate_steps':0}
    w=rng.normal(0,.7,(x.shape[1],k));b=rng.normal(0,.2,k)
    if kind=='shuffled':residual=residual[rng.permutation(len(y))]
    factor=np.ones_like(z) if kind=='covariance' else z*(1-z)
    m=[np.zeros_like(w),np.zeros_like(b)];v=[a.copy() for a in m]
    n=len(y)
    for t in range(1,(0 if kind=='random' else steps)+1):
        h=np.tanh(x@w+b);feat=factor[:,None]*h
        # Remove the intercept component in covariance only (classic centering).
        if kind=='covariance':feat=feat-feat.mean(0)
        cov=np.mean(residual[:,None]*feat,0)
        den=np.ones(k) if kind=='covariance' else np.mean(feat**2,0)+1e-3
        df=(2*cov/den)*residual[:,None]/n
        if kind!='covariance':df-=2*cov**2/den**2*feat/n
        else:df-=df.mean(0)
        dh=df*factor[:,None]*(1-h*h)
        gs=[x.T@dh,dh.sum(0)]
        for j,a in enumerate([w,b]):
            m[j]=.9*m[j]+.1*gs[j];v[j]=.999*v[j]+.001*gs[j]**2
            a+=lr*m[j]/(1-.9**t)/(np.sqrt(v[j]/(1-.999**t))+1e-8)
    h=np.tanh(x@w+b);feat=factor[:,None]*h
    if kind=='covariance':feat-=feat.mean(0)
    cov=np.mean(residual[:,None]*feat,0)
    den=np.ones(k) if kind=='covariance' else np.mean(feat**2,0)+1e-3
    scores=cov**2/den
    chosen=int(rng.integers(k)) if kind=='random' else int(np.argmax(scores))
    return add(p,w[:,chosen],b[chosen]),dict(rule=kind,scores=scores.tolist(),chosen=chosen,
        w=w.tolist(),b=b.tolist(),candidate_count=k,candidate_steps=0 if kind=='random' else steps)

def comparison(a,b,delta):
    """a-b: positive means b is better. Fixed-time one-sided EB bound.

    D in [-1,1], so range length is 2. Sample variance uses ddof=1.
    Maurer-Pontil (2009), Theorem 4, applied to (1-D)/2.
    """
    d=a-b;n=len(d);mean=float(d.mean());var=float(d.var(ddof=1));u=np.log(2/delta)
    radius=float(np.sqrt(2*var*u/n)+14*u/(3*(n-1)))
    return dict(mean=mean,variance=var,radius=radius,lcb=mean-radius,n=n,delta=delta)

def betting_comparison(a,b,delta,tau):
    """Fixed-mixture nonnegative product e-value; candidate fixed before audit.

    This is an established bounded-mean betting construction. All lambda
    streams are fixed in advance, and averaging includes the mixture penalty.
    The inherited EB fields are diagnostic only when gate == betting.
    """
    result=comparison(a,b,delta)
    lambdas=np.asarray([.05,.1,.2,.5,1/(1+tau)])
    factors=1+np.outer(a-b-tau,lambdas)
    if float(factors.min()) < -1e-13:
        raise AssertionError('Betting factor violates bounded loss assumptions')
    with np.errstate(divide='ignore'):
        log_products=np.log(np.maximum(factors,0)).sum(0)
    log_e=float(logsumexp(log_products)-np.log(len(lambdas)))
    result.update(tested_tau=tau,lambdas=lambdas.tolist(),
                  log_products=log_products.tolist(),log_e=log_e,
                  log_threshold=float(-np.log(delta)),
                  betting_accept=bool(log_e>=-np.log(delta)),
                  betting_factor_evaluations=len(a)*len(lambdas))
    return result

def run(seed,task,cfg):
    # Separate RNG streams: changes in candidate fitting cannot change examples.
    streams=np.random.SeedSequence([seed,{'null':0,'linear':1,'additive':2,'interaction':3}[task]]).spawn(5)
    data_rng,init_rng,audit_rng,test_rng,probe_rng=[np.random.default_rng(s) for s in streams]
    x,y,_=sample(data_rng,cfg['ntrain'],task);xt,_,qt=sample(test_rng,cfg['ntest'],task)
    audits=[sample(audit_rng,cfg['naudit'],task)[:2] for _ in range(cfg['events'])]
    p0=init(init_rng,6,2);pstart=train(x,y,p0,cfg['presteps'])
    result=dict(seed=seed,task=task,config=cfg,start_parameters=pack(pstart),initial_risk=risk(xt,qt,pstart),methods={})
    for method in cfg['methods']:
        p=copy(pstart); events=[];elapsed=0.;cost=cfg['presteps']*len(y)*params(p);audit_cost=0;geometry_cost=0
        for t in range(cfg['events']):
            rng=np.random.default_rng(np.random.SeedSequence([seed,731,t]))
            begin=time.perf_counter();inc=copy(p)
            candidate,meta=proposal(x,y,inc,rng,method,cfg['candidates'],cfg['probe_steps'])
            cont=train(x,y,inc,cfg['branch_steps']);grown=train(x,y,candidate,cfg['branch_steps'])
            proposal_cost=meta['candidate_count']*meta['candidate_steps']*(x.shape[1]+1)*len(y)
            if method=='ssd':geometry_cost+=len(y)*len(inc['a'])*x.shape[1]**2+x.shape[1]**3*len(inc['a'])
            cost+=len(y)*cfg['branch_steps']*(params(cont)+params(grown))+proposal_cost
            # No evaluator call is used in decisions; the audit is never trained on.
            xa,ya=audits[t]; li,lc,lg=[losses(xa,ya,v) for v in [inc,cont,grown]]
            delta=cfg['alpha']/(3*cfg['events'])
            if cfg['gate']=='betting':
                ci=betting_comparison(li,lc,delta,0.)
                gi=betting_comparison(li,lg,delta,cfg['tau'])
                gc=betting_comparison(lc,lg,delta,cfg['tau'])
            else:
                ci=comparison(li,lc,delta);gi=comparison(li,lg,delta);gc=comparison(lc,lg,delta)
            pi,pc,pg=[forward(xa,v)[0] for v in [inc,cont,grown]]
            for pair,u,v in [(ci,pi,pc),(gi,pi,pg),(gc,pc,pg)]:pair['prediction_movement']=float(np.mean((u-v)**2))
            if cfg['gate']=='eb':
                action='grow' if gi['lcb']>cfg['tau'] and gc['lcb']>cfg['tau'] else ('continue' if ci['lcb']>0 else 'hold')
            elif cfg['gate']=='betting':
                action='grow' if gi['betting_accept'] and gc['betting_accept'] else ('continue' if ci['betting_accept'] else 'hold')
            elif cfg['gate']=='mean':
                action='grow' if gi['mean']>cfg['tau'] and gc['mean']>cfg['tau'] else ('continue' if ci['mean']>0 else 'hold')
            elif cfg['gate']=='forced':action='grow'
            else:raise ValueError(cfg['gate'])
            p=copy(grown if action=='grow' else cont if action=='continue' else inc)
            elapsed+=time.perf_counter()-begin
            audit_cost+=len(xa)*(params(inc)+params(cont)+params(grown))
            event=dict(t=t+1,action=action,width=len(p['a']),parameters=params(p),
                continuation_vs_incumbent=ci,growth_vs_incumbent=gi,growth_vs_continuation=gc,
                proposal=meta,training_parameter_examples=cost,audit_parameter_examples=audit_cost,
                geometry_arithmetic_proxy=geometry_cost,
                audit_labels=(t+1)*len(ya),elapsed_seconds=elapsed,
                # Evaluator-only diagnostics, computed after the action is fixed.
                incumbent_risk=risk(xt,qt,inc),continuation_risk=risk(xt,qt,cont),growth_risk=risk(xt,qt,grown),
                retained_risk=risk(xt,qt,p),incumbent_parameters=pack(inc),continuation_parameters=pack(cont),
                growth_parameters=pack(grown),retained_parameters=pack(p))
            events.append(event)
        tail=train(x,y,p,cfg['tail_steps'])
        # Fixed final capacity and proxy compute match: use deterministic independent init.
        # Width is chosen after this controller's history, not by evaluator. This is
        # a diagnostic matched-final-capacity comparator, not a deployable selector.
        fixed0=init(np.random.default_rng(np.random.SeedSequence([seed,937,len(p['a'])])),6,len(p['a']))
        matched_steps=int(cost//(len(y)*params(fixed0)))
        fixed=train(x,y,fixed0,matched_steps)
        fixed_tail=train(x,y,fixed,cfg['tail_steps'])
        result['methods'][method]=dict(events=events,final_risk=risk(xt,qt,p),final_width=len(p['a']),
            tail_risk=risk(xt,qt,tail),tail_parameters=pack(tail),training_parameter_examples=cost,
            tail_parameter_examples=cost+len(y)*cfg['tail_steps']*params(p),elapsed_seconds=elapsed,
            fixed_matched_steps=matched_steps,fixed_matched_risk=risk(xt,qt,fixed),
            fixed_matched_tail_risk=risk(xt,qt,fixed_tail),fixed_parameters=pack(fixed),
            fixed_tail_parameters=pack(fixed_tail),audit_labels=cfg['events']*cfg['naudit'],
            audit_parameter_examples=audit_cost)
    totalsteps=cfg['presteps']+cfg['events']*cfg['branch_steps']
    small=train(x,y,p0,totalsteps);small_tail=train(x,y,small,cfg['tail_steps'])
    large0=init(np.random.default_rng(np.random.SeedSequence([seed,1297])),6,2+cfg['events'])
    large=train(x,y,large0,totalsteps);large_tail=train(x,y,large,cfg['tail_steps'])
    # Grant no-growth predictor all audit labels: random-stream online minibatches,
    # same number of parameter-example updates as its original training budget.
    poolx=np.concatenate([x]+[a[0] for a in audits]);pooly=np.concatenate([y]+[a[1] for a in audits])
    extra=copy(p0);rr=np.random.default_rng(np.random.SeedSequence([seed,2291]))
    # Each batch is trained in chunks; resets are explicit, identical within this control.
    for _ in range(cfg['events']+1):
        ix=rr.choice(len(pooly),len(y),replace=False)
        extra=train(poolx[ix],pooly[ix],extra,totalsteps//(cfg['events']+1))
    result['baselines']=dict(small_risk=risk(xt,qt,small),small_tail_risk=risk(xt,qt,small_tail),
        large_risk=risk(xt,qt,large),large_tail_risk=risk(xt,qt,large_tail),
        small_all_data_risk=risk(xt,qt,extra),small_parameters=pack(small),large_parameters=pack(large),
        totalsteps=totalsteps,small_parameter_examples=totalsteps*len(y)*params(small),
        large_parameter_examples=totalsteps*len(y)*params(large),all_data_pool_size=len(pooly))
    # Give a fixed maximum-width and a fixed small model all gate labels, with
    # the normalized-residual controller's total training/probe update proxy.
    ref=result['methods'].get('residual',next(iter(result['methods'].values())))
    for name,base in [('small',p0),('large',large0)]:
        pooled,account=train_pool(poolx,pooly,base,ref['training_parameter_examples'],
                                 np.random.default_rng(np.random.SeedSequence([seed,331,name=='large'])))
        result['baselines'][name+'_pool_matched']={'risk':risk(xt,qt,pooled),'parameters':pack(pooled),**account}
    return result

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',default='pilot.json');a.add_argument('--seeds',type=int,default=3)
    a.add_argument('--seed-base',type=int,default=51000);a.add_argument('--ntrain',type=int,default=1024)
    a.add_argument('--naudit',type=int,default=4096);a.add_argument('--ntest',type=int,default=8192)
    a.add_argument('--events',type=int,default=8);a.add_argument('--presteps',type=int,default=200)
    a.add_argument('--branch-steps',type=int,default=100);a.add_argument('--probe-steps',type=int,default=60)
    a.add_argument('--tail-steps',type=int,default=600);a.add_argument('--candidates',type=int,default=4)
    a.add_argument('--alpha',type=float,default=.05);a.add_argument('--tau',type=float,default=.0005)
    a.add_argument('--gate',choices=['eb','mean','forced','betting'],default='eb')
    a.add_argument('--methods',nargs='+',default=['residual','shuffled','random','covariance','ssd'])
    a.add_argument('--tasks',nargs='+',default=['null','linear','additive','interaction']);args=a.parse_args()
    cfg=vars(args).copy();outfile=ROOT/'results'/cfg.pop('output');out=dict(config=cfg,runs=[],code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),command=sys.argv)
    start=time.perf_counter()
    for task in args.tasks:
        for seed in range(args.seed_base,args.seed_base+args.seeds):
            r=run(seed,task,cfg);out['runs'].append(r);out['wall_seconds']=time.perf_counter()-start
            outfile.write_text(json.dumps(out,indent=2))
            print(task,seed,{m:(v['final_width'],round(v['final_risk'],5)) for m,v in r['methods'].items()},flush=True)
    print('wall_seconds',out['wall_seconds'],flush=True)
if __name__=='__main__':main()

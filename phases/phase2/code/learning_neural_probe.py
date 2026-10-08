"""Exploratory neural split probe. NumPy only; no oracle enters selectors.

Run: .venv/bin/python phase2/code/learning_neural_probe.py --seeds 12
This is an exploratory mechanism diagnostic, not a tuned benchmark claim.
"""
from __future__ import annotations
import argparse, csv, json, platform, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))

def forward(x, par):
    h = np.tanh(x @ par['w'] + par['b'])
    return sigmoid(h @ par['v'] + par['c']), h

def loss(x, y, par):
    p, _ = forward(x, par)
    p = np.clip(p, 1e-9, 1-1e-9)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))

def train(x, y, par, steps, lr=.025):
    par = {k:v.copy() for k,v in par.items()}
    m = {k:np.zeros_like(v) for k,v in par.items()}
    s = {k:np.zeros_like(v) for k,v in par.items()}
    for t in range(1, steps+1):
        p, h = forward(x, par)
        r = (p-y)/len(y)
        dh = (r[:,None]*par['v'][None,:])*(1-h*h)
        g = {'w':x.T@dh, 'b':dh.sum(0), 'v':h.T@r,
             'c':np.asarray(r.sum())}
        for k in par:
            m[k] = .9*m[k]+.1*g[k]
            s[k] = .999*s[k]+.001*g[k]**2
            par[k] -= lr*(m[k]/(1-.9**t))/(np.sqrt(s[k]/(1-.999**t))+1e-8)
    return par

def init(rng, d, k):
    return {'w':rng.normal(0, .8, (d,k)), 'b':rng.normal(0,.1,k),
            'v':rng.normal(0,.5,k),'c':np.asarray(0.)}

def expand(par, j, direction, delta=.25):
    p = {k:v.copy() for k,v in par.items()}
    p['w'] = np.concatenate([p['w'], p['w'][:,j:j+1]],1)
    p['b'] = np.r_[p['b'],p['b'][j]]
    p['v'] = np.r_[p['v'],p['v'][j]/2]
    p['v'][j] /= 2
    p['w'][:,j] += delta*direction
    p['w'][:,-1] -= delta*direction
    return p

def bins(a, n=3):
    return np.searchsorted(np.quantile(a,np.arange(1,n)/n), a, side='right')

def information(y,z,q):
    # Plugin conditional MI in nats. Fixed 3x3x2 table for every candidate.
    joint = np.zeros((3,3,2))
    np.add.at(joint,(q,z,y.astype(int)),1)
    joint /= len(y)
    pq=joint.sum((1,2),keepdims=True)
    pzq=joint.sum(2,keepdims=True)
    pyq=joint.sum(1,keepdims=True)
    mask=joint>0
    ratio=np.divide(joint*pq,pzq*pyq,out=np.ones_like(joint),where=pzq*pyq>0)
    return float((joint[mask]*np.log(ratio[mask])).sum())

def entropy(z):
    p=np.bincount(z,minlength=3)/len(z)
    return float(-np.sum(p[p>0]*np.log(p[p>0])))

def sample(rng, n, regime):
    x = rng.normal(size=(n,3))
    if regime == 'localized':
        logit = 1.5*x[:,0] + 4.0*(np.tanh(2*x[:,1]+1)*np.tanh(2*x[:,2]-1))
    elif regime == 'teacher':
        w=np.array([[1.8,-1.4,.4,1.2],[.4,1.8,-1.7,.8],[-1.5,.3,1.3,-1.2]])
        logit=np.tanh(x@w+np.array([.4,-.5,.2,1.]))@np.array([2.,-2.,2.,-1.5])
    elif regime == 'linear_null':
        logit=2*x[:,0]
    else: raise ValueError(regime)
    p=.05+.90*sigmoid(logit)
    y=rng.binomial(1,p)
    return x, y, p

def safe_corr(a,b):
    if np.std(a)<1e-12 or np.std(b)<1e-12:return None
    return float(np.corrcoef(a,b)[0,1])

def run(seed,regime,ntrain,nscore,ntest,presteps,poststeps):
    rng=np.random.default_rng(seed)
    x,y,_=sample(rng,ntrain,regime)
    xs,ys,_=sample(rng,nscore,regime)
    xt,yt,pt=sample(rng,ntest,regime)
    p0=init(rng,3,2)
    par=train(x,y,p0,presteps)
    pred,h=forward(xs,par)
    q=bins(pred)
    before=loss(xt,pt,par) # noiseless conditional expectation on held-out inputs
    candidates=[]
    for j in range(2):
        for k in range(4):
            d=rng.normal(size=3);d/=np.linalg.norm(d)
            cp=expand(par,j,d)
            a=xs@par['w'][:,j]+par['b'][j]
            probe=np.tanh(a+.25*(xs@d))-np.tanh(a-.25*(xs@d))
            # Fixed absolute bins: marginal entropy varies and scale issue is explicit.
            z=np.digitize(probe,[-.08,.08])
            info=information(ys,z,q)
            aligned = par['v'][j]*(.5*(np.tanh(a+.25*(xs@d))+np.tanh(a-.25*(xs@d)))-np.tanh(a))
            aligned_info=information(ys,bins(aligned),q)
            ent=entropy(z)
            cpfit=train(x,y,cp,poststeps)
            after=loss(xt,pt,cpfit)
            psfit,_=forward(xs,cpfit)
            score_after=loss(xs,ys,cpfit)
            candidates.append({'unit':j,'direction_index':k,'direction':d.tolist(),
                'information_nats':info,'aligned_information_nats':aligned_info,'activation_entropy_nats':ent,
                'immediate_risk':loss(xt,pt,cp), 'post_risk':after,
                'fit_score_risk':score_after,'gain':before-after,
                'prediction_change':float(np.mean(np.abs(forward(xt,par)[0]-forward(xt,cpfit)[0]))),
                'parameter_displacement':float(np.sqrt(sum(np.sum((cpfit[t]-cp[t])**2) for t in cp)))})
    selector={
        'conditional_information':int(np.argmax([a['information_nats'] for a in candidates])),
        'aligned_information':int(np.argmax([a['aligned_information_nats'] for a in candidates])),
        'activation_entropy':int(np.argmax([a['activation_entropy_nats'] for a in candidates])),
        'random':int(rng.integers(len(candidates))),
        # This is privileged: all candidates receive post-fit training and same score data.
        'trained_validation':int(np.argmin([a['fit_score_risk'] for a in candidates])),
        'population_oracle':int(np.argmin([a['post_risk'] for a in candidates]))}
    continued=train(x,y,par,poststeps)
    fixed=train(x,y,init(rng,3,3),presteps+poststeps)
    # One weight update costs O(n*d*k), so width3 gets floor(2/3) pretraining steps.
    fixedcompute=train(x,y,init(rng,3,3),(2*presteps)//3+poststeps)
    result={'seed':seed,'regime':regime,'pre_risk':before,
      'continued_width2':loss(xt,pt,continued),'fixed_width3_steps':loss(xt,pt,fixed),
      'fixed_width3_parameter_compute':loss(xt,pt,fixedcompute),
      'selected':{name:{'index':i,'risk':candidates[i]['post_risk']} for name,i in selector.items()},
      'info_gain_correlation':safe_corr([a['information_nats'] for a in candidates],[a['gain'] for a in candidates]),
      'entropy_gain_correlation':safe_corr([a['activation_entropy_nats'] for a in candidates],[a['gain'] for a in candidates]),
      'candidates':candidates}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=12)
    p.add_argument('--seed-base',type=int,default=17000)
    p.add_argument('--presteps',type=int,default=300);p.add_argument('--poststeps',type=int,default=150)
    p.add_argument('--output',default='learning_neural_probe.json')
    a=p.parse_args(); t=time.time()
    out={'kind':'exploratory','config':{'seeds':a.seeds,'presteps':a.presteps,'poststeps':a.poststeps,
      'ntrain':512,'nscore':512,'ntest':8192,'width_initial':2,'width_final':3,'directions_per_unit':4,
      'lr':.025,'perturbation_delta':.25,'selection_information':'512 disjoint labelled score samples; no population probabilities',
      'evaluation':'Monte Carlo X; exact conditional Bernoulli log loss using held-out teacher probabilities; not finite noisy test-label loss',
      'compute_control':'parameter-compute approximation n*d*width*steps; candidate score overhead separate; oracle fits all candidates',
      'rng_seeds':f'{a.seed_base} + repetition index; same independent data for paired methods within each regime',
      'aligned_probe':'exact initial symmetric split logit deformation, three empirical quantile bins; correction after independent adversarial audit, retained original antisymmetric feature score'},
      'environment':{'python':platform.python_version(),'numpy':np.__version__},'runs':[]}
    for regime in ['teacher','localized','linear_null']:
        for seed in range(a.seed_base,a.seed_base+a.seeds):
            out['runs'].append(run(seed,regime,512,512,8192,a.presteps,a.poststeps))
            print(regime,seed,round(out['runs'][-1]['pre_risk'],5),flush=True)
    out['wall_seconds']=time.time()-t
    path=ROOT/'results'/a.output;path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(out,indent=2))
    for regime in ['teacher','localized','linear_null']:
        rr=[r for r in out['runs'] if r['regime']==regime]
        print(regime)
        for method in list(rr[0]['selected'])+['continued_width2','fixed_width3_steps','fixed_width3_parameter_compute']:
            vals=[r['selected'][method]['risk'] if method in r['selected'] else r[method] for r in rr]
            print(method,np.mean(vals),np.std(vals,ddof=1)/np.sqrt(len(vals)))
    print('seconds',out['wall_seconds'])
if __name__=='__main__':main()

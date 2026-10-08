"""Finite-sample neural growth at the local-observation obstruction.

All methods see the same training and disjoint validation examples. Evaluation
uses only the known finite population, never exposed to fitted selectors.
Run: .venv/bin/python phase2/code/learning_curvature_bridge.py --seeds 64
"""
from __future__ import annotations
import argparse, json, time, platform
from pathlib import Path
import numpy as np
from learning_neural_probe import forward, loss, train, expand, init
ROOT=Path(__file__).resolve().parents[1]

def base(d):
    return {'w':np.zeros((d,1)), 'b':np.array([.5]),'v':np.array([1.]),'c':np.asarray(-np.tanh(.5))}

def task(rng,d,signal=True,rotate=True):
    q=np.linalg.qr(rng.normal(size=(d,d)))[0] if rotate else np.eye(d)
    x=np.r_[q.T,-q.T]
    p=np.full(2*d,.5)
    if signal:
        p[[0,d]]+=.15;p[[1,d+1]]-=.15
    return x,p,q

def draw(rng,n,x,p):
    i=rng.integers(len(x),size=n)
    return x[i],rng.binomial(1,p[i])

def spectral(x,y,par):
    pred,h=forward(x,par)
    a=par['v'][0]*(-2*h[:,0]*(1-h[:,0]**2))*(pred-y)
    s=(x.T*a)@x/len(x)
    eig,vec=np.linalg.eigh(s)
    return vec[:,0],eig,s

def point_loss(x,y,p):
    pred=np.clip(forward(x,p)[0],1e-9,1-1e-9)
    return -y*np.log(pred)-(1-y)*np.log1p(-pred)

def validate(basepar,cand,x,y):
    # One prespecified proposal per method, conditional on its training data.
    d=point_loss(x,y,basepar)-point_loss(x,y,cand)
    gain=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(len(y)))
    return gain>1.96*se,gain,se # heuristic normal gate, not a proved confidence guarantee

def relchange(p,cp):
    return float(np.sqrt(sum(np.sum((p[k]-cp[k])**2) for k in p)))

def run(seed,regime,n,steps,presteps):
    rng=np.random.default_rng(seed)
    d=2 if regime=='four_state' else 8
    xp,pop,q=task(rng,d,regime!='no_signal',regime!='four_state')
    x,y=draw(rng,n,xp,pop);xv,yv=draw(rng,n,xp,pop)
    par=base(d)
    if regime=='random_start':
        par=init(rng,d,1)
        par=train(x,y,par,presteps,lr=.02)
    elif regime=='sample_pretrained':
        par=train(x,y,par,presteps,lr=.02)
    v,eig,s=spectral(x,y,par)
    vr=rng.normal(size=d);vr/=np.linalg.norm(vr)
    # All directions are decided before looking at validation samples.
    oracle_v,oracle_eig,_=spectral(xp,pop,par)
    dirs={'spectral':v,'random':vr,'oracle':oracle_v}
    before=loss(xp,pop,par)
    cont=train(x,y,par,steps,lr=.02)
    continued=loss(xp,pop,cont)
    events={}
    for name,direction in dirs.items():
        cp=expand(par,0,direction,delta=.6)
        fitted=train(x,y,cp,steps,lr=.02)
        accepted,vg,vse=validate(cont,fitted,xv,yv)
        immediate_gain=before-loss(xp,pop,cp)
        events[name]={'direction':direction.tolist(),'negative_axis_alignment':float((direction@q[:,1])**2),
           'initial_split_risk':loss(xp,pop,cp),'initial_gain':immediate_gain,
           'initial_curvature':float(direction@s@direction),
           'post_risk':loss(xp,pop,fitted),'gated_risk':loss(xp,pop,fitted) if accepted else continued,
           'accepted':bool(accepted),'validation_paired_gain':vg,'validation_se':vse,
           'parameter_displacement':relchange(cp,fitted),
           'prediction_change':float(np.mean(np.abs(forward(xp,par)[0]-forward(xp,fitted)[0]))),
           'post_params':{k:a.tolist() for k,a in fitted.items()}}
    # Wide controls: same post-branch updates; extra pretraining given fixed width.
    wide0=init(rng,d,2)
    fixed_steps=train(x,y,wide0,steps+presteps,lr=.02)
    fixed_compute=train(x,y,wide0,steps+(presteps*(d+3))//(2*d+5),lr=.02)
    return {'seed':seed,'regime':regime,'ntrain':n,'nvalidation':n,'poststeps':steps,
       'presteps':presteps if regime in ['random_start','sample_pretrained'] else 0,
       'population_inputs':xp.tolist(),'population_probability':pop.tolist(),
       'initial_risk':before,'continued_width1':continued,
       'fixed_width2_equal_steps':loss(xp,pop,fixed_steps),
       'fixed_width2_approx_equal_compute':loss(xp,pop,fixed_compute),
       'matrix_eigenvalues':eig.tolist(),'estimated_split_matrix':s.tolist(),
       'training_count_per_state':[(np.all(x==row,axis=1)).sum().item() for row in xp],
       'events':events,'initial_params':{k:a.tolist() for k,a in par.items()},
       'final_and_peak_parameters':{'continued':d+3,'grown':2*d+5,
          'gate_live_final':{'spectral':2*d+5 if events['spectral']['accepted'] else d+3,
                           'random':2*d+5 if events['random']['accepted'] else d+3}},
       'resource_units':{'single_grown_fit':n*steps*(2*d+5),'continued_fit':n*steps*(d+3),
         'spectral_matrix_formation':n*d*d,'spectral_eigendecomposition_order':d**3,
         'validation_pairs':n,'note':'gate trains continuation and growth branch; those costs count for both spectral and random; oracle is separately labeled'}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=64)
    p.add_argument('--output',default='learning_curvature_bridge.json')
    p.add_argument('--ns',default='256,1024');p.add_argument('--steps',default='30,300')
    a=p.parse_args();t=time.time()
    ns=[int(x) for x in a.ns.split(',')];steps=[int(x) for x in a.steps.split(',')]
    out={'kind':'finite_sample_neural_training','config':{
      'seed_base':27100,'seeds':a.seeds,'ns':ns,'poststeps':steps,'presteps':100,
      'b':.5,'eta':.15,'delta':.6,'optimizer':'full-batch Adam, lr .02, fresh moments at each branch',
      'all_parameter_training':True,'growth':'clone unit, half outgoing coefficient, opposite input-weight perturbation',
      'controller':'residual-weighted input second moment times tanh second derivative, minimum eigenvector',
      'validation_gate':'paired holdout loss improvement >1.96 estimated standard errors; heuristic normal approximation, no claimed finite-sample guarantee',
      'oracle':'minimum eigenvector of exact population split matrix at current base; separate privileged diagnostic',
      'evaluation':'exact uniform finite input population and true conditional label probabilities; training and validation labels sampled IID',
      'selection':'no test labels/population parameters to learned spectral or random controller',
      'fixed_control_warning':'For stationary regimes no pretraining was executed, so equal-step/equal-compute labels apply after excluding presteps; fixed controls currently use presteps argument 0 there',
      'measurement_warning':'operation-count proxies, not measured FLOPs; only total wallclock measured'},
      'environment':{'python':platform.python_version(),'numpy':np.__version__},'runs':[]}
    for regime in ['four_state','rotated_nuisance','sample_pretrained','random_start','no_signal']:
        for n in ns:
            for st in steps:
                for j in range(a.seeds):
                    r=run(27100+j,regime,n,st,100 if regime in ['sample_pretrained','random_start'] else 0)
                    out['runs'].append(r)
                print(regime,n,st,'spectral',np.mean([r['events']['spectral']['post_risk'] for r in out['runs'][-a.seeds:]]),flush=True)
    out['wall_seconds']=time.time()-t
    path=ROOT/'results'/a.output;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2)); print(path, out['wall_seconds'],flush=True)
if __name__=='__main__':main()

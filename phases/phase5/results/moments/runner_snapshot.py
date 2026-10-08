"""Exact Gaussian quadrature and reproducible diagnostics (not a new algorithm).
Only evaluation functions receive the planted teacher. All outputs are new files.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, time
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite import hermgauss

BASE=Path(__file__).resolve().parents[1]
def q(x,a): return np.einsum('ni,ij,nj->n',x,a,x)-np.trace(a)
def moment(x,y): return ((x.T*y)@x-np.eye(x.shape[1])*y.sum())/(2*len(x))
def variance(a,sigma=0.):
 d=len(a); return ((d*d+9*d+14)/2)*np.sum(a*a)+2*np.trace(a)**2+sigma*sigma*d*(d+1)/4

def quadrature():
 rng=np.random.default_rng(3019); roots,weights=hermgauss(5); out=[]
 for d in range(1,5):
  coords=np.array(list(itertools.product(range(5),repeat=d)))
  x=np.sqrt(2)*roots[coords]; p=np.prod(weights[coords]/np.sqrt(np.pi),axis=1)
  for k in range(5):
   a=rng.normal(size=(d,d));a=(a+a.T)/2
   h=np.einsum('ni,nj->nij',x,x)-np.eye(d)
   z=.5*q(x,a)[:,None,None]*h
   mean=np.einsum('n,nij->ij',p,z)
   empirical=np.dot(p,np.sum((z-a)**2,axis=(1,2)))
   b=rng.normal(size=(d,d));b=(b+b.T)/2
   inner=np.dot(p,q(x,a)*q(x,b))
   out.append(dict(d=d,case=k,mean_error=float(np.max(np.abs(mean-a))),
      variance=empirical,formula=variance(a),variance_error=float(abs(empirical-variance(a))),
      risk_identity_error=float(abs(inner-2*np.sum(a*b)))))
 assert max(r['mean_error'] for r in out)<1e-12
 assert max(r['variance_error'] for r in out)<1e-9
 assert max(r['risk_identity_error'] for r in out)<1e-11
 return out

def collapse():
 rng=np.random.default_rng(193); out=[]
 for d in [1,2,5,12]:
  s=rng.normal(size=(d,d));s=(s+s.T)/2;b=np.zeros_like(s)
  ev,u=np.linalg.eigh(s);order=np.argsort(-np.abs(ev));exact=np.zeros_like(s)
  for t,idx in enumerate(order):
   lam,v=np.linalg.eigh(s-b);j=np.argmax(np.abs(lam));b+=lam[j]*np.outer(v[:,j],v[:,j])
   exact+=ev[idx]*np.outer(u[:,idx],u[:,idx])
   out.append(dict(d=d,step=t+1,error=float(np.max(np.abs(b-exact)))))
 assert max(r['error'] for r in out)<1e-11
 return out

def run(output:Path,reps=64):
 output.mkdir(parents=True,exist_ok=False)
 start=time.time();config=dict(d=8,batch=512,stages=8,seeds=list(range(830000,830000+reps)),sigma=[0.,.1],eigenvalues=[1.,.3],rank_diagnostic=2)
 (output/'config.json').write_text(json.dumps(config,indent=2))
 checks=dict(quadrature=quadrature(),collapse=collapse())
 (output/'exact_checks.json').write_text(json.dumps(checks,indent=2))
 records=[];raw=output/'raw';raw.mkdir()
 for sigma in config['sigma']:
  for seed in config['seeds']:
   rng=np.random.default_rng(seed);d=config['d'];n=config['batch'];T=config['stages']
   u,_=np.linalg.qr(rng.normal(size=(d,d)));a=(u[:,:2]*np.array(config['eigenvalues']))@u[:,:2].T
   X=rng.normal(size=(T,n,d));noise=rng.normal(size=(T,n))*sigma
   Y=np.array([q(x,a) for x in X])+noise
   b=np.zeros((d,d));br=b.copy();s=b.copy();g=np.zeros((d*(d+1)//2,)*2);v=np.zeros(d*(d+1)//2)
   # Orthonormal symmetric-matrix coordinates for the explicitly higher-memory LS control.
   basis=[]
   for i in range(d):
    e=np.zeros((d,d));e[i,i]=1;basis.append(e)
   for i in range(d):
    for j in range(i):
     e=np.zeros((d,d));e[i,j]=e[j,i]=1/np.sqrt(2);basis.append(e)
   basis=np.asarray(basis);path=[]
   for t,(x,y) in enumerate(zip(X,Y),1):
    batch_s=moment(x,y);s+=batch_s
    b=b+moment(x,y-q(x,b))
    c=br+moment(x,y-q(x,br));e,w=np.linalg.eigh(c);idx=np.argsort(-np.abs(e))[:2];br=(w[:,idx]*e[idx])@w[:,idx].T
    features=np.einsum('ni,kij,nj->nk',x,basis,x)-np.trace(basis,axis1=1,axis2=2)
    g+=features.T@features;v+=features.T@y;coef=np.linalg.solve(g,v);bls=np.einsum('k,kij->ij',coef,basis)
    stale=s/t-a;lam=np.linalg.eigvalsh(stale);increase=2*np.max(np.abs(lam))**2
    row=dict(seed=seed,sigma=sigma,stage=t,n_total=t*n,
      raw_risk=2*float(np.sum((s/t-a)**2)),fresh_risk=2*float(np.sum((b-a)**2)),
      rank_known_risk=2*float(np.sum((br-a)**2)),least_squares_risk=2*float(np.sum((bls-a)**2)),
      false_birth_risk_increase=float(increase),stale_squared_error=float(np.sum(stale**2)),
      predicted_raw_risk=2*variance(a,sigma)/(t*n),
      false_birth_lower_bound=2*variance(a,0)/(d*t*n) if sigma==0 else None,
      fresh_bound=2*((75+16)/n)**t*np.sum(a*a)+(2*sigma*sigma*d*(d+1)/(4*n))*(1-((75+16)/n)**t)/(1-(75+16)/n),
      signal_frobenius_squared=float(np.sum(a*a)),trace=float(np.trace(a)))
    records.append(row);path.append([s/t,b.copy(),br.copy(),bls])
   np.savez_compressed(raw/f'sigma{sigma:g}_seed{seed}.npz',x=X,y=Y,teacher=a,estimates=np.asarray(path))
 with (output/'raw.jsonl').open('w') as f:
  for row in records:f.write(json.dumps(row)+'\n')
 summary=[]
 for sigma in config['sigma']:
  for t in range(1,config['stages']+1):
   rows=[r for r in records if r['sigma']==sigma and r['stage']==t]
   for method in ['raw_risk','fresh_risk','rank_known_risk','least_squares_risk','false_birth_risk_increase']:
    values=np.array([r[method] for r in rows]);summary.append(dict(sigma=sigma,stage=t,method=method,mean=float(values.mean()),se=float(values.std(ddof=1)/np.sqrt(len(values))),runs=len(values)))
 (output/'summary.json').write_text(json.dumps(summary,indent=2))
 (output/'execution.json').write_text(json.dumps(dict(seconds=time.time()-start,config=config,numpy=np.__version__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),unique_observed_labels_per_arm=config['batch']*config['stages'],persistent_words=dict(raw=64,fresh=128,known_rank=128,lifted_ls=36*36+36),scope='diagnostic estimators; full matrices may deploy d units; rank is privileged in known-rank arm; persistent words exclude current input/eigensolver/LS scratch'),indent=2))
 print(json.dumps(dict(output=str(output),records=len(records),seconds=time.time()-start)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--reps',type=int,default=64);args=p.parse_args();run(args.output,args.reps)

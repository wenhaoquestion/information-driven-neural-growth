"""Independent finite-difference checks for NumPy neural training and split score."""
import json
from pathlib import Path
import numpy as np
from learning_neural_probe import forward, loss, expand, init
from learning_curvature_bridge import base, spectral
ROOT=Path(__file__).resolve().parents[1]
rng=np.random.default_rng(516)
x=rng.normal(size=(17,3));y=rng.binomial(1,.4,size=17)
p=init(rng,3,2)
pr,h=forward(x,p);r=(pr-y)/len(y)
dh=(r[:,None]*p['v'][None,:])*(1-h*h)
g={'w':x.T@dh,'b':dh.sum(0),'v':h.T@r,'c':np.asarray(r.sum())}
err={}
for k in p:
 ee=[]
 for idx in np.ndindex(p[k].shape):
  pp={a:b.copy() for a,b in p.items()};pm={a:b.copy() for a,b in p.items()}
  pp[k][idx]+=1e-6;pm[k][idx]-=1e-6
  fd=(loss(x,y,pp)-loss(x,y,pm))/2e-6
  ee.append(abs(fd-g[k][idx]))
 err[k]=max(ee)
assert max(err.values())<1e-8
xp=np.array([[1.,0],[-1.,0],[0,1.],[0,-1.]])
pp=np.array([.65,.65,.35,.35]);par=base(2)
v,e,s=spectral(xp,pp,par)
assert abs(v[1])>1-1e-12
zero=expand(par,0,v,delta=0)
assert np.max(np.abs(forward(xp,zero)[0]-forward(xp,par)[0]))<1e-15
Taylor=[]
for eps in [.01,.001,.0001]:
 split=expand(par,0,v,delta=eps)
 diff=loss(xp,pp,split)-loss(xp,pp,par)
 expected=.5*eps**2*(v@s@v)
 Taylor.append({'epsilon':eps,'risk_change':diff,'predicted_quadratic':expected,'ratio':diff/expected})
assert abs(Taylor[-1]['ratio']-1)<1e-5
# The audit-corrected score is exactly the prescribed initial logit deformation.
direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
j=0;delta=.25;cp=expand(p,j,direction,delta)
a=x@p['w'][:,j]+p['b'][j]
expected=p['v'][j]*(.5*(np.tanh(a+delta*(x@direction))+np.tanh(a-delta*(x@direction)))-np.tanh(a))
_,h0=forward(x,p);_,h1=forward(x,cp)
actual=h1@cp['v']+cp['c']-h0@p['v']-p['c']
aligned_error=float(np.max(np.abs(actual-expected)))
assert aligned_error<1e-12
report={'aligned_probe_max_identity_error':aligned_error,'gradient_max_absolute_errors':err,'zero_perturbation_preserves_predictions':True,
 'population_negative_eigenvector':v.tolist(),'split_matrix':s.tolist(),'Taylor_checks':Taylor,
 'status':'passed'}
(ROOT/'results'/'learning_implementation_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

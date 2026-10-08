"""Independent reconstruction of saved diagnostic estimates and edge cases."""
from pathlib import Path
import hashlib,json
import numpy as np
root=Path(__file__).resolve().parents[1]
rows=[json.loads(t) for t in (root/'results/moments/raw.jsonl').read_text().splitlines()]
lookup={(r['sigma'],r['seed'],r['stage']):r for r in rows}
maxima={key:0. for key in ['estimates','risk','raw_prediction','fresh_bound','false_birth']}
count=0
for p in sorted((root/'results/moments/raw').glob('*.npz')):
 dat=np.load(p);xx=dat['x'];yy=dat['y'];a=dat['teacher'];stored=dat['estimates']
 sig=float(p.name.split('_')[0][5:]);seed=int(p.stem.split('seed')[1]);d=len(a);n=xx.shape[1];c=(d*d+9*d+14)/2
 def pred(x,b):return ((x@b)*x).sum(1)-np.trace(b)
 def mm(x,y):
  H=x[:,:,None]*x[:,None,:]-np.eye(d)
  return (y[:,None,None]*H).mean(0)/2
 b=np.zeros_like(a);br=np.zeros_like(a);total=np.zeros_like(a);gram=np.zeros((d*(d+1)//2,)*2);cross=np.zeros(d*(d+1)//2)
 for t,(x,y) in enumerate(zip(xx,yy),1):
  total+=mm(x,y);b+=mm(x,y-pred(x,b));tmp=br+mm(x,y-pred(x,br))
  ev,Q=np.linalg.eigh(tmp);keep=np.argsort(abs(ev))[-2:];br=(Q[:,keep]*ev[keep])@Q[:,keep].T
  cols=[x[:,i]**2-1 for i in range(d)]+[np.sqrt(2)*x[:,i]*x[:,j] for i in range(d) for j in range(i)]
  design=np.array(cols).T;gram+=design.T@design;cross+=design.T@y;coef=np.linalg.solve(gram,cross)
  ls=np.diag(coef[:d]);k=d
  for i in range(d):
   for j in range(i):ls[i,j]=ls[j,i]=coef[k]/np.sqrt(2);k+=1
  estimates=np.array([total/t,b.copy(),br.copy(),ls]);maxima['estimates']=max(maxima['estimates'],float(abs(estimates-stored[t-1]).max()))
  row=lookup[sig,seed,t]
  for j,name in enumerate(['raw_risk','fresh_risk','rank_known_risk','least_squares_risk']):
   maxima['risk']=max(maxima['risk'],abs(row[name]-2*np.sum((estimates[j]-a)**2)))
  V=c*np.sum(a*a)+2*np.trace(a)**2+sig*sig*d*(d+1)/4
  maxima['raw_prediction']=max(maxima['raw_prediction'],abs(row['predicted_raw_risk']-2*V/(t*n)))
  kap=(c+2*d)/n;nu=sig*sig*d*(d+1)/(4*n)
  bound=2*(kap**t*np.sum(a*a)+nu*(1-kap**t)/(1-kap))
  maxima['fresh_bound']=max(maxima['fresh_bound'],abs(row['fresh_bound']-bound))
  birth=2*np.linalg.norm(total/t-a,2)**2
  maxima['false_birth']=max(maxima['false_birth'],abs(row['false_birth_risk_increase']-birth))
  count+=1
# Threshold ties: strict rule is necessary for reported width bound.
b=1.;astar=np.array([[b]]);s=np.array([[2*b]])
strict=int(abs(s[0,0])>2*b);nonstrict=int(abs(s[0,0])>=2*b)
# Explicit signed spectral ties, repeated eigenvalues, and zero matrices.
edge=[]
for vals in [[2,2,-2,-2,0],[0,0,0],[1,-1],[3,1,0]]:
 S=np.diag(vals).astype(float);B=np.zeros_like(S);norms=[]
 for _ in range(np.count_nonzero(vals)):
  v,U=np.linalg.eigh(S-B);j=np.argmax(abs(v));B+=v[j]*np.outer(U[:,j],U[:,j]);norms.append(float(np.linalg.norm(S-B)))
 edge.append({'eigenvalues':vals,'final_error':float(np.linalg.norm(B-S))})
res={'source_sha256':hashlib.sha256((root/'code/moment_diagnostics.py').read_bytes()).hexdigest(),'files':len(list((root/'results/moments/raw').glob('*.npz'))),'stages':count,'rows':len(rows),'maximum_errors':{k:float(v) for k,v in maxima.items()},'spectral_edge_cases':edge,'threshold_tie_counterexample':{'A':1,'S':2,'radius':1,'strict_rank':strict,'nonstrict_rank':nonstrict,'claimed_rank_upper_bound':0},'adaptive_counterexample':'d=n=1,A*=1,sigma=0,H=X^2-1,B=H^2/2 implies T(B)=B^2 and same-block C=2B-B^2; fresh conditional formula does not apply'}
assert count==len(rows)
assert max(maxima.values())<1e-9
(root/'reviews/theory_numerical_audit.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(res,indent=2))

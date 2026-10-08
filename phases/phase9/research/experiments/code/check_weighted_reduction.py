"""Exact rational checks of the new weighted optimizer and kernel reduction.
Finite checks are not universal proof. No float tolerances are used here.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product,combinations
import datetime,hashlib,json,time
root=Path(__file__).resolve().parents[1];start=time.perf_counter()
def parts(n,k):
 def gen(a):
  if len(a)==n:
   if max(a)+1==k:yield tuple(tuple(i for i,x in enumerate(a) if x==j) for j in range(k))
   return
  for j in range(min(k-1,max(a)+1)+1):yield from gen(a+[j])
 return list(gen([0]))
p2,p3=parts(4,2),parts(4,3)
hs=[(i,j) for i,p in enumerate(p2) for j,q in enumerate(p3) if all(any(set(c)<=set(d) for d in p) for c in q)]
assert (len(p2),len(p3),len(hs))==(7,6,18)
def mul(a,b):
 c=[F(0)]*(len(a)+len(b)-1)
 for i,v in enumerate(a):
  for j,w in enumerate(b):c[i+j]+=v*w
 return c
def phi(x,g):return sum(v*x**(j+2)/((j+1)*(j+2)) for j,v in enumerate(g))
curves=[]
for a in [[1],[1,-1],[0,1],[1,-2],[1,-4,4],[0,1,-1],[1,-6,6],[1,-12,30,-20],[1,-20,90,-140,70]]:
 g=mul(list(map(F,a)),list(map(F,a)));g[0]+=F(1,1000);curves.append(g)
grids=[[F(0),b,c,F(1)] for b,c in [(F(1,100),F(99,100)),(F(1,10),F(3,10)),(F(1,10),F(9,10)),(F(1,4),F(3,4)),(F(2,5),F(3,5)),(F(7,10),F(9,10))]]
grids.append([F(1,8),F(3,8),F(5,8),F(7,8)])
rawweights=[[F(1)]*4,[F(1),F(1,100),F(1,100),F(1)],[F(1),F(1,10000),F(1,10),F(2)],[F(2),F(1,10),F(1,10000),F(1)],[F(1,100),F(10),F(1),F(1,10000)],[F(1,10000),F(1),F(10),F(1,100)]]
count=0;conflicts=[];maxrho=F(1)
for g,x,p in product(curves,grids,rawweights):
 def cell(c):
  total=sum(p[i] for i in c);m=sum(p[i]*x[i] for i in c)/total
  return sum(p[i]*phi(x[i],g) for i in c)-total*phi(m,g)
 costs={c:cell(c) for k in range(1,5) for c in combinations(range(4),k)}
 d2=[sum(costs[c] for c in q) for q in p2];d3=[sum(costs[c] for c in q) for q in p3]
 o2,o3=min(d2),min(d3);assert o2>0 and o3>0
 rho=min(max(d2[i]/o2,d3[j]/o3) for i,j in hs)
 pl,q,pr=costs[(0,1)],costs[(1,2)],costs[(2,3)]
 tl,tr=costs[(0,1,2)],costs[(1,2,3)]
 assert o3==min(pl,q,pr) and o2==min(tl,tr,pl+pr)
 assert tl>=pl+q and tr>=q+pr
 if rho>1:
  assert o3==q and o2==pl+pr
  assert rho<=min(pl,pr)/q and rho<=tl/o2 and rho<=tr/o2
  if len(conflicts)<12:conflicts.append({'points':list(map(str,x)),'weights':list(map(str,p)),'curvature':list(map(str,g)),'rho':str(rho),'rho_float':float(rho)})
 maxrho=max(maxrho,rho);count+=1
kernelchecks=0
for u,mult,w,c in product([F(1,10000),F(1,100),F(1)],[F(1),F(2),F(100),F(10000)],[F(1,10000),F(1,100),F(1),F(100)],[F(1,10000),F(1,100),F(1,4),F(1,2),F(99,100),F(9999,10000)]):
 v=u*mult;mean=(v*c+w)/(u+v+w)
 for t in sorted(set([F(j,40) for j in range(41)]+[c,mean,c/4,7*c/8,9*c/8 if 9*c/8<=1 else F(1)])):
  K=min(u*t+v*max(t-c,0),w*(1-t)+v*max(c-t,0));H=min(u*t,w*(1-t))
  q=min(u*t,v*(c-t)) if t<=c else F(0)
  p=min(v*(t-c),w*(1-t)) if t>=c else F(0)
  assert 0<=K-H<=q+p
  if 0<t<c:assert q/K>=min(F(1),(c-t)/t)
  if c<t<1:assert p/K>=(t-c)/(c+2*(t-c))
  kernelchecks+=1
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass','arithmetic':'fractions.Fraction, exact rational arithmetic','source_instances':count,'partition_counts':[7,6,18],'kernel_checks':kernelchecks,'maximum_observed_price':str(maxrho),'maximum_observed_price_float':float(maxrho),'example_conflicts':conflicts,'seconds':time.perf_counter()-start,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Finite checks of weighted flat-contiguity, middle-pair reduction and chosen-triple kernels. Does not prove universal or asymptotic results.'}
(root/'results/weighted_reduction_exact.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='example_conflicts'}))

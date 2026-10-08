"""Exact finite checks of arbitrary-N last-two-capacity quartet domination.
Enumerates ALL relevant partitions/hierarchies, not only contiguous ones.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations,product
from math import comb
import datetime,hashlib,json,platform,time
root=Path(__file__).resolve().parents[1];start=time.perf_counter()
def mul(a,b):
 c=[F(0)]*(len(a)+len(b)-1)
 for i,v in enumerate(a):
  for j,w in enumerate(b):c[i+j]+=v*w
 return c
def shapes(n):
 triples=[(tuple(t),) for t in combinations(range(n),3)]
 doubles=[]
 for a,b,c,d in combinations(range(n),4):doubles += [((a,b),(c,d)),((a,c),(b,d)),((a,d),(b,c))]
 return triples+doubles
def contiguous(shape):return all(c==tuple(range(c[0],c[-1]+1)) for c in shape)
def rgs(n,k):
 def gen(a):
  if len(a)==n:
   if max(a)+1==k:yield tuple(tuple(i for i,v in enumerate(a) if v==j) for j in range(k))
   return
  for j in range(min(k-1,max(a)+1)+1):yield from gen(a+[j])
 return list(gen([0]))
# Check the specialized enumeration against an independent full set-partition enumerator.
for n in range(4,7):
 canon=lambda ss:{tuple(sorted(c for c in shape if len(c)>1)) for shape in ss}
 assert canon(shapes(n))==canon(rgs(n,n-2))
curves=[]
for a in [[1],[1,-2],[1,-6,6],[1,-12,30,-20],[1,-20,90,-140,70]]:
 g=mul(list(map(F,a)),list(map(F,a)));g[0]+=F(1,1000);curves.append(g)
rows=[];examples=[];candidate_checks=0;hierarchy_checks=0;conflicts=0;maxrho=F(1);largest_relief=F(1)
for n in range(4,9):
 grids=[([F(i,n-1) for i in range(n)],'uniform'),([F(i*i,(n-1)**2) for i in range(n)],'quadratic'),([F(i,8*max(n-3,1)) for i in range(n-4)]+[F(1,4),F(13,50),F(37,50),F(3,4)],'quartet_with_anchors')]
 weights=[[F(1)]*n,[F(1,100) if i%2 else F(1) for i in range(n)],[F(1)]*(n-4)+[F(1),F(1,10000),F(1,10000),F(1)],[F(1)]*(n-4)+[F(2),F(1,100),F(1,10000),F(1)]]
 ss=shapes(n);assert len(ss)==comb(n,3)+3*comb(n,4)
 pairs=list(combinations(range(n),2))
 for (g,(x,grid),p) in product(curves,grids,weights):
  assert len(set(x))==n and x==sorted(x)
  def phi(z):return sum(v*z**(j+2)/((j+1)*(j+2)) for j,v in enumerate(g))
  phix=list(map(phi,x));cost={}
  for k in [2,3]:
   for c in combinations(range(n),k):
    mass=sum(p[i] for i in c);mean=sum(p[i]*x[i] for i in c)/mass
    cost[c]=sum(p[i]*phix[i] for i in c)-mass*phi(mean)
    assert cost[c]>0
  q=min(cost[e] for e in pairs)
  value=lambda shape:sum(cost[c] for c in shape)
  Q=min(map(value,ss));opts=[s for s in ss if contiguous(s) and value(s)==Q]
  cheapest=[(i,i+1) for i in range(n-1) if cost[(i,i+1)]==q]
  assert opts and cheapest
  def price(shapes,denfine,dencoarse):
   return min(max(value(s)/dencoarse,min(cost[e] for c in s for e in combinations(c,2))/denfine) for s in shapes)
  rho=price(ss,q,Q);assert rho>=1
  hierarchy_checks+=3*comb(n,3)+6*comb(n,4)
  bridges=0
  for e,s in product(cheapest,opts):
   candidate_checks+=1
   if any(set(e)<=set(c) for c in s):assert rho==1;continue
   if len(s)==1:
    fs=[f for f in combinations(s[0],2) if not set(f)&set(e) and q+cost[f]<=Q]
    assert fs and rho==1;continue
   disjoint=[c for c in s if not set(c)&set(e)]
   if disjoint:
    assert any(q+cost[c]<=Q for c in disjoint) and rho==1;continue
   a,b=sorted(s);assert a[1]+1==b[0] and e==(a[1],b[0])
   quartet=tuple(sorted(a+b));assert quartet==tuple(range(quartet[0],quartet[0]+4))
   q4=min(cost[f] for f in combinations(quartet,2))
   local=[tuple(tuple(quartet[i] for i in c) for c in shape) for shape in shapes(4)]
   Q4=min(map(value,local));assert(q4,Q4)==(q,Q)
   rho4=price(local,q4,Q4);assert rho<=rho4
   largest_relief=max(largest_relief,rho4/rho);bridges+=1
   if rho>1 and len(examples)<15:examples.append({'n':n,'grid':grid,'x':list(map(str,x)),'p':list(map(str,p)),'g':list(map(str,g)),'q':str(q),'Q':str(Q),'rho':str(rho),'rho_float':float(rho),'rho4':str(rho4),'quartet':quartet})
  if rho>1:assert bridges==len(cheapest)*len(opts);conflicts+=1
  maxrho=max(maxrho,rho)
  rows.append({'n':n,'degree':len(g)-1,'grid':grid,'weights':list(map(str,p)),'price':str(rho),'price_float':float(rho),'flat_fine':str(q),'flat_coarse':str(Q),'bridging_choices':bridges})
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass','arithmetic':'exact fractions.Fraction','python':platform.python_version(),'instances':len(rows),'n_values':list(range(4,9)),'curvature_count':len(curves),'grids_per_n':3,'weight_vectors_per_n':4,'all_hierarchy_pairs_evaluated':hierarchy_checks,'optimal_choice_cases_checked':candidate_checks,'instances_with_price_above_one':conflicts,'maximum_observed_price':str(maxrho),'maximum_observed_price_float':float(maxrho),'maximum_quartet_to_global_price_ratio':str(largest_relief),'independent_partition_enumeration_n':[4,5,6],'examples':examples,'rows':rows,'seconds':time.perf_counter()-start,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Finite check of exact quartet domination, including all noncontiguous hierarchical options. Not a universal proof, stochastic optimizer, asymptotic test or neural experiment.'}
(root/'results/top_capacity_reduction_exact.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['rows','examples']}))

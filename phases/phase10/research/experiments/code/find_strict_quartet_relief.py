"""Targeted exact diagnostic: exhibit why quartet domination is not equality.
Uses constant curvature and candidates already recorded in the core check.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations,product
import datetime,hashlib,json,time
root=Path(__file__).resolve().parents[1];start=time.perf_counter()
def shapes(ids):
 out=[(tuple(t),) for t in combinations(ids,3)]
 for a,b,c,d in combinations(ids,4):out += [((a,b),(c,d)),((a,c),(b,d)),((a,d),(b,c))]
 return out
def contiguous(s):return all(c==tuple(range(c[0],c[-1]+1)) for c in s)
prior=json.loads((root/'results/top_capacity_reduction_exact.json').read_text());found=None;checked=0
for row in prior['rows']:
 if row['degree']!=0 or row['n']==4:continue
 n=row['n'];p=list(map(F,row['weights']));grid=row['grid']
 x=([F(i,n-1) for i in range(n)] if grid=='uniform' else [F(i*i,(n-1)**2) for i in range(n)] if grid=='quadratic' else [F(i,8*max(n-3,1)) for i in range(n-4)]+[F(1,4),F(13,50),F(37,50),F(3,4)])
 # phi(t)=t²/2, direct weighted variance formula independent of polynomial code.
 cost={c:sum(p[i]*p[j]*(x[i]-x[j])**2 for i,j in combinations(c,2))/(2*sum(p[i] for i in c)) for k in[2,3] for c in combinations(range(n),k)}
 value=lambda s:sum(cost[c] for c in s)
 ss=shapes(range(n));q=min(cost[e] for e in combinations(range(n),2));Q=min(map(value,ss))
 def candidates(options):
  return [(max(value(s)/Q,cost[e]/q),s,e) for s in options for c in s for e in combinations(c,2)]
 globalbest=min(candidates(ss));rho=globalbest[0];assert str(rho)==row['price'];checked+=1
 for e,s in product([(i,i+1) for i in range(n-1) if cost[(i,i+1)]==q],[s for s in ss if len(s)==2 and contiguous(s) and value(s)==Q]):
  if any(not set(e)&set(c) for c in s):continue
  quartet=tuple(sorted(s[0]+s[1]));localbest=min(candidates(shapes(quartet)));rho4=localbest[0]
  if rho<rho4:
   found={'n':n,'curvature':'1','generator':'x^2/2','points':list(map(str,x)),'original_weights':list(map(str,p)),'flat_fine':str(q),'flat_coarse':str(Q),'cheapest_adjacent_pair':e,'selected_flat_coarse_pairs':s,'quartet':quartet,'global_price':str(rho),'quartet_price':str(rho4),'global_optimal_coarse':globalbest[1],'global_optimal_fine_pair':globalbest[2],'quartet_optimal_coarse':localbest[1],'quartet_optimal_fine_pair':localbest[2],'ratio':str(rho4/rho),'global_price_float':float(rho),'quartet_price_float':float(rho4)};break
 if found:break
assert found,'Expected strict example from previous finite check'
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass','candidate_instances_checked':checked,'example':found,'seconds':time.perf_counter()-start,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'One exact counterexample to per-instance equality; supports only the inequality reduction. Source masses are original and need not sum to1 because ratios are homogeneous.'}
(root/'results/strict_quartet_relief_exact.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

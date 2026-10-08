"""Exact check of a proposed padding counterexample, independent quadratic costs."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations
import datetime,hashlib,json,time
root=Path(__file__).resolve().parents[1];start=time.perf_counter()
def evaluate(x,p):
 n=len(x);cost={c:sum(p[i]*p[j]*(x[i]-x[j])**2 for i,j in combinations(c,2))/sum(p[i] for i in c) for k in[2,3] for c in combinations(range(n),k)}
 shapes=[(t,) for t in combinations(range(n),3)]
 for a,b,c,d in combinations(range(n),4):shapes += [((a,b),(c,d)),((a,c),(b,d)),((a,d),(b,c))]
 val=lambda s:sum(cost[c] for c in s)
 q=min(cost[e] for e in combinations(range(n),2));Q=min(map(val,shapes))
 rho=min(max(val(s)/Q,cost[e]/q) for s in shapes for c in s for e in combinations(c,2))
 return q,Q,rho
core=evaluate(list(map(F,[0,2,3,5])),[F(1)]*4);assert core==(F(1,2),F(4),F(7,6))
rows=[]
for e in [F(1,10000),F(1,100),F(1,10),F(1,2),F(99,100)]:
 q,Q,rho=evaluate(list(map(F,[-1,0,2,3,5])),[e]+[F(1)]*4)
 assert(q,Q,rho)==(e/(1+e),e/(1+e)+F(1,2),F(1))
 triple=(25*e+1)/(e+2)
 gap=triple-Q;assert gap==e*(47*e+45)/(2*(e+1)*(e+2)) and gap>0
 rows.append({'epsilon':str(e),'flat_fine':str(q),'flat_coarse':str(Q),'price':str(rho),'competing_EBC_triple_gap':str(gap)})
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass','generator':'phi=x²','core_points':['0','2','3','5'],'core_weights':['1']*4,'core_flat_fine':str(core[0]),'core_flat_coarse':str(core[1]),'core_price':str(core[2]),'padded_points':['-1','0','2','3','5'],'padded_weights':'(epsilon,1,1,1,1)','rows':rows,'seconds':time.perf_counter()-start,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Five exact finite epsilon values. General0<epsilon<1 claim belongs to separate analytic proof. Affine coordinate lift and common weight normalization preserve ratios.'}
(root/'results/tiny_mass_padding_exact.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

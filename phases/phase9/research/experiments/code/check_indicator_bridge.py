"""Exact finite-network population-risk and birth checks. No training runs."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import datetime,hashlib,json,time
root=Path(__file__).resolve().parents[1];start=time.perf_counter()
def phi(x,g):return sum(c*x**(j+2)/((j+1)*(j+2)) for j,c in enumerate(g))
def derivative(x,g):return sum(c*x**(j+1)/(j+1) for j,c in enumerate(g))
def loss(q,y,g):return phi(F(y),g)-phi(q,g)-derivative(q,g)*(F(y)-q)
def network(z,P,reports):return sum(q*max(F(0),sum(z[i] for i in c)) for c,q in zip(P,reports))
def reports(P,p,x):return [sum(p[i]*x[i] for i in c)/sum(p[i] for i in c) for c in P]
def risk(P,q,p,x,g):return sum(p[i]*(x[i]*loss(network([F(int(j==i)) for j in range(4)],P,q),1,g)+(1-x[i])*loss(network([F(int(j==i)) for j in range(4)],P,q),0,g)) for i in range(4))
curves=[[F(1)],[F(1),F(-2),F(2)],[F(1,10),F(0),F(1,5),F(0),F(1,5)]]
points=[[F(0),F(1,10),F(9,10),F(1)],[F(1,8),F(3,8),F(5,8),F(7,8)],[F(1,4)]*4]
weights=[[F(1,4)]*4,[F(49,100),F(1,100),F(1,100),F(49,100)],[F(1,1111),F(10,1111),F(100,1111),F(1000,1111)]]
paths=[[(0,1,2,3)],[(0,1),(2,3)],[(0,),(1,),(2,3)],[(0,),(1,),(2,),(3,)]]
checks=0;maxgain=F(0)
for g,x,p in product(curves,points,weights):
 bayes=sum(p[i]*(x[i]*loss(x[i],1,g)+(1-x[i])*loss(x[i],0,g)) for i in range(4))
 previous=None
 for P in paths:
  q=reports(P,p,x);cost=sum(p[i]*phi(x[i],g) for i in range(4))-sum(sum(p[i] for i in c)*phi(v,g) for c,v in zip(P,q))
  R=risk(P,q,p,x,g);assert R-bayes==cost
  if previous:
   oldP,oldq,oldR=previous
   copied=[oldq[next(k for k,c in enumerate(oldP) if set(child)<=set(c))] for child in P]
   for z in [[F(int(j==i)) for j in range(4)] for i in range(4)]+[[F(1,4)]*4]:assert network(z,P,copied)==network(z,oldP,oldq)
   assert risk(P,copied,p,x,g)==oldR and R<=oldR;maxgain=max(maxgain,oldR-R)
  previous=(P,q,R);checks+=1
 # One unrestricted ReLU unit with weights x fits all population posteriors.
 unrestricted=[max(F(0),x[i]) for i in range(4)]
 assert unrestricted==x
 assert sum(p[i]*(x[i]*loss(unrestricted[i],1,g)+(1-x[i])*loss(unrestricted[i],0,g)) for i in range(4))==bayes
birth_excess=sum(F(1,2)*(v-F(1,2))**2 for v in [F(1,4),F(3,4)])
assert birth_excess==F(1,16)
a,w,step=F(0),F(1),F(3,4)
ga,gw=4*w*w*(a*w*w-1),8*a*w*(a*w*w-1)
old=2*(a*w*w-1)**2
a,w=a-step*ga,w-step*gw
new=2*(a*w*w-1)**2
assert (old,new,a,w,ga,gw)==(2,8,3,1,-4,0)
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass','arithmetic':'exact fractions.Fraction','finite_populations':27,'population_oracle_states_checked':checks,'split_transitions_checked':81,'unrestricted_one_unit_bayes_fits':27,'max_observed_oracle_split_gain':str(maxgain),'birth_only_example':{'points':['1/4','3/4'],'weights':['1/2','1/2'],'loss':'squared loss','old_report':'1/2','excess_before':'1/16','excess_after_copied_birth':'1/16','new_flat_optimum':'0'},'phase6_gradient_counterexample':{'excess_before':'2','parameters_before':['a=0','w=1'],'learning_rate':'3/4','population_gradient':['-4','0'],'parameters_after':['a=3','w=1'],'excess_after':'8','optimizer_scope':'one ordinary simultaneous gradient step; not Adam, not a training benchmark'},'seconds':time.perf_counter()-start,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Known finite population posteriors, restricted indicator weights and exact oracle reports only; no empirical estimation, SGD training or generalization experiment.'}
(root/'results/indicator_bridge_exact.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

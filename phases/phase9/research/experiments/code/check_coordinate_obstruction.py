"""Finite diagnostics for the proved same-generator h-comparison obstruction.
Both quadrature orders are floating point, not interval certificates.
"""
from pathlib import Path
from functools import lru_cache
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
import numpy as np
import scipy
from scipy.special import roots_legendre
import csv,datetime,hashlib,json,platform,sys,time
root=Path(__file__).resolve().parents[1];start=time.perf_counter()
@lru_cache(None)
def gauss(n):return roots_legendre(n)
def values(t,m):
 t=np.asarray(t,float)
 F=np.sinc(m*np.arcsin(np.sqrt(t))/np.pi)**2*(np.arcsin(np.sqrt(t))/np.sqrt(t))**2
 # The sinc form equals sin²(m asin sqrt(t))/(m² t) and is stable near zero.
 with np.errstate(divide='ignore',invalid='ignore'):
  q=-np.expm1(m*np.log(t))/(1-t)
 q=np.where(t==1,m,q);q=np.where(t==0,1,q)
 F=np.where(t==0,1,F)
 return m**4*F**4+q*q
def integral(m,a,b,func,order,extra=()):
 u,w=gauss(order)
 knots={a,b,*[float(np.sin(k*np.pi/m)**2) for k in range(m//2+1)]}
 for j in range(40):
  t=2.**j/(16*m*m)
  if t<1:knots.add(t)
  t=2.**j/(16*m)
  if t<1:knots.add(1-t)
 knots.update(extra);knots=sorted(x for x in knots if a<=x<=b)
 out=0.
 for left,right in zip(knots[:-1],knots[1:]):
  t=(left+right)/2+(right-left)*u/2
  out+=(right-left)/2*np.dot(w,func(t,values(t,m)))
 return float(out)
def calculate(m,order):
 A=integral(m,0.,.5,lambda t,g:np.sqrt(g),order)
 C=integral(m,.5,1.,lambda t,g:np.sqrt(g),order)
 B=integral(m,0.,.5,lambda t,g:(.5-t)*g,order)
 E=integral(m,.5,1.,lambda t,g:(1-t)*g,order)
 eta=1e-3/m**4;p=np.array([1.,eta,eta**2]);p/=sum(p)
 x=[0.,.5,1.]
 def pair(i,j):
  a,b=x[i],x[j];u,v=p[i],p[j];mean=a+v/(u+v)*(b-a)
  return integral(m,a,b,lambda t,g:np.minimum(u*(t-a),v*(b-t))*g,order,[mean])
 D01,D12,D02=pair(0,1),pair(1,2),pair(0,2)
 V01=p[0]*p[1]/(p[0]+p[1])*A*A
 V12=p[1]*p[2]/(p[1]+p[2])*C*C
 M=m**4+m*m
 return {'m':m,'curvature_degree':4*m-4,'eta':eta,'quadrature_order':order,'forward_B_over_h2':B/(A*A),'inverse_h2_over_B':C*C/E,'pointwise_product':B/(A*A)*C*C/E,'partition_forward_D_over_V':D01/V01,'partition_inverse_V_over_D':V12/D12,'partition_product_lower':D01/V01*V12/D12,'ratio_product_over_r2logr':(D01/V01*V12/D12)/((4*m-4)**2*np.log(4*m-4)),'normalized_D01':D01/M,'normalized_D12':D12/M,'normalized_D02':D02/M,'normalized_OPT2':min(D01,D12,D02)/M,'three_state_hierarchy_price':1.0,'forward_limit_relative_difference':abs(D01/V01/(B/(A*A))-1),'inverse_limit_relative_difference':abs(V12/D12/(C*C/E)-1),'A_left':A,'A_right':C,'B_left':B,'B_right':E}
rows=[]
for m in [4,8,16,32,64,128,256,512]:
 a=calculate(m,96);b=calculate(m,192)
 checks=['forward_B_over_h2','inverse_h2_over_B','partition_product_lower','normalized_D01','normalized_D12','normalized_OPT2']
 err=max(abs(a[k]/b[k]-1) for k in checks)
 assert err<1e-7,(m,err)
 b['relative_two_order_discrepancy']=err;rows.append(b)
 print(json.dumps({'m':m,'product':b['partition_product_lower'],'normalized_OPT2':b['normalized_OPT2'],'discrepancy':err}),flush=True)
with (root/'results/coordinate_obstruction.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass','seconds':time.perf_counter()-start,'rows':rows,'generator':'G_m(t)=m^4 F_m(t)^4+q_m(1-t)^2; F_m=[1-T_m(1-2t)]/(2m^2t), q_m(s)=[1-(1-s)^m]/s','source':'locations(0,1/2,1), weights proportional(1,eta,eta²), eta=1e-3/m^4','normalization':'divide curvature by analytic global upper m^4+m²; all comparison ratios unchanged; canonical Bregman proper losses then in[0,1/2]','hierarchy_price_note':'For three source points, choose a flat-optimal two-cell partition between unique one-cell and singleton three-cell levels. Price is identically1; reported distortion product is not a hierarchy lower bound.','precision':'96 and192 Gauss nodes per analytic-break panel; observed discrepancy only, not interval certification.','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'environment':{'python':sys.version,'executable':sys.executable,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'threads':1,'processes':1}}
(root/'results/coordinate_obstruction.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':'pass','rows':len(rows),'seconds':out['seconds'],'max_discrepancy':max(x['relative_two_order_discrepancy'] for x in rows)}),flush=True)

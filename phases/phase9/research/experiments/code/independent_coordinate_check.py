"""Independent high-precision expanded-polynomial check, m=4 and16.
Production uses trigonometric expressions; this check uses recurrence coefficients.
"""
from pathlib import Path
import hashlib,json,time
import mpmath as mp
root=Path(__file__).resolve().parents[1];start=time.perf_counter();mp.mp.dps=100
def mul(a,b):
 c=[mp.mpf(0)]*(len(a)+len(b)-1)
 for i,x in enumerate(a):
  for j,y in enumerate(b):c[i+j]+=x*y
 return c
def square(a):return mul(a,a)
def polynomial(m):
 a=[mp.mpf(1)];b=[mp.mpf(1),mp.mpf(-2)]
 for k in range(1,m):
  c=[2*x for x in mul([mp.mpf(1),mp.mpf(-2)],b)]
  for j,x in enumerate(a):c[j]-=x
  a,b=b,c
 f=[-x/(2*m*m) for x in b[1:]]
 assert b[0]==1
 g=[m**4*x for x in square(square(f))];q=square([mp.mpf(1)]*m)
 for j,x in enumerate(q):g[j]+=x
 return g
def val(g,t):return mp.polyval(g[::-1],t)
def linear_integral(g,a,b,endpoint):return sum(c*(endpoint*(b**(i+1)-a**(i+1))/(i+1)-(b**(i+2)-a**(i+2))/(i+2)) for i,c in enumerate(g))
observed=json.loads((root/'results/coordinate_obstruction.json').read_text())['rows'];rows=[]
for m in [4,16]:
 g=polynomial(m)
 knots=sorted(set([mp.mpf(0),mp.mpf('.5'),mp.mpf(1)]+[mp.sin(k*mp.pi/m)**2 for k in range(m//2+1)]))
 def arclength(a,b):
  kk=sorted(set([a,b]+[x for x in knots if a<x<b]))
  return sum(mp.quad(lambda t:mp.sqrt(val(g,t)),[x,y]) for x,y in zip(kk[:-1],kk[1:]))
 A=arclength(mp.mpf(0),mp.mpf('.5'));C=arclength(mp.mpf('.5'),mp.mpf(1))
 B=linear_integral(g,mp.mpf(0),mp.mpf('.5'),mp.mpf('.5'))
 E=linear_integral(g,mp.mpf('.5'),mp.mpf(1),mp.mpf(1))
 r=next(x for x in observed if x['m']==m)
 exact={'A_left':A,'A_right':C,'B_left':B,'B_right':E}
 errors={k:float(abs(mp.mpf(str(r[k]))-v)/v) for k,v in exact.items()}
 assert max(errors.values())<1e-10,(m,errors)
 rows.append({'m':m,'curvature_degree':len(g)-1,'values_60digits':{k:mp.nstr(v,60) for k,v in exact.items()},'relative_errors':errors})
out={'status':'pass','method':'Independent expanded Chebyshev recurrence, exact polynomial antiderivatives for B and 100-digit adaptive integration for sqrt(G). Not interval arithmetic.','mpmath_dps':100,'rows':rows,'seconds':time.perf_counter()-start,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root/'results/coordinate_independent_mp.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

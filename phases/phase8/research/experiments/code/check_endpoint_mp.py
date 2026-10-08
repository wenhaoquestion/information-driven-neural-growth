"""Independent 350-digit analytic integration of the endpoint polynomial."""
from run_experiments import *

t=time.perf_counter(); mp.mp.dps=350
n=128;delta_float=(4*np.log(n)/n)**2;delta=mp.mpf(str(delta_float))
a=[-(1+delta)/(1-delta),2/(1-delta)]

def mul(p,q):
    r=[mp.mpf(0)]*(len(p)+len(q)-1)
    for i,v in enumerate(p):
        for j,w in enumerate(q):r[i+j]+=v*w
    return r

p,q=[mp.mpf(1)],a
for _ in range(1,n):
    r=[2*v for v in mul(a,q)]
    for i,v in enumerate(p):r[i]-=v
    p,q=q,r
poly=mul(q,q)
norm=sum(v/(i+1) for i,v in enumerate(poly))

def H(z):
    return sum(v*z**(i+2)/((i+1)*(i+2)) for i,v in enumerate(poly))/norm

def F(z):return H(z)+H(1-z)+delta**2*(z-mp.mpf('.5'))**2/2

x=np.array([0.,2*delta_float,1-2*delta_float,1.])
b=delta_float**2;weights=np.array([.5-b,b,b,.5-b])
f=EndpointCurvature(n,delta_float)
rows=[]
for k in range(2,5):
    for c in combinations(range(4),k):
        xx=x[list(c)];ww=weights[list(c)]
        mx=[mp.mpf(str(v)) for v in xx];mw=[mp.mpf(str(v)) for v in ww]
        mean=sum(v*w for v,w in zip(mx,mw))/sum(mw)
        exact=sum(w*F(v) for w,v in zip(mw,mx))-sum(mw)*F(mean)
        numeric=f.cell(xx,ww)
        rel=float(abs(mp.mpf(numeric)-exact)/exact)
        rows.append({'cell':c,'analytic_350digit':mp.nstr(exact,50),'float64':numeric,'relative_error':rel})
assert max(r['relative_error'] for r in rows)<1e-9
result={'n':n,'delta':delta_float,'mpmath_dps':350,'method':'independent monomial recurrence and exact antiderivative at 350 decimal digits','rows':rows,'max_relative_error':max(r['relative_error'] for r in rows),'elapsed_seconds':time.perf_counter()-t,'status':'pass','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
save_json('raw/endpoint_mpmath_crosscheck.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

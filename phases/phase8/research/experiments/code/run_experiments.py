"""Reproducible finite verification; no training and no network access."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import argparse, csv, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import scipy
import mpmath as mp
from hierarchy_core import *

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026100708
X = np.array([-1.,0,1,2])
P = np.ones(4)/4


def save_json(path, obj):
    (ROOT/path).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                     default=lambda v: v.item() if isinstance(v, np.generic) else str(v))+'\n')


def save_csv(path, rows):
    with (ROOT/path).open('w', newline='') as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader(); out.writerows(rows)


def high_precision_polynomial(n, eps):
    """Independent 150-digit recurrence, exact monomial integration at low n."""
    mp.mp.dps = 150
    eps = mp.mpf(str(eps))
    a = [1+eps**2/16, mp.mpf(0), -mp.mpf(1)/16]
    def mul(p, q):
        r = [mp.mpf(0)]*(len(p)+len(q)-1)
        for i, v in enumerate(p):
            for j, w in enumerate(q): r[i+j] += v*w
        return r
    p, q = [mp.mpf(1)], a
    for _ in range(1, n):
        r = [2*v for v in mul(a,q)]
        for i, v in enumerate(p): r[i] -= v
        p,q = q,r
    poly = mul(q,q)
    def primitive(t, extra=0):
        return sum(v*t**(i+extra+1)/(i+extra+1) for i,v in enumerate(poly))
    norm = primitive(mp.mpf(4))-primitive(mp.mpf(-4))
    def H(t):
        t = mp.mpf(str(t))
        return (t*(primitive(t)-primitive(mp.mpf(-4)))
                -(primitive(t,1)-primitive(mp.mpf(-4),1)))/norm
    def F(z):
        z=mp.mpf(str(z))
        return H(-z-eps)+H(z-1-eps)+4*eps**2*(z-mp.mpf('.5'))**2
    def cell(x, w):
        xx=[mp.mpf(str(v)) for v in x]; ww=[mp.mpf(str(v)) for v in w]
        mean=sum(a*b for a,b in zip(xx,ww))/sum(ww)
        return sum(a*F(b) for a,b in zip(ww,xx))-sum(ww)*F(mean)
    return cell


def correctness():
    t=time.perf_counter(); details={}
    eps=1/32
    h=lambda x,w: hinge_cell(x,w,[-eps,1+eps],4*eps**2)
    r=evaluate(X,P,h)
    a=eps/4+eps**2/2; s=eps**2/2; b=(1-eps)/4+2*eps**2
    expected=min(a/s,b/(2*a))
    assert abs(r['rho_det']-expected)<1e-11
    c=channel(X,P,EXPLICIT_Q,EXPLICIT_R,h,r['opt2'],r['opt3'])
    assert abs(c['d3']-(eps/8+2*eps**2/3))<1e-14
    assert abs(c['d2']-(1/8+eps/2+26*eps**2/15))<1e-14
    alpha,beta=a/s,b/(2*a)
    expected_lower=1+(alpha-1)*(beta-1)/(alpha+beta-2)
    assert abs(expected_lower-r['stochastic_convex_lower'])<1e-12
    details['reference_hinge']={'enumeration':r,'channel':c,'formula_rho':expected}
    checks=[]
    for n in [4,16,64]:
        f=NeedleCurvature(n,1/16)
        ref=high_precision_polynomial(n,1/16)
        errors=[]
        for k in range(2,5):
            for cc in combinations(range(4),k):
                xx,pp=X[list(cc)],P[list(cc)]
                m=float(ref(xx,pp)); v=f.cell(xx,pp)
                errors.append(abs(v-m)/m)
        assert max(errors)<2e-10,(n,max(errors))
        checks.append({'n':n,'curvature_degree':4*n,'mpmath_dps':150,'max_cell_relative_error':max(errors)})
    details['high_precision_analytic_integrals']=checks
    rng=np.random.default_rng(SEED)
    f=NeedleCurvature(64,1/16); r=evaluate(X,P,f.cell)
    random=[]
    for j in range(30):
        Q=rng.dirichlet(np.ones(3)*(.1 if j%2 else 1),size=4)
        R=rng.dirichlet(np.ones(2)*(.1 if j%2 else 1),size=3)
        c=channel(X,P,Q,R,f.cell,r['opt2'],r['opt3'])
        assert c['upper']+1e-9 >= r['stochastic_convex_lower']
        random.append({'Q':Q.tolist(),'R':R.tolist(),**c})
    details['random_channels']={'seed':SEED,'count':30,'convex_lower':r['stochastic_convex_lower'],'rows':random}
    details['counts']={'coarse':len(P2),'fine':len(P3),'hierarchies':len(HIERARCHIES)}
    details['elapsed_seconds']=time.perf_counter()-t
    details['status']='pass'
    save_json('raw/correctness.json',details)
    print(json.dumps({'correctness':'pass','seconds':details['elapsed_seconds'],'mp_errors':checks}),flush=True)


def weighted():
    t=time.perf_counter(); rows=[]; raw=[]
    for delta in [.2,.1,.03,.01,.003,.001]:
        for power in [1,2,3]:
            b=delta**power
            if b>=.5: continue
            p=np.array([.5-b,b,b,.5-b]); x=np.array([0,delta,1-delta,1.])
            eps=delta/2; h=delta-eps
            for scale in [.0625,.25,1,4]:
                lam=scale*delta**2
                cell=lambda x,w: hinge_cell(x,w,[h,1-h],lam)
                r=evaluate(x,p,cell)
                row={'delta':delta,'epsilon':eps,'inner_weight_power':power,'inner_weight':b,'quadratic_scale':scale,'quadratic_coefficient':lam,'rho_det':r['rho_det'],'rho_times_delta':r['rho_det']*delta,'convex_lower':r['stochastic_convex_lower'],'opt2':r['opt2'],'opt3':r['opt3'],'flat2':str(P2[r['flat2']]),'flat3':str(P3[r['flat3']])}
                rows.append(row); raw.append({**row,**r})
    save_csv('results/weighted_endpoint_hinge.csv',rows)
    save_json('raw/weighted_endpoint_hinge.json',raw)
    print(json.dumps({'weighted_rows':len(rows),'seconds':time.perf_counter()-t,'best':max(rows,key=lambda a:a['rho_det'])}),flush=True)


def polynomial():
    t=time.perf_counter(); rows=[]; raw=[]
    ns=[16,32,64,128,256,512,1024,2048,4096,8192]
    for n in ns:
        asym=16*np.log(n)/n
        epss=sorted(set([1/16]+[float(asym*factor) for factor in [.125,.25,.5,1] if asym*factor<=1/16]))
        for eps in epss:
            f=NeedleCurvature(n,eps); r=evaluate(X,P,f.cell)
            c=channel(X,P,EXPLICIT_Q,EXPLICIT_R,f.cell,r['opt2'],r['opt3'])
            assert c['upper']+1e-8>=r['stochastic_convex_lower']
            h=lambda x,w:hinge_cell(x,w,[-eps,1+eps],4*eps**2)
            ref=evaluate(X,P,h)
            # phi(x)=F(4x-1.5)/(16*M_bound) ensures 0<phi''<=1.
            scale=1/(16*f.max_upper())
            best=r['best_hierarchy']; ii,jj=HIERARCHIES[best]
            row={'n':n,'curvature_degree':4*n,'epsilon':eps,'epsilon_fraction_of_v1':eps/asym,'v1_sequence':abs(eps-asym)<1e-14,'rho_det':r['rho_det'],'stochastic_lower':r['stochastic_convex_lower'],'stochastic_feasible_upper':min(c['upper'],r['rho_det']),'explicit_channel_upper':c['upper'],'hinge_rho_det':ref['rho_det'],'leading_n_over_32logn':n/(32*np.log(n)),'opt2_normalized':r['opt2']*scale,'opt3_normalized':r['opt3']*scale,'hier2_normalized':r['d2'][ii]*scale,'hier3_normalized':r['d3'][jj]*scale,'normalization_curvature_upper':f.max_upper(),'quadrature_order':f.order,'log_normalizer':f.logZ}
            rows.append(row); raw.append({**row,'enumeration':r,'channel':c})
        print(json.dumps({'completed_n':n,'elapsed_seconds':time.perf_counter()-t,'rows':len(rows)}),flush=True)
    # Spacing and positive-weight screens at a single fixed generator.
    f=NeedleCurvature(1024,1/16); screen=[]
    for inner in [.0,.1,.25,.4]:
        for b in [.001,.01,.1,.25,.4,.49]:
            x=np.array([-1,inner,1-inner,2.]); p=np.array([.5-b,b,b,.5-b])
            r=evaluate(x,p,f.cell)
            screen.append({'n':1024,'epsilon':1/16,'inner_shift':inner,'inner_weight':b,'rho_det':r['rho_det'],'stochastic_lower':r['stochastic_convex_lower'],'opt2':r['opt2'],'opt3':r['opt3']})
    save_csv('results/finite_degree.csv',rows); save_json('raw/finite_degree.json',raw)
    save_csv('results/spacing_weight_screen.csv',screen)
    # Increase polynomial-exact quadrature order: independent floating-point nodes.
    cross=[]
    for n in [256,2048,8192]:
        eps=min(1/16,16*np.log(n)/n)
        f=NeedleCurvature(n,eps); g=NeedleCurvature(n,eps,order=3*n+3)
        a=evaluate(X,P,f.cell); b=evaluate(X,P,g.cell)
        err=max(abs(a['cells'][k]-b['cells'][k])/max(a['cells'][k],1e-300) for k in a['cells'])
        assert err<2e-7,(n,err)
        cross.append({'n':n,'epsilon':eps,'order1':f.order,'order2':g.order,'max_cell_relative_error':err,'rho_det_difference':a['rho_det']-b['rho_det']})
    save_json('raw/quadrature_crosscheck.json',cross)
    print(json.dumps({'polynomial_rows':len(rows),'screen_rows':len(screen),'seconds':time.perf_counter()-t,'crosscheck':cross}),flush=True)


def endpoint():
    t=time.perf_counter(); rows=[];raw=[];rng=np.random.default_rng(SEED+1)
    for n in [64,128,256,512,1024,2048,4096,8192]:
        delta=(4*np.log(n)/n)**2
        if delta>=1/16:continue
        x=np.array([0.,2*delta,1-2*delta,1.]); b=delta**2;p=np.array([.5-b,b,b,.5-b])
        f=EndpointCurvature(n,delta);f2=EndpointCurvature(n,delta,2*n+3)
        r=evaluate(x,p,f.cell);r2=evaluate(x,p,f2.cell)
        err=max(abs(r['cells'][k]-r2['cells'][k])/max(r['cells'][k],1e-300) for k in r['cells'])
        assert err<2e-6,(n,err)
        c=channel(x,p,EXPLICIT_Q,EXPLICIT_R,f.cell,r['opt2'],r['opt3'])
        chans=[]
        if n in [128,1024]:
            for j in range(12):
                Q=rng.dirichlet(np.ones(3)*.2,size=4);R=rng.dirichlet(np.ones(2)*.2,size=3)
                cc=channel(x,p,Q,R,f.cell,r['opt2'],r['opt3'])
                assert cc['upper']+1e-8>=r['stochastic_convex_lower']
                chans.append({'Q':Q.tolist(),'R':R.tolist(),**cc})
        row={'n':n,'curvature_degree':2*n,'delta':delta,'inner_weight':b,'rho_det':r['rho_det'],'rho_times_delta':r['rho_det']*delta,'stochastic_lower':r['stochastic_convex_lower'],'stoch_lower_times_delta':r['stochastic_convex_lower']*delta,'explicit_channel_upper':c['upper'],'feasible_upper':min(c['upper'],r['rho_det']),'opt2':r['opt2'],'opt3':r['opt3'],'opt2_normalized':r['opt2']/f.max_upper(),'opt3_normalized':r['opt3']/f.max_upper(),'rho_over_r_logr':r['rho_det']/(2*n/np.log(2*n)),'quadrature_relative_crosscheck':err,'log_normalizer':f.logZ,'flat2':str(P2[r['flat2']]),'flat3':str(P3[r['flat3']])}
        rows.append(row);raw.append({**row,'enumeration':r,'channel':c,'random_channels':chans,'seed':SEED+1})
        print(json.dumps(row),flush=True)
    save_csv('results/endpoint_polynomial.csv',rows);save_json('raw/endpoint_polynomial.json',raw)
    print(json.dumps({'endpoint_rows':len(rows),'seconds':time.perf_counter()-t}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['correctness','weighted','polynomial','endpoint']);args=parser.parse_args()
    meta={'python':sys.version,'executable':sys.executable,'numpy':np.__version__,'scipy':scipy.__version__,'mpmath':mp.__version__,'platform':platform.platform(),'processor':platform.processor(),'seed':SEED,'thread_env':{k:os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']},'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'code').glob('*.py'))},'argv':sys.argv}
    save_json('logs/environment_'+args.stage+'.json',meta)
    globals()[args.stage]()

if __name__=='__main__':main()

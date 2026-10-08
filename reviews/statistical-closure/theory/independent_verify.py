"""New deterministic audit. Reads full original JSON, never imports old code.

No RNG, simulation, training, dependency installation, or old-file writes.
Only this directory receives the audit result. Analytic arguments live in the
companion report; finite checks are not substitutes for the full-class proof.
"""
from fractions import Fraction as F
from itertools import product
from math import isqrt, factorial
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2] / 'studies' / 'additive-risk'
checks = []
reads = {}

def require(ok, name):
    if not ok:
        raise AssertionError(name)
    checks.append(name)

def read(path):
    raw = (BASE/path).read_bytes()
    reads[path] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return json.loads(raw)

def decode(x):
    if isinstance(x, dict):
        if 'rational' in x:
            return F(x['rational'])
        return {k: decode(v) for k, v in x.items()}
    if isinstance(x, list):
        return [decode(v) for v in x]
    return x

def interval(x):
    return (x['lo'], x['hi'])

def irange(x):
    return F(x['lower_rational']), F(x['upper_rational'])

def add(a,b): return a[0]+b[0], a[1]+b[1]
def sub(a,b): return a[0]-b[1], a[1]-b[0]
def scale(a,v): return min(a[0]*v,a[1]*v), max(a[0]*v,a[1]*v)
def divide(a,b):
    xs = [x/y for x in a for y in b]
    return min(xs),max(xs)
def floorgrid(x,d=30): return F((x*10**d).__floor__(),10**d)
def ceilgrid(x,d=30): return -floorgrid(-x,d)
def sqrtfloor(x,scale): return F(isqrt(x.numerator*scale*scale//x.denominator),scale)
def sqrtceil(x,scale=2**64):
    y = sqrtfloor(x,scale)
    return y if y*y == x else y+F(1,scale)
def taylor(x,n): return sum((x**j/factorial(j) for j in range(n+1)),F())
def expminus(x): return ceilgrid(1/taylor(x,200),50)

protocol = read('PROTOCOL.json')
freeze = read('qa/protocol_freeze.json')
require(freeze == {'sha256':reads['PROTOCOL.json']['sha256'],
                   'bytes':reads['PROTOCOL.json']['bytes']},'protocol byte/hash freeze')
pop = read('theory/rational_certificates.json')
local = decode(read('theory/LOCAL_ORACLE_CERTIFICATE.json'))
env = decode(read('statistics/INTERIOR_2048_POWER_ENVELOPE.json'))
power = decode(read('statistics/INTERIOR_2048_POWER_CERTIFICATE.json'))
reference = read('statistics/CONFIRMATION_REFERENCE.json')
public = read('statistics/confirmation/COUNTS_PUBLIC.json')
infer = decode(read('statistics/confirmation/INFERENCE.json'))
runtime = read('statistics/confirmation/RUNTIME.json')
completed = read('statistics/confirmation/COMPLETED.json')
require(power['envelope'] == env,'entire embedded power envelope equals standalone file')
for rec in (runtime, infer, power):
    for p,h in rec['source_sha256'].items():
        raw=(BASE/p).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==h,'source hash '+p)
for p,key in [('theory/local_loss_oracle.py','oracle_sha256'),
              ('statistics/INTERIOR_2048_POWER_ENVELOPE.json','envelope_sha256')]:
    require(hashlib.sha256((BASE/p).read_bytes()).hexdigest()==local[key],key)
require(infer['counts_sha256']==completed['counts_sha256']==reads[
    'statistics/confirmation/COUNTS_PUBLIC.json']['sha256'],'both count hashes')

# Independent positive-series log enclosure, including a tighter 47-term pair.
def log2_bounds(n):
    a=F(1,3); total=F()
    for j in range(n):
        total += 2*a/(2*j+1)
        a /= 9
    return total,total+2*a/F(2*n+1)/(1-F(1,9))
ll,lu=log2_bounds(40)
nl,nu=log2_bounds(47)
require(irange(pop['log2'])==(ll,lu) and ll<nl<nu<lu,'log2 exact and tighter independent enclosure')
for row in pop['interior']:
    n=row['n']; k=n.bit_length()-1
    el,eu=16*k*ll/n,16*k*lu/n
    delta=2/(n**5*k*ll)
    j=(2+32*el*el-delta,2+32*eu*eu)
    a=scale(j,2)
    pairs={'epsilon':(el,eu),'J':j,'A_star_for_F_composed':a,
        'Delta':((el/4-8*delta)/a[1],(eu/4+8*delta)/a[0]),
        'OPT2':((el/2+el*el-4*delta)/a[1],(eu/2+eu*eu+4*delta)/a[0]),
        'OPT3':((el*el/2-4*delta)/a[1],(eu*eu/2+4*delta)/a[0])}
    require(all(irange(row[k])==v for k,v in pairs.items()),'all population fractions interior '+str(n))
    require(F(row['tail_error_bound'])==delta and pairs['Delta'][0]>F(1,1000),
            'interior tail and threshold '+str(n))
for row in pop['endpoint']:
    n=row['n']; k=n.bit_length()-1
    dl,du=(4*k*ll/n)**2,(4*k*lu/n)**2
    cap=(4-8*du*du)*du**3
    require(irange(row['d'])==(dl,du) and irange(row['A'])==(1+dl*dl/2,1+du*du/2)
            and irange(row['Delta'])==(F(),cap) and cap<F(1,1000),
            'all population fractions endpoint '+str(n))

# Build and verify oracle constants from the analytic inequalities.
el,eu=floorgrid(11*ll/128),ceilgrid(11*lu/128)
de=ceilgrid(2/(2048**5*11*ll)); alo=floorgrid(4+64*el*el-2*de)
ahi=ceilgrid(4+64*eu*eu); r=el-F(1,40); eta=F(1,200)
u=(el*el-eta*eta)/16
v0=sqrtfloor(2*u/(1+u/2),10**35)
vr=sqrtfloor((eu*eu-r*r)/8,10**35)+F(1,10**35)
exponent=floorgrid(4096*(v0-vr))
central=ceilgrid(2*(eu-r)/eta*expminus(exponent))
outer=ceilgrid(8/eta*expminus(floorgrid(4096*v0)))
tail=ceilgrid(central+outer); hinge=ceilgrid((eu-r)*central+4*outer)
global_hinge=ceilgrid(eu/2+de)
hw=ceilgrid((2*(eu-el)+16*(eu*eu-el*el)+2*hinge)/alo+
            (2+16*eu*eu)*(1/alo-1/ahi))
sw=ceilgrid(4*(2*tail+16*(eu*eu-el*el))/alo+
            4*(1+16*eu*eu)*(1/alo-1/ahi))
ours=dict(el=el,eu=eu,delta=de,alo=alo,ahi=ahi,r=r,eta=eta,exponent=exponent,
    central=central,outer=outer,tail=tail,hinge_error=hinge,
    global_hinge_error=global_hinge,entropy_width=hw,slope_width=sw)
require(local['constants']==ours,'all 15 oracle rational constants independently reproduced')
require(0<eta<r<el<eu<F(1,16) and v0>vr,'tail separation hypotheses')

def args(q):
    z=4*q-F(3,2)
    return z,((-z-eu,-z-el),(z-1-eu,z-1-el))
def safe(x): return x[1]<=-r or x[0]>=r
def oracle(q,derivative=False):
    z,xs=args(q)
    if derivative:
        cdf=[(F(),tail) if x[1]<=-r else (1-tail,F(1)) if x[0]>=r
             else (F(),F(1)) for x in xs]
        raw=add(sub(cdf[1],cdf[0]),scale((el*el,eu*eu),8*(z-F(1,2))))
        raw=scale(raw,4)
    else:
        raw=scale((el*el,eu*eu),4*(z-F(1,2))**2)
        for x in xs:
            raw=add(raw,(max(0,x[0]),max(0,x[1])+(hinge if safe(x) else global_hinge)))
    return scale(divide(raw,(alo,ahi)),-1)

# Independent partition representation: 4-bit blocks, enumerated from 4^4 maps.
parts=set()
for labels in product(range(4),repeat=4):
    parts.add(tuple(sorted(sum(1<<i for i,v in enumerate(labels) if v==j)
                 for j in set(labels))))
parts=sorted(parts)
p2=[p for p in parts if len(p)<=2]; p3=[p for p in parts if len(p)<=3]
def refines(a,b): return all(any(x&y==x for y in b) for x in a)
hier=[(a,b) for a in p2 for b in p3 if refines(b,a)]
require((len(parts),len(p2),len(p3),len(hier))==(15,8,14,39),'independent 4-bit exhaustive enumeration 15/8/14/39')
def mask(p): return tuple(sorted(sum(1<<i for i in b) for b in p))
def mean(b,x): return sum(x[i] for i in range(4) if b&(1<<i))/b.bit_count()
def cell(p,i): return next(b for b in p if b&(1<<i))
def risks(x):
    return {p:tuple(sum(F(b.bit_count(),4)*oracle(mean(b,x))[side] for b in p)
                    for side in (0,1)) for p in parts}
def slopes(rect):
    return {b:(oracle(mean(b,[a[1] for a in rect]),True)[0],
               oracle(mean(b,[a[0] for a in rect]),True)[1]) for b in range(1,16)}
def slope_cap(a,b,i,s):
    ba,bb=cell(a,i),cell(b,i)
    if ba==bb: return F()
    d=sub(s[ba],s[bb]); return min(F(2),max(abs(d[0]),abs(d[1])))

# Analytic quadratic-root brackets, independently of old bisection algorithm.
# On each side q=t+s*d: A*d^2+B*d+C <= 0, with A=n+2c.
def distance_root(n,t,sign,c=F(1269,250)):
    a=n+2*c; b=-2*c*(sign*(1-2*t)+F(1,3)); cc=-2*c*t*(1-t)
    rad=b*b-4*a*cc
    sl=sqrtfloor(rad,2**160); su=sqrtceil(rad,2**160)
    return (-b+sl)/(2*a),(-b+su)/(2*a)
def check_ci(n,t,ci):
    dl,du=distance_root(n,t,-1); ul,uu=distance_root(n,t,1)
    exact_lo=(max(F(),t-du),max(F(),t-dl))
    exact_hi=(min(F(1),t+ul),min(F(1),t+uu))
    pad=F(1,2**56)
    return ci[0]<=exact_lo[0] and ci[1]>=exact_hi[1] and \
        exact_lo[1]-ci[0]<=pad and ci[1]-exact_hi[0]<=pad
for c,target in [(F(1269,250),160),(F(3689,1000),40),(F(4383,1000),80)]:
    require(taylor(c,30)>target,'strict exponential budget '+str(target))

truth=[F(1,8),F(3,8),F(5,8),F(7,8)]
R=[interval(x) for x in env['rectangle']]
for i,t in enumerate(truth):
    n=env['n_min'][i]; cutoff=250000-sqrtceil(F(500000)*F(3689,1000))
    require(n==cutoff.__ceil__(),'count cutoff '+str(i))
    b=sqrtceil(2*t*(1-t)*F(4383,1000)/n)+2*F(4383,1000)/(3*n)
    require(env['estimation_deviations'][i]==b and interval(env['estimator_ranges'][i])==(t-b,t+b),
            'estimator power envelope '+str(i))
    pad=F(1,2**56)
    # Stored R is the outward bisection at two extremal t values plus one pad.
    lower=R[i][0]+pad; upper=R[i][1]-pad
    dl,du=distance_root(n,t-b,-1); ul,uu=distance_root(n,t+b,1)
    require(lower<=t-b-du and (t-b-dl)-lower<=pad and
            upper>=t+b+uu and upper-(t+b+ul)<=pad,'analytic outer CI root bracket '+str(i))
    vmax=max(x*(1-x) for x in R[i]); off=F(1269,250)/(3*n)
    rad=off+sqrtceil(off*off+2*F(1269,250)*vmax/n)+pad
    require(rad==env['confidence_radii'][i],'moving-center maximum CI radius '+str(i))
require(power['oracle_entropy_width_assumption']==hw and power['oracle_slope_width_assumption']==sw,
        'power uses certified oracle widths')
for item in local['cells']:
    b=sum(1<<i for i in item['block']); a=mean(b,[x[0] for x in R]); z=mean(b,[x[1] for x in R])
    require(item['mean']==[a,z] and item['hprime_interval']==[oracle(z,True)[0],oracle(a,True)[1]],
            'full recorded cell interval '+str(b))
    xa=args(a)[1]; xz=args(z)[1]
    require(all(safe((min(x[0],y[0]),max(x[1],y[1]))) for x,y in zip(xa,xz)),
            'whole mean interval safely outside both needles '+str(b))

rv=risks(truth); ss=slopes(R); lower={}
for a in parts:
    for b in parts:
        if a==b: lower[a,b]=F(); continue
        penalty=2*hw
        for i in range(4):
            if cell(a,i)==cell(b,i): continue
            cap=slope_cap(a,b,i,ss)
            penalty+=(cap*env['estimation_deviations'][i]+
                      min(F(2),cap+2*sw)*env['confidence_radii'][i])/4
        v=sub(rv[a],rv[b])[0]-penalty
        lower[a,b]=max(F(),v) if refines(b,a) else v
plower={(a,b):max(max(lower[a,q] for q in p2),max(lower[b,q] for q in p3)) for a,b in hier}
stored={(mask(a),mask(b)):v for a,b,v in power['hierarchy_lower_bounds']}
require(stored==plower,'all 39 exact moving-center power lower bounds')
require(min(plower.values())==power['inference_lower_bound_on_power_event']>F(1,2000),
        'minimum power-event lower bound strictly exceeds fixed null')
require(power['power_lower_if_certified']==env['power_lower_if_uniform_rejection']==F(4,5),
        'power probability 1 - 1/10 - 1/10')

counts=public['counts']; centers=[F(k,n) for n,k in counts]
rectangle=[interval(x) for x in infer['posterior_simultaneous_95_rectangle']]
require(sum(n for n,k in counts)==10**6 and infer['counts']==counts,'recorded count totals and transfer')
for i,((n,k),ci) in enumerate(zip(counts,rectangle)):
    require(check_ci(n,F(k,n),ci),'actual CI independently bracketed by quadratic roots '+str(i))
rr=risks(centers); ss=slopes(rectangle)
radii=[max(t-lo,hi-t) for t,(lo,hi) in zip(centers,rectangle)]
contrasts={}
for a in parts:
    for b in parts:
        if a==b: contrasts[a,b]=(F(),F()); continue
        radius=sum(slope_cap(a,b,i,ss)*radii[i]/4 for i in range(4))
        j=sub(rr[a],rr[b]); lo,hi=j[0]-radius,j[1]+radius
        if refines(a,b): hi=min(hi,F())
        if refines(b,a): lo=max(lo,F())
        require(lo<=hi,'actual contrast valid ordered endpoints '+str((a,b)))
        contrasts[a,b]=(lo,hi)
levels={(k,a):tuple(max(contrasts[a,b][side] for b in flat) for side in (0,1))
        for k,flat in [(2,p2),(3,p3)] for a in flat}
hb={(a,b):tuple(max(levels[2,a][side],levels[3,b][side]) for side in (0,1)) for a,b in hier}
stored={(mask(a),mask(b)):interval(x) for (a,b),x in infer['hierarchy_95_bounds']}
require(stored==hb,'all 39 exact actual hierarchy CI endpoints')
delta=tuple(min(x[side] for x in hb.values()) for side in (0,1))
require(interval(infer['delta_95_interval'])==delta and infer['reject_null'] and delta[0]>F(1,2000),
        'actual exact interval and rejection independently reconstructed')
for item in infer['all_15_partition_report_error_bounds']:
    part=mask(item['partition']); fitted=item['sample_fitted_reports']
    require(sorted(fitted)==sorted(mean(b,centers) for b in part),'fitted reports '+str(part))
    upper=F()
    for b in part:
        q=mean(b,centers); aa,bb=mean(b,[x[0] for x in rectangle]),mean(b,[x[1] for x in rectangle])
        candidates=[sub(add(oracle(q),scale(oracle(q,True),t-q)),oracle(t))[1] for t in (aa,bb)]
        upper+=F(b.bit_count(),4)*max(F(),min(F(1),max(candidates),2*max(abs(aa-q),abs(bb-q))))
    require(interval(item['population_report_error_interval'])==(F(),upper),
            'recorded report-error bound '+str(part))

result={'status':'PASS','method':'new standalone exact-rational reimplementation; no old module execution',
        'checks_passed':len(checks),'checks':checks,'complete_json_inputs_read':reads,
        'power_lower_bound':str(min(plower.values())), 'actual_delta_interval':[str(v) for v in delta],
        'decimal_display_only':{'power_lower_bound':float(min(plower.values())),
                                'actual_delta_interval':[float(v) for v in delta]},
        'not_performed':['RNG or seed replay','resampling','old-script imports/execution','training',
                         'dependency installation','full high-degree polynomial quadrature',
                         'external timestamp authentication','exhaustion over continuum theta']}
(HERE/'PUBLICATION_THEORY_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','checks_passed','decimal_display_only']},ensure_ascii=False,indent=2))

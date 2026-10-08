"""Small independent exact checks of inference plumbing, without sampling."""
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
import json
import math
import runpy
import signal
import time

signal.alarm(120)
started = time.monotonic()
HERE = Path(__file__).resolve().parent
api = runpy.run_path(str(HERE.parent/'statistics/honest_delta.py'))
I = api['Interval']
bern = api['bernstein_interval']
c = Q(1269,250)

# Exact pointwise Bernstein coverage on a finite audit grid. This is a check,
# not the all-q proof (which is Bernstein plus the stated inversion).
max_fail=Q(0); worst=None; checked=0
for n in range(1, 21):
    cis=[bern(n,Q(k,n)) for k in range(n+1)]
    for j in range(21):
        q=Q(j,20)
        fail=sum((Q(math.comb(n,k))*q**k*(1-q)**(n-k)
                  for k,ci in enumerate(cis) if not ci.lo<=q<=ci.hi),Q(0))
        assert fail <= Q(1,80)
        checked+=1
        if fail>max_fail:
            max_fail,worst=fail,(n,str(q))
    assert all(cis[k].lo<=cis[k+1].lo and cis[k].hi<=cis[k+1].hi
               for k in range(n))


class SquareLoss:
    variation=Q(2)
    width=Q(0)
    def entropy(self,x):
        value=x*(1-x)
        return I(value-self.width,value) if x.numerator%2 else I(value,value+self.width)
    def slope(self,x):
        value=1-2*x
        return I(value-self.width,value) if x.numerator%2 else I(value,value+self.width)


def risk(part,theta,p):
    return sum((sum(p[i] for i in b) *
                (sum(p[i]*theta[i] for i in b)/sum(p[i] for i in b)) *
                (1-sum(p[i]*theta[i] for i in b)/sum(p[i] for i in b))
                for b in part),Q(0))


loss=SquareLoss(); p=(Q(1,10),Q(2,10),Q(3,10),Q(4,10))
rect=(I(Q(1,10),Q(1,5)),I(Q(3,10),Q(9,20)),
      I(Q(11,20),Q(7,10)),I(Q(4,5),Q(9,10)))
enclosed=api['delta_rectangle'](rect,p,loss)
grid=list(product(*[(r.lo,(r.lo+r.hi)/2,r.hi) for r in rect]))
for theta in grid:
    rr={part:risk(part,theta,p) for part in api['PARTITIONS']}
    for (a,b),bound in enclosed['contrast_bounds'].items():
        assert bound.lo<=rr[a]-rr[b]<=bound.hi
    o2=min(rr[a] for a in api['P2']); o3=min(rr[b] for b in api['P3'])
    delta=min(max(rr[a]-o2,rr[b]-o3) for a,b in api['HIERARCHIES'])
    assert enclosed['delta'].lo<=delta<=enclosed['delta'].hi

# Stress the distinction between true Delta and output of the actual moving
# center implementation. Oracle enclosures vary asymmetrically with x.
loss.width=Q(1,10**8)
theta=(Q(1,10),Q(2,5),Q(3,5),Q(9,10));p=(Q(1,4),)*4
power=api['certify_power'](theta,p,loss,entropy_width=loss.width,slope_width=loss.width)
assert power['certified']
env=power['envelope']; ns=(250000,)*4
ks=[]
for n,r in zip(ns,env['estimator_ranges']):
    lower=(r.lo*n).numerator//(r.lo*n).denominator+1
    upper=(r.hi*n).numerator//(r.hi*n).denominator
    ks.append((lower,upper))
moving_lower=[]
for vals in product(*ks):
    counts=list(zip(ns,vals))
    got=api['infer_delta'](counts,p,loss)
    assert got['reject_null']
    assert got['delta'].lo>=power['inference_lower_bound_on_power_event']
    for actual,outer in zip(got['rectangle'],env['rectangle']):
        assert outer.lo<=actual.lo<=actual.hi<=outer.hi
    moving_lower.append(got['delta'].lo)

report={
 'binomial_grid_checks':checked,
 'binomial_max_noncoverage_exact':str(max_fail),'binomial_worst_grid_case':worst,
 'contrast_grid_points':len(grid),'contrasts_checked_per_point':225,
 'moving_center_extremal_integer_count_cases':len(moving_lower),
 'artificial_oracle_uniform_width':str(loss.width),
 'synthetic_power_certificate_lower':str(power['inference_lower_bound_on_power_event']),
 'minimum_checked_inference_lower':str(min(moving_lower)),
 'scope':'Finite exact implementation checks; not candidate power certification; no labels sampled.',
 'all_assertions_passed':True,'seconds':time.monotonic()-started}
(HERE/'statistics_exact_review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

"""Read-only checks of pilot deliverables; writes only this review directory."""
from fractions import Fraction as Q
from pathlib import Path
from itertools import product
import hashlib
import json
import math
import signal

signal.alarm(120)
HERE = Path(__file__).resolve().parent
PILOT = HERE.parent


def read(rel):
    return json.loads((PILOT / rel).read_text())


def interval(row):
    return Q(row['lower_rational']), Q(row['upper_rational'])


def contains(outer, inner):
    return outer[0] <= inner[0] <= inner[1] <= outer[1]


# Independent 50-term recursive rational sum, distinct from certificate's 40.
power, total = Q(1, 3), Q(0)
for j in range(50):
    total += 2 * power / (2*j+1)
    power /= 9
log2 = (total, total + 2*power/Q(101)/Q(8,9))
cert = read('theory/rational_certificates.json')
assert contains(interval(cert['log2']), log2)
assert Q(3) > Q(3,2)**2
checks = []
for row in cert['interior']:
    n = row['n']
    k = n.bit_length() - 1
    assert n == 2**k
    lo, hi = k*log2[0], k*log2[1]
    e0, e1 = 16*lo/n, 16*hi/n
    d = 2/(n**5*lo)
    bounds = ((e0/4-8*d)/(4+64*e1**2),
              (e1/4+8*d)/(4+64*e0**2-2*d))
    assert 0 < e0 <= e1 <= Q(1,16)
    assert contains(interval(row['epsilon']), (e0,e1))
    assert contains(interval(row['Delta']), bounds)
    assert bounds[0] > Q(1,1000)
    checks.append({'candidate': f'interior_v1_n{n}',
                   'independent_tighter_interval_contained': True,
                   'threshold_passes_exactly': True})

for row in cert['endpoint']:
    n = row['n']; k = n.bit_length()-1
    d0, d1 = ((4*k*t/n)**2 for t in log2)
    upper = (4-8*d1**2)*d1**3
    assert 0 < d0 <= d1 < Q(1,4)
    assert contains(interval(row['d']), (d0,d1))
    assert contains(interval(row['Delta']), (Q(0),upper))
    assert upper < Q(1,1000)
    checks.append({'candidate': f'endpoint_n{n}',
                   'independent_tighter_interval_contained': True,
                   'threshold_fails_exactly': True})


def all_parts():
    result = []
    # Set-of-frozensets removes block labeling, independent brute-force map method.
    seen = set()
    for lab in product(range(4), repeat=4):
        part = frozenset(frozenset(i for i in range(4) if lab[i] == k)
                         for k in set(lab))
        if part not in seen:
            seen.add(part); result.append(part)
    return result


parts = all_parts()
p2 = [p for p in parts if len(p)<=2]
p3 = [p for p in parts if len(p)<=3]
pairset = {(c,f) for c,f in product(p2,p3)
           if all(any(b<=a for a in c) for b in f)}
exactset = {(c,f) for c,f in pairset if len(c)==2 and len(f)==3}
assert (len(parts),len(pairset),len(exactset)) == (15,39,18)

first = read('computation/first_pass.json')
frozen = read('computation/first_pass_freeze.json')
assert hashlib.sha256((PILOT/'computation/first_pass.json').read_bytes()).hexdigest() == frozen['sha256']
enum_checks = []
for row in first['candidates']:
    ps = {r['partition_id']: frozenset(frozenset(b) for b in r['cells'])
          for r in row['partitions']}
    assert set(ps.values()) == set(parts)
    hs = {(ps[h['coarse_id']],ps[h['fine_id']]) for h in row['hierarchies']}
    assert hs == pairset
    costs = {r['partition_id']:r['cost_normalized'] for r in row['partitions']}
    o2 = min(v for i,v in costs.items() if len(ps[i])<=2)
    o3 = min(v for i,v in costs.items() if len(ps[i])<=3)
    ds = [(max(costs[h['coarse_id']]-o2,costs[h['fine_id']]-o3),
           (ps[h['coarse_id']],ps[h['fine_id']])) for h in row['hierarchies']]
    delta = min(v for v,pair in ds)
    exact = min(v for v,pair in ds if pair in exactset)
    assert delta == exact == row['summary']['delta']
    assert o2 == row['summary']['opt2'] and o3 == row['summary']['opt3']
    case = {'candidate':row['parameters']['candidate_id'], 'all15_and39_match':True,
            'exact18_same_additive_minimum':True}
    if row['parameters']['family']=='interior_v1':
        saved = next(r for r in cert['interior'] if r['n']==row['parameters']['n'])
        lo,hi=interval(saved['Delta']); fdelta=Q.from_float(delta)
        case['floating_delta_inside_exact_certificate']=lo<=fdelta<=hi
        case['float_minus_certificate_midpoint']=float(fdelta-(lo+hi)/2)
    enum_checks.append(case)

files = ['PROTOCOL.json','theory/FIRST_PASS_THEORY.md','theory/certify_bounds.py',
         'theory/rational_certificates.json','computation/compute_pilot.py',
         'computation/first_pass.json','computation/first_pass_freeze.json']
report={'exact_interval_checks':checks,'enumeration_checks':enum_checks,
        'first_pass_freeze_hash_verified':True,
        'inputs_sha256':{p:hashlib.sha256((PILOT/p).read_bytes()).hexdigest() for p in files},
        'numerical_results_are_not_certificates':True,'all_assertions_passed':True}
(HERE/'secondary_review_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

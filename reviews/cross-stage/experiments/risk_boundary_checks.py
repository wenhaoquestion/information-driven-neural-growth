"""New bounded adversarial checks of reporting, delayed schedules and normalization.

Exact rational arithmetic; no training, no imports from the historical science.
The universal claims are audited algebraically in risk_evidence_first_pass.md.
"""
from fractions import Fraction as F
from itertools import product, combinations
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
def partitions(n):
    out = []
    for a in product(range(n), repeat=n):
        if a[0] or any(a[i] > 1 + max(a[:i]) for i in range(1, n)):
            continue
        out.append(tuple(tuple(i for i in range(n) if a[i] == k) for k in range(max(a) + 1)))
    return out
def refines(fine, coarse):
    return all(any(set(c) <= set(d) for d in coarse) for c in fine)
def phi(x):
    return x*x/2 + x**4/12
def derivative(x):
    return x + x**3/3
def B(x, q):
    return phi(x) - phi(q) - derivative(q)*(x-q)
def distortion(P, p, x):
    return sum(sum(p[i]*phi(x[i]) for i in c)
               - sum(p[i] for i in c)*phi(sum(p[i]*x[i] for i in c)/sum(p[i] for i in c)) for c in P)

n = 4
ps = partitions(n)
schedules = [[((0, 1, 2, 3),)]]
for k in range(2, n+1):
    schedules = [h+[p] for h in schedules for p in ps if len(p) <= k and refines(p, h[-1])]
weights = [F(1,10), F(2,10), F(3,10), F(4,10)]
sources = [list(map(F, [0, 0, 1, 1])), [F(1,2)]*4,
           [F(1,8), F(3,8), F(5,8), F(7,8)]]
backwards_checked = 0
for h in schedules:
    q = [None]*n
    q[-1] = tuple((i,) for i in range(n))
    for k in range(n-2, -1, -1):
        blocks = list(q[k+1])
        i, j = next((i, j) for i, j in combinations(range(len(blocks)), 2)
                    if any(set(blocks[i]+blocks[j]) <= set(c) for c in h[k]))
        q[k] = tuple([c for a, c in enumerate(blocks) if a not in (i, j)] + [tuple(sorted(blocks[i]+blocks[j]))])
        assert len(q[k]) == k+1 and refines(q[k+1], q[k]) and refines(q[k], h[k])
    for x in sources:
        for original, refined in zip(h, q):
            assert distortion(refined, weights, x) <= distortion(original, weights, x)
    backwards_checked += 1

# The reporting error term is positive even when a partition itself is lossless.
x = [F(1,4), F(3,4)]
p = [F(1,2), F(1,2)]
q = [F(1,2), F(1,2)]
report_excess = sum(a*B(b, c) for a, b, c in zip(p, x, q))
assert report_excess > 0 and distortion(((0,), (1,)), p, x) == 0
# Verify the complete decomposition for every partition, including noncontiguous
# ones, at deliberately non-oracle reports.
report_checks = 0
for P in ps:
    q = [F(k+1, len(P)+2) for k in range(len(P))]
    x = sources[-1]
    direct = sum(weights[i]*B(x[i], v) for c, v in zip(P, q) for i in c)
    report = sum(sum(weights[i] for i in c)*B(sum(weights[i]*x[i] for i in c)/sum(weights[i] for i in c), v)
                 for c, v in zip(P, q))
    assert direct == distortion(P, weights, x) + report
    report_checks += 1

# Off-support/global function preservation would be false.
negative_input = (F(1), F(-1))
parent = F(1,2)*max(0, sum(negative_input))
children = F(1,2)*sum(max(0, z) for z in negative_input)
assert (parent, children) == (0, F(1,2))
# Unrestricted one-unit and copy-only squared-risk examples.
posterior = [F(1,4), F(3,4)]
assert [max(0, z) for z in posterior] == posterior
copied_squared_excess = sum(F(1,2)*(z-F(1,2))**2 for z in posterior)
assert copied_squared_excess == F(1,16)
# Population gradient of 2(aw²-1)² at (0,1), one simultaneous step 3/4.
a, w, eta = F(0), F(1), F(3,4)
da, dw = 4*w*w*(a*w*w-1), 8*a*w*(a*w*w-1)
a, w = a-eta*da, w-eta*dw
assert (a, w, 2*(a*w*w-1)**2) == (3, 1, 8)

# Asymmetric full-report affine normalization with inverse interval [-3,7].
s, t, M = F(10), F(-3), F(50)
phit = lambda y: phi(s*y+t)/(s*s*M)
dphit = lambda y: derivative(s*y+t)/(s*M)
Bt = lambda y, q: phit(y)-phit(q)-dphit(q)*(y-q)
for y in [F(i,20) for i in range(21)]:
    for q in [F(i,20) for i in range(21)]:
        assert Bt(y,q) == B(s*y+t,s*q+t)/(s*s*M)
        l1, l0 = Bt(F(1),q), Bt(F(0),q)
        assert 0 <= l1 <= F(1,2) and 0 <= l0 <= F(1,2)
        conditional_regret = y*l1+(1-y)*l0 - (y*Bt(F(1),y)+(1-y)*Bt(F(0),y))
        assert conditional_regret == Bt(y,q)

out = {
    'status':'pass', 'arithmetic':'fractions.Fraction',
    'scope':'New bounded adversarial checks, not rerun of archived experiments or proof by enumeration',
    'at_most_four_capacity_schedules':backwards_checked,
    'sources_per_schedule':len(sources),
    'non_oracle_partition_decompositions':report_checks,
    'lossless_partition_with_nonzero_reporting_excess':str(report_excess),
    'negative_input_copy_birth_outputs':[str(parent),str(children)],
    'copied_squared_excess':str(copied_squared_excess),
    'gradient_after_parameters':['3','1'], 'gradient_after_excess':'8',
    'affine_full_report_grid_pairs':441,
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(HERE/'risk_boundary_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

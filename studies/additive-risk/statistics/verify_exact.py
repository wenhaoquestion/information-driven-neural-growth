"""Deterministic finite checks.  No random generator and no simulated labels."""
import itertools
import json
from fractions import Fraction as Q
from math import comb
from pathlib import Path
from honest_delta import (
    Interval, PARTITIONS, P2, P3, HIERARCHIES, bernstein_interval,
    exp_strictly_above, sqrt_upper, fitted_risk, delta_rectangle,
    PositiveCoefficientPolynomialLoss, report_error_bounds,
    certify_power, infer_delta,
)


def exact_delta(theta, p, loss):
    risk = {part: fitted_risk(part, theta, p, loss).lo for part in PARTITIONS}
    opt2, opt3 = min(risk[x] for x in P2), min(risk[x] for x in P3)
    return min(max(risk[a]-opt2, risk[b]-opt3) for a, b in HIERARCHIES)


def run():
    checks = {}
    assert (len(PARTITIONS), len(P2), len(P3), len(HIERARCHIES)) == (15,8,14,39)
    checks['enumeration'] = {'partitions':15, 'at_most_2':8, 'at_most_3':14,
                             'compatible_pairs':39}
    for c, target in [(Q(1269,250),160),(Q(3689,1000),40),(Q(4383,1000),80)]:
        assert exp_strictly_above(c,Q(target))
    for x in [Q(0),Q(1),Q(2),Q(1,3),Q(10**20,7)]:
        y = sqrt_upper(x)
        assert y*y >= x and (y == 0 or (y-Q(1,1<<64))**2 < x)
    checks['rational_probability_constants'] = 'passed'

    cases, worst = 0, Q(0)
    for n in (1,2,4,8,16,32):
        ci = [bernstein_interval(n,Q(k,n)) for k in range(n+1)]
        for j in range(17):
            theta = Q(j,16)
            failure = sum(Q(comb(n,k))*theta**k*(1-theta)**(n-k)
                          for k,a in enumerate(ci) if not a.lo <= theta <= a.hi)
            assert failure <= Q(1,80)
            worst = max(worst,failure)
            cases += 1
    checks['exact_binomial_coverage_spot_checks'] = {
        'cases':cases, 'worst_failure_fraction':str(worst),
        'worst_failure_float':float(worst), 'per_state_budget':0.0125,
        'scope':'finite deterministic checks; all-theta proof is analytic'}

    p = (Q(1,10),Q(2,10),Q(3,10),Q(4,10))
    rectangle = tuple(Interval(Q(a,10),Q(a+1,10)) for a in (1,3,5,8))
    grid = tuple(tuple(a.lo+(a.hi-a.lo)*Q(j,2) for j in range(3))
                 for a in rectangle)
    posterior_cases, report_cases = 0, 0
    for raw in ((1,), (1,2,3)):
        loss = PositiveCoefficientPolynomialLoss(raw)
        cert = delta_rectangle(rectangle,p,loss)['delta']
        center = tuple((a.lo+a.hi)/2 for a in rectangle)
        report_bounds = {}
        reports = {}
        for part in PARTITIONS:
            reports[part] = tuple(sum(p[i]*center[i] for i in b)/sum(p[i] for i in b)
                                  for b in part)
            report_bounds[part] = report_error_bounds(part,reports[part],rectangle,p,loss)
        for theta in itertools.product(*grid):
            delta = exact_delta(theta,p,loss)
            assert cert.lo <= delta <= cert.hi
            point = delta_rectangle(tuple(Interval.point(x) for x in theta),p,loss)['delta']
            assert point.lo == point.hi == delta
            posterior_cases += 1
            for part in PARTITIONS:
                regret = Q(0)
                for b,q in zip(part,reports[part]):
                    w = sum(p[i] for i in b)
                    t = sum(p[i]*theta[i] for i in b)/w
                    regret += w*(loss.entropy(q).lo+(t-q)*loss.slope(q).lo-
                                 loss.entropy(t).lo)
                assert 0 <= regret <= report_bounds[part].hi
                report_cases += 1
    checks['rectangle_contrast_checks'] = posterior_cases
    checks['unknown_report_error_checks'] = report_cases

    # Hypothetical count vectors form a finite exact verification grid.  They
    # are not samples, and do not use or evaluate any pilot candidate loss.
    p = (Q(1,4),)*4
    theta = (Q(1,10),Q(3,10),Q(7,10),Q(9,10))
    loss = PositiveCoefficientPolynomialLoss((1,2,3))
    cert = certify_power(theta,p,loss,m=40000)
    ranges = cert['envelope']['estimator_ranges']
    choices = []
    for a in ranges:
        low = -((-10000*a.lo.numerator)//a.lo.denominator)
        high = (10000*a.hi.numerator)//a.hi.denominator
        choices.append((low,high))
    power_cases = 0
    for ks in itertools.product(*choices):
        out = infer_delta(tuple((10000,k) for k in ks),p,loss)
        assert out['delta'].lo >= cert['inference_lower_bound_on_power_event']
        power_cases += 1
    checks['moving_center_power_checks'] = power_cases
    checks['status'] = 'all deterministic exact checks passed'
    return checks


if __name__ == '__main__':
    result = run()
    destination = Path(__file__).with_name('EXACT_VERIFICATION.json')
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

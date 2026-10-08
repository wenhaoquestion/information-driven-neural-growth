"""New C02 first-pass diagnostics. Exact arithmetic; no historical reruns.

Targets: all-partition/at-most geometry, collapsing gaps, fixed-weight
comparison with recomputed means, and large-price nonpolynomial gadgets.
These finite checks cannot establish the polynomial asymptotic theorem.
"""
from fractions import Fraction as Q
from itertools import product
import hashlib
import json
from pathlib import Path


def partitions(n):
    def rec(i, blocks):
        if i == n:
            yield tuple(tuple(b) for b in blocks)
            return
        for j in range(len(blocks)):
            blocks[j].append(i)
            yield from rec(i + 1, blocks)
            blocks[j].pop()
        blocks.append([i])
        yield from rec(i + 1, blocks)
        blocks.pop()
    return list(rec(0, []))


PARTITIONS = partitions(4)
COARSE = [p for p in PARTITIONS if len(p) <= 2]
FINE = [p for p in PARTITIONS if len(p) <= 3]
HIERARCHIES = [(f, c) for f in FINE for c in COARSE
               if all(any(set(b) <= set(a) for a in c) for b in f)]


def multiply(a, b):
    out = [Q(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return out


def power(a, n):
    out = [Q(1)]
    for _ in range(n):
        out = multiply(out, a)
    return out


def generator(g):
    coef = [Q(0), Q(0)] + [c / ((i+1)*(i+2)) for i, c in enumerate(g)]
    def phi(x):
        value = Q(0)
        for c in reversed(coef):
            value = value*x+c
        return value
    return phi


def costs(phi, xs, ps):
    def cell(indices):
        mass = sum(ps[i] for i in indices)
        mean = sum(ps[i]*xs[i] for i in indices) / mass
        return sum(ps[i]*phi(xs[i]) for i in indices) - mass*phi(mean)
    return cell, {p: sum(cell(c) for c in p) for p in PARTITIONS}


def price(allcost):
    opt2 = min(allcost[p] for p in COARSE)
    opt3 = min(allcost[p] for p in FINE)
    rho = min(max(allcost[f]/opt3, allcost[c]/opt2) for f, c in HIERARCHIES)
    return rho, opt2, opt3


def channel_cost(phi, xs, ps, channel):
    total = sum(p*phi(x) for p, x in zip(ps, xs))
    for z in range(len(channel[0])):
        mass = sum(ps[i]*channel[i][z] for i in range(4))
        if mass:
            mean = sum(ps[i]*channel[i][z]*xs[i] for i in range(4)) / mass
            total -= mass*phi(mean)
    return total


EQ = (Q(1,4),)*4
WEIGHTS = ((Q(1,2), Q(1,6), Q(1,6), Q(1,6)),
           (Q(1,10000), Q(1,100), Q(1,10), Q(8899,10000)))
CHANNEL = ((Q(1), Q(0), Q(0)), (Q(1,3), Q(2,3), Q(0)),
           (Q(0), Q(3,5), Q(2,5)), (Q(0), Q(0), Q(1)))


def check(phi, xs, do_weights):
    cell, allcost = costs(phi, xs, EQ)
    rho, opt2, opt3 = price(allcost)
    a, q, d = [cell((i, i+1)) for i in range(3)]
    t, u = cell((0,1,2)), cell((1,2,3))
    assert opt3 == min(a,q,d)
    assert opt2 == min(t,u,a+d)
    assert t >= a+q and u >= q+d
    if rho > 1:
        assert opt3 == q and opt2 == a+d
        assert rho <= min(a,d)/q
        assert rho <= min(t,u)/(a+d)
        assert q <= t/rho**2
        assert a+q <= 2*t/rho
    if do_weights:
        eqch = channel_cost(phi, xs, EQ, CHANNEL)
        for ps in WEIGHTS:
            _, pcost = costs(phi, xs, ps)
            lo, hi = 4*min(ps), 4*max(ps)
            assert all(lo*allcost[p] <= pcost[p] <= hi*allcost[p]
                       for p in PARTITIONS)
            prho, _, _ = price(pcost)
            assert lo/hi*rho <= prho <= hi/lo*rho
            ch = channel_cost(phi, xs, ps, CHANNEL)
            assert lo*eqch <= ch <= hi*eqch
    return rho


quartets = []
for k in (1, 3, 6, 12):
    e = Q(1,10**k)
    quartets += [(Q(0),e,2*e,Q(1)), (Q(0),e,1-e,Q(1)),
                 (Q(0),Q(1,2),Q(1,2)+e,Q(1)),
                 (Q(0),1-2*e,1-e,Q(1))]
polynomials = [[Q(1)]]
for root, m in product((Q(0),Q(1,1000000),Q(1,2),Q(1)), (1,2,4)):
    g = power([-root,Q(1)],2*m)
    polynomials.append(g)
    g = list(g)
    g[0] += Q(1,10**12)
    polynomials.append(g)
polynomials += [multiply(power([Q(0),Q(1)],m), power([Q(1),Q(-1)],m))
                for m in (1,2,4)]
max_polynomial = Q(1)
for xs, g in product(quartets, polynomials):
    max_polynomial = max(max_polynomial, check(generator(g), xs, True))

gadget_results = []
for k in (2,4,8,12,16):
    e = Q(1,10**k)
    def phi(x):
        return max(-x-e,Q(0)) + max(x-1-e,Q(0)) + 4*e*e*(x-Q(1,2))**2
    rho = check(phi, (Q(-1),Q(0),Q(1),Q(2)), False)
    gadget_results.append({"epsilon": str(e), "price": str(rho),
                           "2_epsilon_price": str(2*e*rho)})

result = {"status": "all exact assertions passed",
          "partitions": len(PARTITIONS), "at_most_hierarchies": len(HIERARCHIES),
          "polynomial_cases": len(quartets)*len(polynomials),
          "curvature_degree_max": max(len(g)-1 for g in polynomials),
          "posterior_gap_min": "1/10^12",
          "max_polynomial_price": str(max_polynomial),
          "weight_vectors_checked_per_polynomial_case": 2,
          "fixed_channel_rows": [[str(x) for x in row] for row in CHANNEL],
          "large_price_reference_gadgets": gadget_results,
          "limitations": "Finite diagnostic, not an asymptotic proof or stochastic optimization.",
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
path = Path(__file__).with_name("adversarial_diagnostics_results.json")
path.write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))

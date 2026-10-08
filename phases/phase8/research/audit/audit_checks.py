#!/usr/bin/env python3
"""Independent, standard-library-only algebra and geometry checks for v1.

Finite checks are supporting evidence, not a proof of the asymptotic theorem.
Run from any directory; output is written alongside this script.
"""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import math


def partitions(n):
    def visit(i, cells):
        if i == n:
            yield tuple(tuple(c) for c in cells)
            return
        for j in range(len(cells)):
            cells[j].append(i)
            yield from visit(i + 1, cells)
            cells[j].pop()
        cells.append([i])
        yield from visit(i + 1, cells)
        cells.pop()
    return list(visit(0, []))


PARTS = partitions(4)
COARSE = [p for p in PARTS if len(p) == 2]
FINE = [p for p in PARTS if len(p) == 3]
NESTED = [(c, f) for c in COARSE for f in FINE
          if all(any(set(a) <= set(b) for b in c) for a in f)]
assert (len(COARSE), len(FINE), len(NESTED)) == (7, 6, 18)


def cell_cost(xs, weights, cell, phi):
    mass = sum(weights[i] for i in cell)
    mean = sum(weights[i] * xs[i] for i in cell) / mass
    return sum(weights[i] * phi(xs[i]) for i in cell) - mass * phi(mean)


def partition_cost(xs, weights, part, phi):
    return sum(cell_cost(xs, weights, cell, phi) for cell in part)


def evaluate(xs, weights, phi):
    costs = {p: partition_cost(xs, weights, p, phi) for p in PARTS}
    o2 = min(costs[p] for p in COARSE)
    o3 = min(costs[p] for p in FINE)
    price = min(max(costs[c] / o2, costs[f] / o3) for c, f in NESTED)
    return costs, o2, o3, price


def hinge_checks():
    out = []
    xs = list(map(F, [-1, 0, 1, 2]))
    weights = [F(1, 4)] * 4
    for eps in [F(1, 16), F(1, 20), F(1, 100), F(1, 10000)]:
        def phi(x):
            return max(-x-eps, 0) + max(x-1-eps, 0) + 4*eps**2*(x-F(1, 2))**2
        a = eps/4 + eps**2/2
        s = eps**2/2
        b = t = (1-eps)/4 + 2*eps**2
        q = (1-eps)/2 + F(9, 2)*eps**2
        u = (1-eps)/2 + F(14, 3)*eps**2
        for cell, expected in [((0,1),a), ((2,3),a), ((0,2),b), ((1,3),b),
                               ((1,2),s), ((0,3),q), ((0,1,2),t), ((1,2,3),t),
                               ((0,1,3),u), ((0,2,3),u)]:
            assert cell_cost(xs, weights, cell, phi) == expected
        costs, o2, o3, price = evaluate(xs, weights, phi)
        alpha, beta = a/s, t/(2*a)
        assert (o2, o3, price) == (2*a, s, min(alpha, beta))
        # Every hierarchy is dominated by one of the two claimed Pareto vectors.
        for c, f in NESTED:
            v = (costs[c]/o2, costs[f]/o3)
            assert (v[0] >= 1 and v[1] >= alpha) or (v[0] >= beta and v[1] >= 1)
        # Fractional cells of the explicit stochastic chain, original weights.
        fine_rows = [[F(1,4),0,0,0], [0,F(1,4),F(1,8),0], [0,0,F(1,8),F(1,4)]]
        coarse_rows = [[F(1,4),F(1,4),F(1,8),0], fine_rows[2]]
        def row_cost(row):
            return cell_cost(xs, row, [i for i,w in enumerate(row) if w], phi)
        fine_cost = sum(map(row_cost, fine_rows))
        coarse_cost = sum(map(row_cost, coarse_rows))
        assert fine_cost == eps/8 + F(2,3)*eps**2
        assert coarse_cost == F(1,8) + eps/2 + F(26,15)*eps**2
        lb = 1+(alpha-1)*(beta-1)/(alpha+beta-2)
        lam = (beta-1)/(alpha+beta-2)
        assert lam+(1-lam)*beta == lam*alpha+1-lam == lb
        out.append({'epsilon':str(eps), 'deterministic_price':str(price),
                    'stochastic_convexification_lower':str(lb),
                    'explicit_channel_price':str(max(fine_cost/o3,coarse_cost/o2))})
    return out


def multiply(p, q):
    out = [F(0)]*(len(p)+len(q)-1)
    for i,a in enumerate(p):
        for j,b in enumerate(q): out[i+j] += a*b
    return out


def primitive_twice(g):
    coeff = [F(0),F(0)] + [c/F((j+1)*(j+2)) for j,c in enumerate(g)]
    def phi(x):
        total = F(0)
        for c in reversed(coeff): total = total*x+c
        return total
    return phi


def polynomial_checks():
    # All curvatures are manifestly nonnegative; some have exact interior zeros.
    gs = [[F(1)], [F(0),F(1)], [F(1),F(-1)]]
    for center in [F(0),F(1,3),F(1,2),F(1)]:
        base, poly = [-center,F(1)], [F(1)]
        for power in range(1,9):
            poly = multiply(poly,base)
            if power % 2 == 0: gs.append(poly[:])
    quartets = [[F(0),F(2,5),F(3,5),F(1)],
                [F(0),F(1,10**12),F(1,2),F(1)],
                [F(0),F(1,2),1-F(1,10**12),F(1)],
                [F(0),F(1,2)-F(1,10**12),F(1,2)+F(1,10**12),F(1)],
                [F(0),F(1,3),F(2,3),F(1)]]
    weights = [F(1,4)]*4
    conflicts = 0
    max_price = F(1)
    for g in gs:
        phi = primitive_twice(g)
        for xs in quartets:
            costs,o2,o3,R = evaluate(xs,weights,phi)
            pairs = [cell_cost(xs,weights,(i,i+1),phi) for i in range(3)]
            tleft = cell_cost(xs,weights,(0,1,2),phi)
            tright = cell_cost(xs,weights,(1,2,3),phi)
            assert o3 == min(pairs)
            assert o2 == min(tleft,pairs[0]+pairs[2],tright)
            assert tleft >= pairs[0]+pairs[1]
            assert tright >= pairs[1]+pairs[2]
            max_price = max(max_price,R)
            if R > 1:
                conflicts += 1
                p,q,s = pairs
                Q,T = p+s,tleft
                assert q == o3 and Q == o2
                assert R <= min(p,s)/q
                assert R <= min(tleft,tright)/Q
                assert q <= T/R**2
                assert p+q <= 2*T/R
    return {'instances':len(gs)*len(quartets), 'conflicts':conflicts,
            'max_exact_price':str(max_price),'curvatures':len(gs),'quartets':len(quartets)}


def poisson_geometry_checks():
    bs = [0.0,-0.9,0.9]
    bs += [sign*(1-10**(-k)) for sign in [-1,1] for k in [2,4,8,12,15]]
    eps_values = [1/64,1/100,1e-4,1e-8]
    min_alpha = 1.0
    min_left_ratio = math.inf
    min_right_ratio = math.inf
    count = 0
    for b in bs:
        wb = 1-abs(b)
        s2 = (1-b)*(1+b)
        s = math.sqrt(s2)
        theta_b = math.atan2(s,b)
        for eps in eps_values:
            h=eps*s
            assert theta_b > 5*h and theta_b-4*h < math.pi
            # Stable identity cos(theta_b-v)-cos(theta_b).
            for k in [4,4.5,5]:
                v=k*h
                displacement=2*math.sin(v/2)*(s*math.cos(v/2)-b*math.sin(v/2))
                left=displacement/(eps*wb)
                right=((1-b)-displacement)/(eps*wb)
                assert left >= 3*(1-1e-12) and left <= 12*(1+1e-12)
                assert right >= 3*(1-1e-12)
                min_left_ratio=min(min_left_ratio,left)
                min_right_ratio=min(min_right_ratio,right)
                count+=1
            # theta0-theta_b may lie anywhere in [-2h,2h].
            # Do not form a=1-h for tiny h; use (1+a)/h=(2-h)/h.
            scale=(2-h)/h
            for offset in [-2,-1,0,1,2]:
                ulo=(-5-offset)*h
                uhi=(-4-offset)*h
                alpha=(math.atan(scale*math.tan(uhi/2))-math.atan(scale*math.tan(ulo/2)))/math.pi
                assert alpha >= 1/(100*math.pi)*(1-1e-12)
                min_alpha=min(min_alpha,alpha)
    return {'arc_point_checks':count,'minimum_arc_mass':min_alpha,
            'claimed_arc_mass_lower':1/(100*math.pi),
            'minimum_left_distance_over_eps_wb':min_left_ratio,
            'minimum_right_distance_over_eps_wb':min_right_ratio}


def source_hashes(root):
    names=['curvature_hierarchy.tex','fixed_quartet.tex','uniform_log_upper.tex',
           'UNIFORM_LOG_BOUND_PROOF.md']
    return {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}


if __name__ == '__main__':
    here=Path(__file__).resolve().parent
    work=here.parents[1]
    first_n=2
    while 16*math.log(first_n)/first_n > 1/16: first_n+=1
    logK=math.log(64)+100*math.pi*math.log(512)
    report={'status':'all assertions passed',
            'limitations':'Finite arithmetic/geometry checks support, but do not prove, universal or asymptotic claims.',
            'counts':{'two_cell_partitions':7,'three_cell_partitions':6,'nested_pairs':18},
            'reference_hinge':hinge_checks(),
            'polynomial_enumeration':polynomial_checks(),
            'endpoint_poisson_geometry':poisson_geometry_checks(),
            'parameter_thresholds':{'first_n_ge_2_with_eps_le_1_16':first_n,
                                    'curvature_degree_at_threshold':4*first_n,
                                    'logK':logK,'log10K':logK/math.log(10)},
            'source_sha256':source_hashes(work/'historical/v1')}
    target=here/'audit_checks.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

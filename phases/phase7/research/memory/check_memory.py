#!/usr/bin/env python3
"""Exact algebra checks only. Standard library; no training or installation."""
from fractions import Fraction as Q
from itertools import combinations_with_replacement
from math import comb
from pathlib import Path
import json
import random


def add(a, b, scale=1):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + scale*v
        if not out[k]:
            del out[k]
    return out


def mul(a, b):
    out = {}
    for e, c in a.items():
        for f, v in b.items():
            k = tuple(x+y for x, y in zip(e, f))
            out[k] = out.get(k, 0)+c*v
    return {k: v for k, v in out.items() if v}


def rank(rows):
    """Sparse row reduction over Q."""
    pivots = {}
    for raw in rows:
        row = {k: Q(v) for k, v in raw.items() if v}
        while row:
            lead = min(row)
            if lead not in pivots:
                factor = row[lead]
                pivots[lead] = {k: v/factor for k, v in row.items()}
                break
            factor = row[lead]
            row = add(row, pivots[lead], -factor)
    return len(pivots)


def monomial(d, indices, ypower=0):
    exponents = [0]*(d+1)
    for i in indices:
        exponents[i] += 1
    exponents[-1] = ypower
    return {tuple(exponents): 1}


def feature_rank(d):
    pairs = list(combinations_with_replacement(range(d), 2))
    one = monomial(d, [])
    y = monomial(d, [], 1)
    hs = [add(monomial(d, [i, j]), one, -(i == j)) for i, j in pairs]
    products = [mul(hs[i], hs[j]) for i, j in combinations_with_replacement(range(len(hs)), 2)]
    labels = [mul(y, h) for h in hs]
    p, q = comb(d+1, 2), comb(d+3, 4)
    input_rank = rank([one]+hs+products)-1
    full_rank = rank([one]+hs+products+labels)-1
    no_intercept_rank = rank([one]+products+labels)-1
    assert input_rank == p+q
    assert full_rank == 2*p+q
    expected_no_intercept = 2 if d == 1 else 9 if d == 2 else 2*p+q
    assert no_intercept_rank == expected_no_intercept
    return dict(d=d, p=p, q=q, input_dimension=input_rank,
                full_dimension=full_rank, no_intercept_dimension=no_intercept_rank)


def evaluate(poly, z):
    return sum(c*prod(zj**ej for zj, ej in zip(z, e)) for e, c in poly.items())


def prod(values):
    result = Q(1)
    for value in values:
        result *= value
    return result


def derivative(poly, a):
    out = {}
    for e, c in poly.items():
        if e[a]:
            f = list(e)
            f[a] -= 1
            out[tuple(f)] = c*e[a]
    return out


def jacobian_check(d):
    p, q = comb(d+1, 2), comb(d+3, 4)
    n = p+q
    pairs = list(combinations_with_replacement(range(d), 2))
    inputs = [monomial(d, indices) for degree in (2, 4)
              for indices in combinations_with_replacement(range(d), degree)]
    one = monomial(d, [])
    hs = [add(monomial(d, [i, j]), one, -(i == j)) for i, j in pairs]
    rng = random.Random(743+d)
    for attempt in range(20):
        points = [tuple([Q(rng.randrange(-5, 6)) for _ in range(d)]+[Q(0)])
                  for _ in range(n)]
        dx = [{(i, a): evaluate(derivative(f, a), x)
               for i, x in enumerate(points) for a in range(d)} for f in inputs]
        hy = [{i: evaluate(h, x) for i, x in enumerate(points)} for h in hs]
        input_rank, label_rank = rank(dx), rank(hy)
        if input_rank == n and label_rank == p:
            return dict(d=d, n=n, input_jacobian_rank=input_rank,
                        label_jacobian_rank=label_rank,
                        full_jacobian_rank=input_rank+label_rank,
                        deterministic_draws=attempt+1)
    raise AssertionError((d, input_rank, label_rank))


def reconstruction_check():
    d = 3
    xs = [[Q(v) for v in row] for row in
          [[1, 2, -1], [0, 3, 1], [-2, -1, 2], [2, 0, 1], [1, -1, 0]]]
    ys = [Q(1), Q(-2), Q(3), Q(1, 2), Q(4, 3)]
    n = len(xs)
    hs = [[[x[i]*x[j]-int(i == j) for j in range(d)] for i in range(d)] for x in xs]
    sy = [[sum(y*h[i][j] for y, h in zip(ys, hs))/(2*n) for j in range(d)] for i in range(d)]
    m2 = [[sum(x[i]*x[j] for x in xs)/n for j in range(d)] for i in range(d)]
    m4 = {(i,j,k,l):sum(x[i]*x[j]*x[k]*x[l] for x in xs)/n
          for i in range(d) for j in range(d) for k in range(d) for l in range(d)}
    b = [[Q(2), Q(1,3), Q(-1,2)], [Q(1,3), Q(-1), Q(2,5)],
         [Q(-1,2), Q(2,5), Q(3,2)]]
    c = Q(7, 6)
    direct, reconstructed = [], []
    for i in range(d):
        for j in range(d):
            gd = sum((y-sum(b[k][l]*h[k][l] for k in range(d) for l in range(d))-c)
                     *h[i][j] for y,h in zip(ys,hs))/(2*n)
            t = sum(b[k][l]*(m4[i,j,k,l]-int(i==j)*m2[k][l]
                            -int(k==l)*m2[i][j]+int(i==j)*int(k==l))
                    for k in range(d) for l in range(d))/2
            gr = sy[i][j]-t-c*(m2[i][j]-int(i==j))/2
            direct.append(gd)
            reconstructed.append(gr)
    assert direct == reconstructed
    return dict(d=d, n=n, exact_equality=True,
                residual_entries=[str(v) for v in direct])


def collision_check():
    # Work in x^2 to avoid an irrational sqrt(2) representation.
    squares_a, squares_b = [Q(1), Q(1)], [Q(0), Q(2)]
    def summary(squares):
        hs = [s-1 for s in squares]
        return dict(S_y=Q(0), M2=sum(squares)/len(squares),
                    Hbar=sum(hs)/len(hs), H2=sum(h*h for h in hs)/len(hs),
                    G_1_0=-sum(h*h for h in hs)/(2*len(hs)))
    a, b = summary(squares_a), summary(squares_b)
    assert a['S_y'] == b['S_y'] and a['M2'] == b['M2'] and a['Hbar'] == b['Hbar']
    assert a['G_1_0'] == 0 and b['G_1_0'] == Q(-1, 2)
    return {key: {k: str(v) for k,v in val.items()} for key,val in [('A',a),('B',b)]}


def main():
    result = dict(status='PASS', arithmetic='exact fractions; standard library only',
                  feature_ranks=[feature_rank(d) for d in range(1,6)],
                  jacobian_checks=[jacobian_check(d) for d in range(1,5)],
                  empirical_query_reconstruction=reconstruction_check(),
                  fourth_moment_collision=collision_check(),
                  phase6_dimensions=dict(d=12,p=78,q=1365,input=1443,total=1521),
                  scope='Algebraic checks support but do not replace the universal proofs.')
    path = Path(__file__).with_name('check_results.json')
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

"""Independent stdlib exact checks; no imports from pilot implementations."""

from fractions import Fraction as F
from itertools import product
import json
import math
from pathlib import Path


def partitions(n):
    out = []

    def rec(labels):
        if len(labels) == n:
            out.append(tuple(tuple(i for i, b in enumerate(labels) if b == k)
                             for k in range(max(labels) + 1)))
            return
        for label in range(max(labels) + 2):
            rec(labels + [label])

    rec([0])
    return out


parts = partitions(4)
p2 = [p for p in parts if len(p) <= 2]
p3 = [p for p in parts if len(p) <= 3]


def refines(fine, coarse):
    return all(any(set(cell) <= set(parent) for parent in coarse)
               for cell in fine)


pairs = [(coarse, fine) for coarse, fine in product(p2, p3)
         if refines(fine, coarse)]
full_pairs = [(coarse, fine) for coarse, fine in pairs
              if len(coarse) == 2 and len(fine) == 3]
assert (len(parts), len(p2), len(p3), len(pairs), len(full_pairs)) == (15, 8, 14, 39, 18)


def evaluate(phi, xs, ps):
    costs = {}
    for part in parts:
        cost = F(0)
        for cell in part:
            mass = sum(ps[i] for i in cell)
            mean = sum(ps[i] * xs[i] for i in cell) / mass
            cost += sum(ps[i] * phi(xs[i]) for i in cell) - mass * phi(mean)
        costs[part] = cost
    o2 = min(costs[p] for p in p2)
    o3 = min(costs[p] for p in p3)
    additive = [(max(costs[c] - o2, costs[f] - o3), c, f) for c, f in pairs]
    delta = min(row[0] for row in additive)
    price = [(max(costs[c] / o2, costs[f] / o3), c, f) for c, f in pairs]
    rho = min(row[0] for row in price)
    return {
        'opt2': o2, 'opt3': o3, 'delta': delta, 'rho': rho,
        'delta_optimizers': [(c, f) for v, c, f in additive if v == delta],
        'rho_optimizers': [(c, f) for v, c, f in price if v == rho],
        'exact_delta': min(max(costs[c] - o2, costs[f] - o3) for c, f in full_pairs),
    }


reference_checks = []
for eps in (F(1, 16), F(1, 32), F(1, 64)):
    def phi(z):
        return max(-z - eps, 0) + max(z - 1 - eps, 0) + 4 * eps**2 * (z - F(1, 2))**2

    result = evaluate(phi, tuple(map(F, (-1, 0, 1, 2))), (F(1, 4),) * 4)
    assert result['delta'] == result['exact_delta'] == eps / 4
    assert set(result['delta_optimizers']).isdisjoint(result['rho_optimizers'])
    result['epsilon'] = eps
    result['normalized_reference_delta'] = result['delta'] / (4 + 64 * eps**2)
    reference_checks.append(result)

# Fully rational finite proof used in the weighted-family upper bound.
exp5_partial = sum(F(5**k, math.factorial(k)) for k in range(8))
assert exp5_partial > 128
weighted_uniform_upper = 4 * F(25, 1024)**3
assert weighted_uniform_upper < F(1, 1000)


def encode(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {key: encode(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(val) for val in value]
    return value


report = encode({
    'counts': {'all_partitions': len(parts), 'at_most_2': len(p2), 'at_most_3': len(p3),
               'at_most_hierarchies': len(pairs), 'exact_hierarchies': len(full_pairs)},
    'reference_checks': reference_checks,
    'exp5_partial_sum_through_7': exp5_partial,
    'weighted_uniform_delta_upper': weighted_uniform_upper,
    'all_assertions_passed': True,
    'scope': 'Exact original nonsmooth reference and analytic bound only; no new pilot inputs.',
})
output = Path(__file__).with_name('independent_exact_checks.json')
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

"""Exact rational checks for the equal-mass smooth-gadget proof.

Only standard-library arithmetic is used. Smoothing is handled by proving
every evaluated mean is outside both smoothing bands, where the convolution
equals the raw hinge exactly. This does not approximate the C-infinity
gluing or optimize over all stochastic channels.
Author: Wenyu Huang, UCSD.
"""

import argparse
from fractions import Fraction as Q
import json
from pathlib import Path


def partitions(n):
    def visit(i, blocks):
        if i == n:
            yield tuple(tuple(block) for block in blocks)
            return
        for pos in range(len(blocks)):
            blocks[pos].append(i)
            yield from visit(i + 1, blocks)
            blocks[pos].pop()
        blocks.append([i])
        yield from visit(i + 1, blocks)
        blocks.pop()
    yield from visit(0, [])


def check(j):
    e = Q(1, j)
    smoothing_width = e / 4
    locations = (-1, 0, 1, 2)

    def f(z):
        # This argument is used only where smooth and raw hinges agree.
        assert abs(-z - e) >= smoothing_width
        assert abs(z - 1 - e) >= smoothing_width
        return max(Q(0), -z - e) + max(Q(0), z - 1 - e) + 4 * e**2 * (z - Q(1, 2))**2

    def cell_cost(weighted_indices):
        mass = sum((weight for _, weight in weighted_indices), Q(0))
        mean = sum((weight * locations[i] for i, weight in weighted_indices), Q(0)) / mass
        return sum((weight * f(Q(locations[i])) for i, weight in weighted_indices), Q(0)) - mass * f(mean)

    def cost(part):
        return sum((cell_cost([(i, Q(1, 4)) for i in cell]) for cell in part), Q(0))

    a = e / 4 + e**2 / 2
    r = e**2 / 2
    b = (1 - e) / 4 + 2 * e**2
    q = (1 - e) / 2 + Q(9, 2) * e**2
    t = b
    u = (1 - e) / 2 + Q(14, 3) * e**2
    expected = {
        (0, 1): a, (2, 3): a,
        (0, 2): b, (1, 3): b,
        (1, 2): r, (0, 3): q,
        (0, 1, 2): t, (1, 2, 3): t,
        (0, 1, 3): u, (0, 2, 3): u,
    }
    for cell, value in expected.items():
        assert cell_cost([(i, Q(1, 4)) for i in cell]) == value
    allparts = list(partitions(4))
    twos = [part for part in allparts if len(part) == 2]
    threes = [part for part in allparts if len(part) == 3]
    costs = {part: cost(part) for part in allparts}
    d2 = min(costs[part] for part in twos)
    d3 = min(costs[part] for part in threes)
    assert d2 == 2 * a and d3 == r and r > 0
    assert [part for part in twos if costs[part] == d2] == [((0, 1), (2, 3))]
    assert [part for part in threes if costs[part] == d3] == [((0,), (1, 2), (3,))]
    hierarchies = [(coarse, fine) for coarse in twos for fine in threes
                   if all(any(set(fc) <= set(cc) for cc in coarse) for fc in fine)]
    assert len(hierarchies) == 18
    alpha, beta = a / r, t / (2 * a)
    for coarse, fine in hierarchies:
        c2, c3 = costs[coarse] / d2, costs[fine] / d3
        assert (c2 >= 1 and c3 >= alpha) or (c2 >= beta and c3 >= 1)
    rho = min(max(costs[coarse] / d2, costs[fine] / d3) for coarse, fine in hierarchies)
    assert rho == beta == Q(j * j - j + 8, 2 * j + 4)

    fine_cost = (
        cell_cost([(0, Q(1, 4))])
        + cell_cost([(1, Q(1, 4)), (2, Q(1, 8))])
        + cell_cost([(2, Q(1, 8)), (3, Q(1, 4))])
    )
    coarse_cost = (
        cell_cost([(0, Q(1, 4)), (1, Q(1, 4)), (2, Q(1, 8))])
        + cell_cost([(2, Q(1, 8)), (3, Q(1, 4))])
    )
    assert fine_cost == e / 8 + Q(2, 3) * e**2
    assert coarse_cost == Q(1, 8) + e / 2 + Q(26, 15) * e**2
    mixed_lower = 1 + (alpha - 1) * (beta - 1) / (alpha + beta - 2)
    stochastic_upper = max(fine_cost / d3, coarse_cost / d2)
    assert stochastic_upper == Q(j, 4) + Q(4, 3)
    assert mixed_lower <= stochastic_upper
    return {
        "j": j,
        "source_mass": "1/4 at every source state",
        "hierarchies_enumerated": len(hierarchies),
        "det_price_exact": str(rho),
        "det_price_decimal": float(rho),
        "stochastic_lower_exact": str(mixed_lower),
        "stochastic_upper_exact": str(stochastic_upper),
        "all_rational_assertions_passed": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = [check(j) for j in (16, 17, 32, 64, 128, 256, 1024)]
    body = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(body, encoding="utf-8")
    print(body, end="")

#!/usr/bin/env python3
"""Reproduce the C02 lower-family exact-rational partition/channel diagnostic.

Standard library only. All costs are polynomials of degree at most two in
epsilon. Polynomial nonnegativity is checked over the entire closed interval
[0, 1/16] by exact endpoint and (when applicable) interior-vertex evaluation.
The conclusions about ratios apply for epsilon > 0, where denominators are
positive. This checks the reference hinge gadget, not numerical polynomial
kernel integration or global stochastic optimization.
"""

import argparse
from fractions import Fraction as Q
import json
from pathlib import Path


Z = (Q(-1), Q(0), Q(1), Q(2))
W = (Q(1, 4),) * 4
ZERO = (Q(0),) * 3
EPS_MAX = Q(1, 16)


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def scale(p, c):
    return tuple(c * x for x in p)


def sub(a, b):
    return add(a, scale(b, -1))


def evaluate(p, epsilon):
    return sum(c * epsilon**i for i, c in enumerate(p))


def reference_generator(x):
    """Coefficients of (-x-e)_+ + (x-1-e)_+ + 4e^2(x-1/2)^2."""
    # Every point evaluated by this diagnostic stays in one hinge region
    # throughout the interval. Check this premise explicitly.
    if x < 0:
        assert x <= -EPS_MAX
    elif x > 1:
        assert x >= 1 + EPS_MAX
    return (
        -x if x < 0 else x - 1 if x > 1 else Q(0),
        Q(-1) if x < 0 or x > 1 else Q(0),
        4 * (x - Q(1, 2)) ** 2,
    )


def cost(weights):
    mass = sum(weights)
    mean = sum(wi * zi for wi, zi in zip(weights, Z)) / mass
    ans = ZERO
    for wi, zi in zip(weights, Z):
        ans = add(ans, scale(reference_generator(zi), wi))
    return sub(ans, scale(reference_generator(mean), mass))


def cell(subset):
    return cost(tuple(W[i] if i in subset else Q(0) for i in range(4)))


def partitions(items):
    if not items:
        yield ()
        return
    first, *rest = items
    for partition in partitions(rest):
        yield ((first,),) + partition
        for j in range(len(partition)):
            yield (
                partition[:j]
                + ((first,) + partition[j],)
                + partition[j + 1 :]
            )


def partition_cost(partition):
    ans = ZERO
    for subset in partition:
        ans = add(ans, cell(subset))
    return ans


def refines(fine, coarse):
    return all(any(set(a) <= set(b) for b in coarse) for a in fine)


def interval_minimum(p):
    points = [Q(0), EPS_MAX]
    if p[2] > 0:
        vertex = -p[1] / (2 * p[2])
        if 0 < vertex < EPS_MAX:
            points.append(vertex)
    return min((evaluate(p, x), x) for x in points)


def nonnegative(p):
    return interval_minimum(p)[0] >= 0


def stringify_poly(p):
    return [str(c) for c in p]


def channel_cost(channel):
    ans = ZERO
    for weights in channel:
        ans = add(ans, cost(weights))
    return ans


def run_check():
    all_partitions = list(partitions(list(range(4))))
    fine = [p for p in all_partitions if len(p) <= 3]
    coarse = [p for p in all_partitions if len(p) <= 2]
    hierarchies = [(c, f) for c in coarse for f in fine if refines(f, c)]
    assert len(all_partitions) == 15
    assert len(hierarchies) == 39

    expected = {
        "a": ((0, 1), (Q(0), Q(1, 4), Q(1, 2))),
        "b": ((0, 2), (Q(1, 4), Q(-1, 4), Q(2))),
        "s": ((1, 2), (Q(0), Q(0), Q(1, 2))),
        "q": ((0, 3), (Q(1, 2), Q(-1, 2), Q(9, 2))),
        "t": ((0, 1, 2), (Q(1, 4), Q(-1, 4), Q(2))),
        "u": ((0, 1, 3), (Q(1, 2), Q(-1, 2), Q(14, 3))),
    }
    for subset, coefficients in expected.values():
        assert cell(subset) == coefficients

    s = cell((1, 2))
    a = cell((0, 1))
    t = cell((0, 1, 2))
    twoa = scale(a, 2)
    assert all(nonnegative(sub(partition_cost(p), s)) for p in fine)
    assert all(nonnegative(sub(partition_cost(p), twoa)) for p in coarse)
    assert any(partition_cost(p) == s for p in fine)
    assert any(partition_cost(p) == twoa for p in coarse)

    hierarchy_results = []
    corner_costs = {"(2a,a)": (twoa, a), "(t,s)": (t, s)}
    for c, f in hierarchies:
        cc, fc = partition_cost(c), partition_cost(f)
        dominated_by = []
        certificates = {}
        for name, (corner_c, corner_f) in corner_costs.items():
            differences = (sub(cc, corner_c), sub(fc, corner_f))
            if all(nonnegative(p) for p in differences):
                dominated_by.append(name)
                certificates[name] = [
                    {
                        "difference_coefficients": stringify_poly(p),
                        "minimum": str(interval_minimum(p)[0]),
                        "minimizer": str(interval_minimum(p)[1]),
                    }
                    for p in differences
                ]
        assert dominated_by, (c, f)
        hierarchy_results.append(
            {
                "coarse": c,
                "fine": f,
                "coarse_cost": stringify_poly(cc),
                "fine_cost": stringify_poly(fc),
                "dominated_by": dominated_by,
                "whole_interval_certificates": certificates,
            }
        )
    for corner in corner_costs.values():
        assert any(
            (partition_cost(c), partition_cost(f)) == corner
            for c, f in hierarchies
        )

    fine_channel = [
        (Q(1, 4), Q(0), Q(0), Q(0)),
        (Q(0), Q(1, 4), Q(1, 8), Q(0)),
        (Q(0), Q(0), Q(1, 8), Q(1, 4)),
    ]
    coarse_channel = [
        tuple(x + y for x, y in zip(fine_channel[0], fine_channel[1])),
        fine_channel[2],
    ]
    fc = channel_cost(fine_channel)
    cc = channel_cost(coarse_channel)
    assert fc == (Q(0), Q(1, 8), Q(2, 3))
    assert cc == (Q(1, 8), Q(1, 2), Q(26, 15))

    examples = []
    for epsilon in [Q(1, 16), Q(1, 64), Q(1, 1024)]:
        opt2 = min(evaluate(partition_cost(p), epsilon) for p in coarse)
        opt3 = min(evaluate(partition_cost(p), epsilon) for p in fine)
        rho = min(
            max(
                evaluate(partition_cost(c), epsilon) / opt2,
                evaluate(partition_cost(f), epsilon) / opt3,
            )
            for c, f in hierarchies
        )
        beta = evaluate(t, epsilon) / evaluate(twoa, epsilon)
        assert rho == beta
        examples.append(
            {
                "epsilon": str(epsilon),
                "opt2": str(opt2),
                "opt3": str(opt3),
                "deterministic_price": str(rho),
                "equals_t_over_2a": rho == beta,
            }
        )

    return {
        "status": "all assertions passed",
        "arithmetic": "Python fractions.Fraction; no floating-point calculations",
        "coefficient_order": ["constant", "epsilon", "epsilon_squared"],
        "source_labels": ["A", "C", "D", "B"],
        "locations": [str(x) for x in Z],
        "weights": [str(x) for x in W],
        "whole_interval": ["0", "1/16"],
        "ratio_domain": "0 < epsilon <= 1/16",
        "minimum_method": "exact endpoints plus interior vertex for convex quadratics",
        "set_partition_count": len(all_partitions),
        "at_most_2_partition_count": len(coarse),
        "at_most_3_partition_count": len(fine),
        "at_most_2_3_hierarchy_count": len(hierarchies),
        "cell_costs": {
            name: stringify_poly(coefficients)
            for name, (_, coefficients) in expected.items()
        },
        "flat_optima": {"opt2": stringify_poly(twoa), "opt3": stringify_poly(s)},
        "both_pareto_cost_corners_attained": True,
        "every_hierarchy_dominated_by_a_corner_on_whole_interval": True,
        "explicit_channel": {
            "fine_cost": stringify_poly(fc),
            "coarse_cost": stringify_poly(cc),
        },
        "examples": examples,
        "hierarchies": hierarchy_results,
        "scope_limit": "Reference hinge costs and explicit channel only; not numerical polynomial kernel integration or global stochastic optimization.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("lower_partition_results.json"),
    )
    args = parser.parse_args()
    result = run_check()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])
    print(f"Set partitions: {result['set_partition_count']}")
    print(f"At-most (2,3) hierarchies: {result['at_most_2_3_hierarchy_count']}")
    print("Whole-interval flat minima and Pareto domination verified.")
    print("Both corner attainments and explicit stochastic channel verified.")
    print(f"Results: {args.output}")

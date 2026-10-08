#!/usr/bin/env python3
"""Independent exact checks; no external packages and no floating point."""

from fractions import Fraction as Q


def partitions(items):
    if not items:
        yield ()
        return
    head, *tail = items
    for rest in partitions(tail):
        yield ((head,),) + rest
        for j, block in enumerate(rest):
            yield rest[:j] + ((head,) + block,) + rest[j + 1:]


def canonical(p):
    return tuple(sorted(tuple(sorted(b)) for b in p))


PARTITIONS = tuple(sorted(set(canonical(p) for p in partitions(tuple(range(4))))))
Z = (Q(-1), Q(0), Q(1), Q(2))


def generator(z, e):
    return max(Q(0), -z - e) + max(Q(0), z - 1 - e) + 4 * e**2 * (z - Q(1, 2))**2


def cost(p, e):
    result = Q(0)
    for b in p:
        mean = sum((Z[i] for i in b), Q(0)) / len(b)
        result += sum((generator(Z[i], e) for i in b), Q(0)) - len(b) * generator(mean, e)
    return result / 4


def refines(fine, coarse):
    return all(any(set(a) <= set(b) for b in coarse) for a in fine)


def label(p):
    return "|".join("".join(str(i + 1) for i in b) for b in p)


def check(e):
    costs = {p: cost(p, e) for p in PARTITIONS}
    coarse = tuple(p for p in PARTITIONS if len(p) <= 2)
    fine = tuple(p for p in PARTITIONS if len(p) <= 3)
    compatible = tuple((a, b) for a in coarse for b in fine if refines(b, a))
    assert len(PARTITIONS) == 15
    assert len(coarse) == 8 and len(fine) == 14 and len(compatible) == 39
    opt2 = min(costs[p] for p in coarse)
    opt3 = min(costs[p] for p in fine)
    assert opt2 == e / 2 + e**2
    assert opt3 == e**2 / 2
    gaps = {(a, b): max(costs[a] - opt2, costs[b] - opt3) for a, b in compatible}
    delta = min(gaps.values())
    assert delta == e / 4
    best = sorted((label(a), label(b)) for (a, b), gap in gaps.items() if gap == delta)
    assert best == [("12|34", "12|3|4"), ("12|34", "1|2|34")]
    for a in coarse:
        if label(a) != "12|34":
            assert costs[a] - opt2 > e / 4
    print(f"e={e}: partitions=15, compatible=39, OPT2={opt2}, OPT3={opt3}, delta={delta}, minimizers={best}")


if __name__ == "__main__":
    for parameter in (Q(1, 16), Q(1, 32), Q(1, 64)):
        check(parameter)
    print("PASS: all rational checks agree with the analytic envelope.")

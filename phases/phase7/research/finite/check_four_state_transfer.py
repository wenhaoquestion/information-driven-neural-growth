"""Finite enumeration supporting the imported four-state lower-bound family.

Standard library only. This checks the reference costs and all 18 hierarchies.
The C-infinity gluing and exact affine transfer are proved analytically in the
accompanying TeX; this calculation is not a numerical proof of those facts.
Author: Wenyu Huang, UCSD (research support script).
"""

from decimal import Decimal as D, localcontext
import json


def partitions(n):
    def visit(i, blocks):
        if i == n:
            yield tuple(tuple(b) for b in blocks)
            return
        for pos in range(len(blocks)):
            blocks[pos].append(i)
            yield from visit(i + 1, blocks)
            blocks[pos].pop()
        blocks.append([i])
        yield from visit(i + 1, blocks)
        blocks.pop()
    yield from visit(0, [])


def entropy(q):
    if q == 0 or q == 1:
        return D(0)
    return -q * q.ln() - (1 - q) * (1 - q).ln()


def calculate(j):
    with localcontext() as ctx:
        ctx.prec = 100 + 3 * j
        L = D(j)
        w = (-L).exp()
        c = (1 - 2 * w) / 2
        eps = (-2 * L).exp()
        log2 = D(2).ln()
        delta = (log2 / L).sqrt()
        p = (c, w, w, c)
        q = (eps, delta, 1 - delta, 1 - eps)
        base = sum((p[i] * entropy(q[i]) for i in range(4)), D(0))

        def risk(part):
            ans = D(0)
            for cell in part:
                mass = sum((p[i] for i in cell), D(0))
                mean = sum((p[i] * q[i] for i in cell), D(0)) / mass
                ans += mass * entropy(mean)
            return ans - base

        allparts = list(partitions(4))
        two = [part for part in allparts if len(part) == 2]
        three = [part for part in allparts if len(part) == 3]
        costs = {part: risk(part) for part in two + three}
        opt2 = min(costs[part] for part in two)
        opt3 = min(costs[part] for part in three)
        hierarchies = [
            (coarse, fine)
            for coarse in two for fine in three
            if all(any(set(fc) <= set(cc) for cc in coarse) for fc in fine)
        ]
        assert len(two) == 7 and len(three) == 6 and len(hierarchies) == 18
        rho = min(max(costs[c2] / opt2, costs[c3] / opt3)
                  for c2, c3 in hierarchies)
        near = costs[((0, 1), (2,), (3,))]
        rare = costs[((0,), (1, 2), (3,))]
        triple = costs[((0, 1, 2), (3,))]
        formula = min(near / rare, triple / (2 * near))
        assert abs(rho - formula) < D(10) ** (-(ctx.prec // 2))
        assert costs[((0, 1), (2, 3))] == opt2
        assert rare == opt3
        return {
            "j": j,
            "partitions_at_2": len(two),
            "partitions_at_3": len(three),
            "hierarchies": len(hierarchies),
            "det_excess_price": str(+rho)[:30],
            "price_over_asymptotic": str(rho / (L.sqrt() / (2 * log2.sqrt())))[:30],
            "baseline_over_D3": str(base / opt3)[:30],
            "exact_eventual_formula_verified": True,
        }


if __name__ == "__main__":
    print(json.dumps([calculate(j) for j in (8, 16, 32, 64, 128, 256)], indent=2))

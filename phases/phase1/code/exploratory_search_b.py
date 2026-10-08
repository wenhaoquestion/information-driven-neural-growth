"""Original unoptimized exploratory search that suggested the quartet.

This is discovery-only numerical evidence. Its naive floating-point
entropy subtraction is unsuitable for certifying extremely small costs.
The manuscript's dedicated quartet experiment uses high precision.
Run from the project root with:
    python3 code/exploratory_search_b.py > results/exploratory_search_b.json
"""
import json
import math
import random

random.seed(841)


def ent(v):
    s = sum(v)
    return s * math.log(s) - sum(x * math.log(x) if x else 0 for x in v) if s else 0


pairs = [(1 << i) | (1 << j) for i in range(4) for j in range(i)]
part2 = [(a, 15 ^ a) for a in range(1, 15) if a < 15 ^ a]
best = (1, None)
for t in range(150000):
    dim = random.choice([2, 3, 4, 6])
    spread = random.choice([2, 5, 10, 20, 40])
    v = [[math.exp(random.uniform(-spread, 0)) for j in range(dim)] for i in range(4)]
    e = [ent(x) for x in v]
    costs = {0: 0}
    for a in range(1, 16):
        inds = [i for i in range(4) if (a >> i) & 1]
        costs[a] = max(0, ent([sum(v[i][j] for i in inds) for j in range(dim)]) - sum(e[i] for i in inds))
    opt2 = min(costs[a] + costs[b] for a, b in part2)
    opt3 = min(costs[p] for p in pairs)
    if opt2 < 1e-14 or opt3 < 1e-14:
        continue
    rho = min(max((costs[a] + costs[b]) / opt2, costs[p] / opt3)
              for p in pairs for a, b in part2
              if (p & a) == p or (p & b) == p)
    if rho > best[0]:
        best = (rho, {"v": v, "C": costs, "opt2": opt2,
                      "opt3": opt3, "spread": spread, "t": t})
print(json.dumps(best))

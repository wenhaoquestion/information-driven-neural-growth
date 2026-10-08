#!/usr/bin/env python3
"""Fresh bounded exact checks. Standard library only; no inherited test code.

This is supporting finite evidence, not an asymptotic proof.
Run from the workspace root: python3 cross_stage_audit_work/research/weighted_top/adversarial_exact.py
"""
from fractions import Fraction as F
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import hashlib
import json
import random
import sys

ROOT = Path(__file__).resolve().parent

@lru_cache(None)
def partitions(n):
    if n == 0:
        return ((),)
    out = []
    bit = 1 << (n - 1)
    for p in partitions(n - 1):
        out.append(p + (bit,))
        for j in range(len(p)):
            out.append(p[:j] + (p[j] | bit,) + p[j + 1:])
    return tuple(out)

def canon(p):
    return tuple(sorted(p))

def is_contiguous(mask):
    low = (mask & -mask).bit_length() - 1
    return (mask >> low) == (1 << (mask.bit_length() - low)) - 1

@lru_cache(None)
def hierarchy_indices(n):
    ps = partitions(n)
    lookup = {canon(p): i for i, p in enumerate(ps)}
    pairs = []
    for fi, fp in enumerate(ps):
        if len(fp) > n - 1:
            continue
        for grouping in partitions(len(fp)):
            if len(grouping) > n - 2:
                continue
            cp = []
            for group in grouping:
                mask = 0
                for j, fine_mask in enumerate(fp):
                    if group & (1 << j):
                        mask |= fine_mask
                cp.append(mask)
            pairs.append((fi, lookup[canon(cp)]))
    return tuple(pairs)

def source_costs(x, p, phi):
    n = len(x)
    dc = {0: F(0)}
    for mask in range(1, 1 << n):
        inds = [i for i in range(n) if mask & (1 << i)]
        mass = sum(p[i] for i in inds)
        mean = sum(p[i] * x[i] for i in inds) / mass
        dc[mask] = sum(p[i] * phi(x[i]) for i in inds) - mass * phi(mean)
        assert dc[mask] >= 0
    return dc

def analyze(x, p, phi):
    n = len(x)
    ps = partitions(n)
    dc = source_costs(x, p, phi)
    costs = [sum(dc[m] for m in part) for part in ps]
    q = min(c for c, part in zip(costs, ps) if len(part) <= n - 1)
    coarse = min(c for c, part in zip(costs, ps) if len(part) <= n - 2)
    assert q > 0 and coarse > 0
    all_pairs = hierarchy_indices(n)
    ratios = [(max(costs[fi] / q, costs[ci] / coarse), fi, ci) for fi, ci in all_pairs]
    price = min(t[0] for t in ratios)
    full_price = min(t[0] for t in ratios if len(ps[t[1]]) == n - 1 and len(ps[t[2]]) == n - 2)
    assert price == full_price, "At-most versus full-budget hierarchy mismatch"
    assert q == min(dc[(1 << i) | (1 << (i + 1))] for i in range(n - 1))
    contiguous_opts = [part for cost, part in zip(costs, ps) if cost == coarse and all(is_contiguous(mask) for mask in part)]
    assert contiguous_opts
    cheapest_edges = [(1 << i) | (1 << (i + 1)) for i in range(n - 1) if dc[(1 << i) | (1 << (i + 1))] == q]
    conflicts_checked = 0
    if price > 1:
        for edge in cheapest_edges:
            for opt in contiguous_opts:
                nons = [mask for mask in opt if mask.bit_count() > 1]
                assert len(nons) == 2 and all(mask.bit_count() == 2 for mask in nons)
                support = nons[0] | nons[1]
                assert support.bit_count() == 4 and is_contiguous(support)
                inds = [i for i in range(n) if support & (1 << i)]
                assert edge == ((1 << inds[1]) | (1 << inds[2]))
                sub = analyze_quartet([x[i] for i in inds], [p[i] for i in inds], phi)
                assert sub[0] == q and sub[1] == coarse and price <= sub[2]
                conflicts_checked += 1
    return {"n": n, "flat_fine": str(q), "flat_coarse": str(coarse), "price": str(price),
            "at_most_hierarchies": len(all_pairs), "conflict_choices_checked": conflicts_checked,
            "partitions": len(ps)}

def analyze_quartet(x, p, phi):
    ps = partitions(4)
    dc = source_costs(x, p, phi)
    costs = [sum(dc[m] for m in part) for part in ps]
    q = min(c for c, part in zip(costs, ps) if len(part) <= 3)
    coarse = min(c for c, part in zip(costs, ps) if len(part) <= 2)
    price = min(max(costs[fi] / q, costs[ci] / coarse) for fi, ci in hierarchy_indices(4))
    return q, coarse, price

def kernel_checks():
    rng = random.Random(619271)
    checked = 0
    for _ in range(20000):
        u = F(1, 10 ** rng.randrange(0, 13))
        v = u * rng.choice([F(1), F(10), F(10 ** 8)])
        w = F(1, 10 ** rng.randrange(0, 13))
        c = rng.choice([F(1, 10 ** 8), F(1, 100), F(1, 2), 1 - F(1, 10 ** 8)])
        t = rng.choice([c / 10, c / 2, c - c / 10 ** 8, c + (1 - c) / 10 ** 8, (1 + c) / 2])
        K = min(u * t + v * max(t - c, 0), w * (1 - t) + v * max(c - t, 0))
        H = min(u * t, w * (1 - t))
        kq = min(u * t, v * (c - t)) if t <= c else F(0)
        kp = min(v * (t - c), w * (1 - t)) if t >= c else F(0)
        assert 0 <= K - H <= kq + kp
        if t < c:
            assert kq / K >= min(1, (c - t) / t)
        elif t > c:
            assert kp / K >= (t - c) / (c + 2 * (t - c))
        checked += 1
    return checked

def main():
    rng = random.Random(20261007)
    records = []
    for n in range(3, 8):
        for trial in range(5):
            raw_gaps = [rng.choice([1, 2, 11, 1000]) for _ in range(n - 1)]
            x = [F(0)]
            for gap in raw_gaps:
                x.append(x[-1] + gap)
            x = [value / x[-1] for value in x]
            masses = [F(rng.choice([1, 3, 17]), rng.choice([1, 10, 10 ** 6])) for _ in range(n)]
            p = [mass / sum(masses) for mass in masses]
            shift, degree = F(rng.randrange(0, 5), 4), 2 * rng.randrange(1, 5)
            eps = F(1, 10 ** rng.randrange(1, 8))
            phi = lambda t, shift=shift, degree=degree, eps=eps: (t - shift) ** degree + eps * t * t
            rec = analyze(x, p, phi)
            rec["label"] = f"random_{n}_{trial}"
            records.append(rec)

    # A deliberately conflicting core, then exact far-anchor embeddings for N=5..7.
    x4 = [F(0), F(1, 8), F(7, 8), F(1)]
    b = F(1, 100)
    p4 = [F(1, 2) - b, b, b, F(1, 2) - b]
    eps = F(1, 10 ** 8)
    phi = lambda t: (t - F(1, 2)) ** 20 + eps * t * t
    core = analyze(x4, p4, phi)
    core["label"] = "designed_conflict_core"
    assert F(core["price"]) > 1
    records.append(core)
    for m in range(1, 4):
        # floor g >= 2 eps; every source mass >= 1/200; L=10^7 gives pair penalty >=2500.
        ax = [F(-j * 10 ** 7) for j in range(m, 0, -1)] + x4
        ap = [F(1, 2 * m)] * m + [mass / 2 for mass in p4]
        rec = analyze(ax, ap, phi)
        rec["label"] = f"far_anchor_{m}"
        assert F(rec["price"]) == F(core["price"])
        assert F(rec["flat_fine"]) == F(core["flat_fine"]) / 2
        assert F(rec["flat_coarse"]) == F(core["flat_coarse"]) / 2
        records.append(rec)

    sources = [
        "phases/phase8/manuscript/unequal_weights_endpoint.tex",
        "phases/phase9/manuscript/weighted_four_state_upper.tex",
        "phases/phase9/manuscript/endpoint_lower.tex",
        "phases/phase10/manuscript/model.tex",
        "phases/phase10/manuscript/quartet_reduction.tex",
        "phases/phase10/manuscript/degree_upper.tex",
        "phases/phase10/manuscript/anchor_embedding.tex",
        "phases/phase10/manuscript/endpoint_lower.tex",
        "phases/phase10/manuscript/quartet_analytic_upper.tex",
    ]
    result = {"status": "passed", "python": sys.version, "rational_kernel_checks": kernel_checks(),
              "source_instances": len(records), "hierarchy_candidates": sum(r["at_most_hierarchies"] for r in records),
              "nontrivial_instances": sum(F(r["price"]) > 1 for r in records),
              "conflict_choices_checked": sum(r["conflict_choices_checked"] for r in records),
              "records": records,
              "source_sha256": {p: hashlib.sha256((ROOT.parents[3] / p).read_bytes()).hexdigest() for p in sources}}
    (ROOT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "source_sha256")}, indent=2))

if __name__ == "__main__":
    main()

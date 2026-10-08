#!/usr/bin/env python3
"""Independent finite computational checks of the quartet upper-bound proof.

Run with python research/main_review_checks.py.
These randomized checks challenge constants and boundary cases; they are not
proofs. All partitions are exactly enumerated for every generated instance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def partitions(n: int):
    def recurse(i, blocks):
        if i == n:
            yield tuple(tuple(b) for b in blocks)
            return
        for j in range(len(blocks)):
            blocks[j].append(i)
            yield from recurse(i + 1, blocks)
            blocks[j].pop()
        blocks.append([i])
        yield from recurse(i + 1, blocks)
        blocks.pop()
    yield from recurse(0, [])


PARTS = list(partitions(4))


def costs(w, p, part):
    u = np.sqrt(p)
    divergence, variance = 0.0, 0.0
    for block in part:
        ids = np.asarray(block)
        wb = w[ids]
        q = np.sum(wb[:, None] * p[ids], axis=0) / np.sum(wb)
        mean = np.sum(wb[:, None] * u[ids], axis=0) / np.sum(wb)
        variance += float(np.sum(wb[:, None] * (u[ids] - mean) ** 2))
        for i in ids:
            positive = p[i] > 0
            divergence += w[i] * float(np.sum(p[i, positive] * np.log(p[i, positive] / q[positive])))
    return max(0.0, divergence), variance


def main():
    seed, count = 191733, 2000
    rng = np.random.default_rng(seed)
    largest_kl_vs_v = 0.0
    largest_forced_pair_ratio = 0.0
    worst = None
    for trial in range(count + 2):
        m = 2 + trial % 7
        w = np.exp(rng.uniform(-16, 0, 4))
        w /= w.sum()
        p = np.exp(rng.uniform(-30, 0, (4, m)))
        # Explicit support-zero cases: each row retains at least one coordinate.
        if trial % 3 == 0:
            p[rng.random(p.shape) < 0.35] = 0
            p[:, 0] += 1e-7
        p /= p.sum(axis=1, keepdims=True)
        if trial == count:
            p[:] = p[0]  # identical posterior degeneracy
        if trial == count + 1:
            p[:] = 0
            p[:, 0] = 1  # deterministic constant target, all costs zero
        computed = [(part, *costs(w, p, part)) for part in PARTS]
        c = 2 + np.log(1 / np.min(w))
        for _, d, v in computed:
            tol = 2e-13
            assert v <= d + tol, (trial, v, d)
            assert d <= 2 * c * v + tol, (trial, d, v, c)
            if v > 1e-12:
                largest_kl_vs_v = max(largest_kl_vs_v, d / (2 * c * v))
        level3 = min((row for row in computed if len(row[0]) == 3), key=lambda row: row[2])
        pair = next(block for block in level3[0] if len(block) == 2)
        optimum2 = min(row[2] for row in computed if len(row[0]) == 2)
        compatible2 = min(
            row[2] for row in computed
            if len(row[0]) == 2 and any(set(pair).issubset(block) for block in row[0])
        )
        assert 2 * level3[2] <= optimum2 + 2e-13
        assert compatible2 <= 5 * optimum2 + 2e-13
        if optimum2 > 1e-12 and compatible2 / optimum2 > largest_forced_pair_ratio:
            largest_forced_pair_ratio = compatible2 / optimum2
            worst = dict(trial=trial, weights=w.tolist(), posteriors=p.tolist(), pair=pair)
    result = dict(
        seed=seed,
        random_instances=count,
        explicit_degeneracies=2,
        partitions_per_instance=len(PARTS),
        target_sizes=list(range(2, 9)),
        max_fraction_of_kl_upper_bound=largest_kl_vs_v,
        max_forced_minimum_pair_euclidean_ratio=largest_forced_pair_ratio,
        worst_euclidean_witness=worst,
        status="all assertions passed",
    )
    path = Path(__file__).with_name("main_review_checks.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Auxiliary frozen-feature regression diagnostic; not a neural-growth benchmark.

Repeated empirical cost gates are deliberately heuristic: squared Gaussian loss
is unbounded and candidate selection is not multiplicity corrected. Each event
gets fresh validation labels. The exact population evaluator cannot select.
Run: python3 phase3/code/complementarity_diagnostic.py
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def feature_matrix(epsilon: float) -> np.ndarray:
    # Latent coordinates U_1..U_4,V_1..V_4,N_1..N_4, all independent N(0,1).
    # This matrix defines the data distribution, not controller observations.
    a = np.zeros((12, 12))
    for j in range(4):
        a[j, 2 * j:2 * j + 2] = 1
        a[4 + j, 2 * j] = epsilon
        a[4 + j, 2 * j + 1] = -epsilon
    a[8:, 8:] = np.eye(4)
    return a


def sample(rng, n, a, beta):
    latent = rng.normal(size=(n, 12))
    return latent @ a, latent @ beta + rng.normal(scale=0.5, size=n)


def fit(x, y, support):
    z = np.column_stack((np.ones(len(y)), x[:, support]))
    return np.linalg.lstsq(z, y, rcond=None)[0]


def predict(x, support, coef):
    return coef[0] + x[:, support] @ coef[1:]


def risk(a, beta, support, coef):
    # Integrates continuous Gaussian inputs and response noise exactly.
    discrepancy = a[:, support] @ coef[1:] - beta
    return float(0.25 + coef[0] ** 2 + discrepancy @ discrepancy)


def run(seed, epsilon, signal, max_block):
    a = feature_matrix(epsilon)
    beta = np.zeros(12)
    if signal:
        beta[4:8] = [1, 0.8, 0.6, 0.4]
    rng = np.random.default_rng(seed)
    xt, yt = sample(rng, 1024, a, beta)
    # Both policies use the same sequence of validation samples.
    validations = [sample(rng, 512, a, beta) for _ in range(10)]
    support = []
    coef = fit(xt, yt, support)
    penalty = 0.03
    events = []
    all_candidates = []
    fits = 1
    lstsq_proxy = len(yt)
    start = time.perf_counter()
    for t, (xv, yv) in enumerate(validations):
        base_val = float(np.mean((predict(xv, support, coef) - yv) ** 2))
        best_score = 0.0
        best = None
        available = sorted(set(range(12)) - set(support))
        attempts = 0
        for block_size in range(1, max_block + 1):
            for block in itertools.combinations(available, block_size):
                new_support = support + list(block)
                candidate = fit(xt, yt, new_support)
                val = float(np.mean((predict(xv, new_support, candidate) - yv) ** 2))
                score = base_val - val - penalty * block_size
                all_candidates.append(dict(seed=seed, epsilon=epsilon, signal=signal,
                    max_block=max_block, event=t, block=list(block),
                    validation_gain=base_val-val, penalized_gain=score,
                    population_risk=risk(a,beta,new_support,candidate)))
                attempts += 1
                fits += 1
                lstsq_proxy += len(yt) * (len(new_support) + 1) ** 2
                if score > best_score:
                    best_score = score
                    best = (new_support, candidate, block)
        if best is not None:
            support, coef, block = best
        else:
            block = ()
        events.append(dict(event=t, accepted=best is not None, added=list(block),
            width=len(support), score=best_score, attempts=attempts,
            risk=risk(a,beta,support,coef), output_l2=float(np.linalg.norm(coef[1:])),
            cumulative_fits=fits, cumulative_lstsq_proxy=lstsq_proxy,
            fresh_validation_labels=512))
    elapsed=time.perf_counter()-start
    return dict(seed=seed,epsilon=epsilon,signal=signal,max_block=max_block,
        train_labels=1024,validation_labels=5120,penalty=penalty,
        support=support,coef=coef.tolist(),risk=risk(a,beta,support,coef),
        width=len(support),events=events,fits=fits,lstsq_proxy=lstsq_proxy,
        wall_seconds=elapsed), all_candidates


def main():
    (ROOT / 'results').mkdir(exist_ok=True)
    results=[]
    raw = ROOT/'results/complementarity_candidates.jsonl'
    with raw.open('w') as stream:
        for epsilon in [0.1,0.25]:
            for signal in [False,True]:
                for seed in range(93000,93032):
                    for max_block in [1,2]:
                        row,candidates=run(seed,epsilon,signal,max_block)
                        results.append(row)
                        for c in candidates:
                            stream.write(json.dumps(c)+'\n')
    (ROOT/'results/complementarity_runs.json').write_text(json.dumps(results,indent=2))
    summaries=[]
    for epsilon,signal,max_block in itertools.product([0.1,0.25],[False,True],[1,2]):
        rows=[r for r in results if (r['epsilon'],r['signal'],r['max_block'])==(epsilon,signal,max_block)]
        out=dict(epsilon=epsilon,signal=signal,max_block=max_block,runs=len(rows))
        for key in ['risk','width','fits','lstsq_proxy','wall_seconds']:
            out[key+'_mean']=float(np.mean([r[key] for r in rows]))
        out['any_growth']=sum(r['width']>0 for r in rows)
        summaries.append(out)
    (ROOT/'results/complementarity_summary.json').write_text(json.dumps(summaries,indent=2))
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()

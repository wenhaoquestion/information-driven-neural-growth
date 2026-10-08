"""Independent reconstruction of the completed primary statistics from raw JSON.

Does not import analysis/analyze.py or alter the endpoint, sample, or test.
Distribution summaries are descriptive diagnostics, never alternative decisions.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('historical_moment', 'random_growth128', 'random_growth138', 'fixed7')
BUDGETS = np.asarray([3200000, 5000000, 6400000, 8000000, 10000000])


def main():
    paths = sorted((ROOT/'experiments/confirmation_primary/results').glob('*.json'))
    cube = np.empty((256, 4, 5))
    ids, filled = [], set()
    for path in paths:
        raw = json.loads(path.read_text())
        i = raw['seed']-96100000
        assert 0 <= i < 256 and raw['task'] == 'indefinite_noisy'
        ids.append(raw['seed'])
        for result in raw['trajectories']:
            assert result['age_mode'] == 'global'
            j = METHODS.index(result['method'])
            k = int(np.flatnonzero(BUDGETS == result['budget'])[0])
            assert (i,j,k) not in filled
            filled.add((i,j,k))
            cube[i,j,k] = result['final_excess_risk']
    assert sorted(ids) == list(range(96100000, 96100256))
    assert len(filled) == 5120
    # Form the fixed nodal trapezoid weights independently, then matrix multiply.
    intervals = np.log(BUDGETS[1:]/BUDGETS[:-1])
    weights = np.r_[intervals[0], intervals[:-1]+intervals[1:], intervals[-1]]
    weights /= 2*np.log(BUDGETS[-1]/BUDGETS[0])
    auc = cube @ weights
    differences = auc[:, :1]-auc[:, 1:]
    rows = []
    for j, method in enumerate(METHODS[1:]):
        values = differences[:, j]
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        se = sd/16
        rows.append(dict(comparator=method, n=256, mean=mean, sd=sd, se=se,
                         upper95_one_sided=mean+float(t.ppf(.95,255))*se,
                         p_one_sided_historical_lower=float(t.cdf(mean/se,255))))
    # Different chunk size than production, same prespecified whole-row draws.
    rng = np.random.default_rng(96070031)
    draws = np.empty((10000, 3))
    for start in range(0, 10000, 1000):
        resampled = rng.integers(0, 256, (1000, 256))
        draws[start:start+1000] = differences[resampled].mean(axis=1)
    quantiles = np.quantile(draws, [.025,.975], axis=0)
    reported = json.loads((ROOT/'analysis/confirmation/analysis_summary.json').read_text())
    for j, row in enumerate(rows):
        production = next(item for item in reported['primary_comparisons'] if item['comparator'] == row['comparator'])
        for key, value in row.items():
            if key not in ('comparator',):
                np.testing.assert_allclose(value, production[key], atol=1e-14, rtol=1e-11)
        np.testing.assert_allclose(quantiles[:,j], [production['bootstrap95_low'],production['bootstrap95_high']], atol=1e-14, rtol=1e-11)
        row.update(bootstrap95_low=float(quantiles[0,j]), bootstrap95_high=float(quantiles[1,j]))
    success = all(row['upper95_one_sided'] < 0 for row in rows)
    max_p = max(row['p_one_sided_historical_lower'] for row in rows)
    assert success == reported['primary']['success']
    np.testing.assert_allclose(max_p, reported['primary']['iut_p'], atol=1e-14)
    diagnostics = {}
    for j, method in enumerate(METHODS):
        values = auc[:,j]
        diagnostics[method] = dict(mean=float(values.mean()), sd=float(values.std(ddof=1)),
                                  quantile_levels=[0,.25,.5,.75,.9,.95,.99,1],
                                  quantiles=np.quantile(values,[0,.25,.5,.75,.9,.95,.99,1]).tolist())
    fixed_difference = differences[:,-1]
    most_favorable = np.argsort(fixed_difference)[:5]
    tail = dict(interpretation='Descriptive attribution only; every seed remains in the frozen primary test.',
                paired_median=float(np.median(fixed_difference)),
                paired_historical_lower_count=int(np.sum(fixed_difference < 0)),
                paired_historical_higher_count=int(np.sum(fixed_difference > 0)),
                most_historical_favorable_five=[dict(seed=int(i+96100000), difference=float(fixed_difference[i]),
                                                     historical_auc=float(auc[i,0]), fixed_auc=float(auc[i,-1]))
                                                for i in most_favorable],
                those_five_fraction_of_net_mean_difference=float(fixed_difference[most_favorable].sum()/fixed_difference.sum()))
    # Stored per-seed AUC CSV must preserve the same raw-data pairing.
    with (ROOT/'analysis/confirmation/seed_auc.csv').open() as file:
        for row in csv.DictReader(file):
            if row['group'] == 'indefinite_noisy_global':
                np.testing.assert_allclose(float(row['auc']), auc[int(row['seed'])-96100000, METHODS.index(row['method'])], atol=1e-14)
    result = dict(status='passed', source_primary_datasets=256, source_trajectories=5120,
                  normalized_log_budget_weights=weights.tolist(), paired_comparisons=rows,
                  frozen_iut_success=success, frozen_iut_p=max_p,
                  method_auc_distributions=diagnostics, fixed7_tail_diagnostic=tail,
                  analysis_source_sha256=hashlib.sha256((ROOT/'analysis/analyze.py').read_bytes()).hexdigest(),
                  note='Reproduces the frozen result; no observation dropped, no new test or tuning performed.')
    (ROOT/'reviews/final_statistics_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

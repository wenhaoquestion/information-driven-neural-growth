"""Independent report arithmetic audit; reads existing outcomes, does not train."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'experiments' / 'results'


def main():
    with (RESULTS / 'run_summary.csv').open() as f:
        runs = list(csv.DictReader(f))
    with (RESULTS / 'paired_contrasts.csv').open() as f:
        reported = list(csv.DictReader(f))
    lookup = {(r['task'], int(r['seed']), r['method']): r for r in runs}
    assert len(lookup) == len(runs) == 336
    raw_discrepancies = []
    events = 0
    for path in sorted((RESULTS / 'confirmation').glob('*_75*.json')):
        raw = json.loads(path.read_text())
        for method, record in raw['methods'].items():
            summary = lookup[raw['task'], raw['seed'], method]
            for metric in ['initial_risk', 'final_risk', 'tail_risk']:
                raw_discrepancies.append(abs(record[metric] - float(summary[metric])))
            assert len(record['events']) == 24
            events += len(record['events'])
            assert record['events'][-1]['width'] == int(summary['final_width'])
    assert events == 8064
    assert max(raw_discrepancies) < 1e-14
    pairs = {
        'release_residual': [('released_residual', 1), ('withheld_residual', -1)],
        'release_random': [('released_random', 1), ('withheld_random', -1)],
        'residual_minus_random_released': [('released_residual', 1), ('released_random', -1)],
        'released_residual_minus_fixed2': [('released_residual', 1), ('fixed2_online', -1)],
        'released_residual_minus_fixed26': [('released_residual', 1), ('fixed26_online', -1)],
        'released_residual_minus_fixed26_gated': [('released_residual', 1), ('fixed26_gated', -1)],
        'fixed26_gated_minus_online': [('fixed26_gated', 1), ('fixed26_online', -1)],
        'release_by_proposal_interaction': [('released_residual', 1), ('withheld_residual', -1),
                                          ('released_random', -1), ('withheld_random', 1)]}
    checks = []
    for row in reported:
        task, metric = row['task'], row['metric']
        seeds = sorted({int(r['seed']) for r in runs if r['task'] == task})
        values = np.array([sum(sign * float(lookup[task, seed, method][metric])
                               for method, sign in pairs[row['contrast']]) for seed in seeds])
        n = len(values)
        assert n == int(row['n']) == 12
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        se = sd / np.sqrt(n)
        half = t.ppf(.975, n - 1) * se
        computed = dict(mean=mean, sd=sd, se=se, ci95_low=mean-half, ci95_high=mean+half)
        error = max(abs(value - float(row[key])) for key, value in computed.items())
        assert error < 1e-13
        checks.append(dict(task=task, metric=metric, contrast=row['contrast'],
                           maximum_discrepancy=error))
    result = dict(scope='Arithmetic from archived raw run JSON and CSV summaries; no neural retraining, '
                        'no independent reconstruction of all candidate predictions or e-values.',
                  status='passed', run_rows=len(runs), raw_events=events,
                  paired_contrasts=len(checks),
                  maximum_raw_risk_discrepancy=max(raw_discrepancies),
                  maximum_contrast_discrepancy=max(c['maximum_discrepancy'] for c in checks),
                  sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [RESULTS/'run_summary.csv', RESULTS/'paired_contrasts.csv']},
                  checks=checks)
    (HERE / 'independent_report_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['checks','sources']}, indent=2))


if __name__ == '__main__':
    main()

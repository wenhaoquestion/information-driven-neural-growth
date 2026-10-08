"""Read-only independent validation of completed Phase 6 raw artifacts.

No training, endpoint selection, or significance testing. Writes an audit JSON
under reviews/. Run separately for each existing experiment directory.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def risk_from_parameters(p, teacher, noise):
    w, a, c = np.asarray(p['w']), np.asarray(p['a']), float(p['c'])
    difference = sum(a[j]*np.outer(w[:, j], w[:, j]) for j in range(len(a))) - teacher
    return float(noise*noise + 2*np.einsum('ij,ij->', difference, difference) + c*c)


def audit(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    cfg = manifest['config']
    canonical = json.dumps(cfg, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == manifest['config_sha256']
    expected = set(itertools.product(cfg['tasks'], cfg['seeds'], cfg['methods'], cfg['age_modes'], cfg['budgets']))
    observed, index_hashes = set(), {}
    for path in list(directory.glob('completion_*.json')) + list(directory.glob('failure_*.json')):
        index = json.loads(path.read_text())
        for row in index.get('runs', []):
            key, value = row['result_file'], row['result_sha256']
            if key in index_hashes:
                assert index_hashes[key] == value, 'Conflicting indexed result hashes'
            index_hashes[key] = value
    max_error = 0.
    events_checked, parameters_checked = 0, 0
    result_provenance = []
    for path in sorted((directory/'results').glob('*.json')):
        raw = json.loads(path.read_text())
        assert raw['complete'] is True
        assert raw['config_sha256'] == manifest['config_sha256']
        assert raw['code_sha256'] == manifest['code_sha256']
        relative = str(path.relative_to(directory))
        assert relative in index_hashes, f'Missing indexed result digest: {relative}'
        assert sha(path) == index_hashes[relative], f'Result bytes changed: {relative}'
        datafile = (directory/raw['data_file']).resolve()
        assert datafile.is_relative_to(directory)
        assert sha(datafile) == raw['data_sha256']
        with np.load(datafile, allow_pickle=False) as source:
            x, y = source['x'], source['y']
            teacher, noise = source['teacher_matrix'], float(source['noise'])
        d = cfg['d']
        cumulative_moments = []
        moment = np.zeros((d, d))
        for event in range(cfg['arrivals']):
            left, right = event*cfg['block'], (event+1)*cfg['block']
            bx, by = x[left:right], y[left:right]
            moment += (np.einsum('n,ni,nj->ij', by, bx, bx)-by.sum()*np.eye(d))/2
            cumulative_moments.append(moment.copy())
        for trajectory in raw['trajectories']:
            method, age, budget = trajectory['method'], trajectory['age_mode'], trajectory['budget']
            key = raw['task'], raw['seed'], method, age, budget
            assert key in expected and key not in observed
            observed.add(key)
            assert trajectory['effective_lr'] == cfg.get('method_lrs', {}).get(method, cfg['lr'])
            for field in ('initial', 'final'):
                calculated = risk_from_parameters(trajectory[f'{field}_parameters'], teacher, noise)
                error = abs(calculated-trajectory[f'{field}_risk'])
                np.testing.assert_allclose(calculated, trajectory[f'{field}_risk'], atol=1e-11, rtol=1e-11)
                max_error = max(max_error, error)
                parameters_checked += 1
            np.testing.assert_allclose(trajectory['final_excess_risk'], trajectory['final_risk']-noise*noise, atol=1e-12)
            events = trajectory['events']
            assert len(events) == cfg['arrivals']
            accum = {key: np.zeros(len(y), dtype=np.int64) for key in ('fit', 'moment', 'retention')}
            cumulative_costs = {key: 0 for key in trajectory['cumulative_costs']}
            old_retained, previous_step = [], 0
            for t, event in enumerate(events):
                assert event['t'] == t
                left, right = t*cfg['block'], (t+1)*cfg['block']
                assert event['arrival_range'] == [left, right]
                assert event['width'] == (7 if method == 'fixed7' else t+1)
                assert event['fit_pool_ids'] == old_retained + list(range(left, right))
                assert event['retained_ids_before'] == old_retained
                old_retained = event['retained_ids_after']
                assert len(set(old_retained)) == len(old_retained)
                assert all(0 <= value < right for value in old_retained)
                ids, counts = np.asarray(event['fit_pool_ids']), np.asarray(event['fit_pool_access_counts'])
                assert len(ids) == len(counts)
                assert counts.min() >= 0
                fit = event['fit']
                visits = int(counts.sum())
                assert visits == fit['total_fit_visits'] == event['event_label_accesses']['fit']
                full, small = fit['full_minibatches'], fit['terminal_budget_batch_size']
                assert visits == cfg['batch']*full + small
                assert fit['steps'] == full + int(small > 0)
                assert 0 <= small < cfg['batch']
                assert fit['generated_permutations'] == (visits+len(ids)-1)//len(ids)
                assert fit['permutation_proxy'] == 4*len(ids)*fit['generated_permutations']
                k = event['width']
                p = (d+1)*k+1
                fixed = 30*p+4*d*k+4*k+32
                row = 6*d*k+12*k+2*d+12
                assert fit['gradient_adam_proxy'] == visits*row + fit['steps']*fixed
                assert event['before_optimizer']['step'] == previous_step
                previous_step += fit['steps']
                assert event['after_optimizer']['step'] == previous_step
                assert event['memory']['model_optimizer_arrays_bytes'] == 4*p*8
                assert sum(event['costs'].values()) == event['total_event_proxy'] <= budget
                assert event['total_event_proxy'] + event['unused_event_proxy'] == budget
                assert fit['proxy'] == fit['gradient_adam_proxy'] + fit['permutation_proxy']
                for category, work in event['costs'].items():
                    cumulative_costs[category] += work
                np.add.at(accum['fit'], ids, counts)
                accum['retention'][left:right] += 1
                if method == 'historical_moment':
                    accum['moment'][left:right] += 1
                    np.testing.assert_allclose(event['historical_moment_sum'], cumulative_moments[t], rtol=1e-10, atol=1e-9)
                for field in ('before', 'after'):
                    calculated = risk_from_parameters(event[f'{field}_parameters'], teacher, noise)
                    error = abs(calculated-event[f'risk_{field}'])
                    np.testing.assert_allclose(calculated, event[f'risk_{field}'], atol=1e-11, rtol=1e-11)
                    max_error = max(max_error, error)
                    parameters_checked += 1
                events_checked += 1
            assert cumulative_costs == trajectory['cumulative_costs']
            assert sum(cumulative_costs.values()) == trajectory['total_proxy']
            for category, counts in accum.items():
                np.testing.assert_array_equal(counts, trajectory['label_access_counts'][category])
        result_provenance.append(dict(path=relative, sha256=sha(path), data_sha256=raw['data_sha256']))
    assert expected == observed, f'Missing {len(expected-observed)} planned trajectories'
    return dict(status='passed', directory=str(directory), config_sha256=manifest['config_sha256'],
                code_sha256=manifest['code_sha256'], datasets=len(result_provenance), trajectories=len(observed),
                events_checked=events_checked, parameter_risks_recomputed=parameters_checked,
                max_risk_absolute_error=max_error, actual_result_hashes_verified=True,
                files=result_provenance, audit_script_sha256=sha(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'reviews'):
        raise ValueError('Audit output must be under phase6_work/reviews')
    result = audit(args.run_dir)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key != 'files'}, indent=2))


if __name__ == '__main__':
    main()

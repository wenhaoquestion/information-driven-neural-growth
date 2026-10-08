"""Independent small-fixture audit for frontier.py; no confirmation execution."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import frontier as f
from audit_first_principles import gaussian_grid


def normalized(value):
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()
                if not key.endswith('seconds')}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return value


def small_config():
    c = f.default_config()
    c.update(d=4, arrivals=3, block=11, batch=8,
             historical_capacity=5, replay_capacity=7,
             budgets=[20000, 50000], age_modes=['global', 'local'],
             seeds=[703193])
    return c


def assert_ledger(r, cfg, x, y):
    aggregate = {kind: np.zeros(len(y), dtype=np.int64)
                 for kind in ('fit', 'moment', 'retention')}
    component_sum = {key: 0 for key in r['cumulative_costs']}
    previous_retained = []
    for event in r['events']:
        left, right = event['arrival_range']
        ids = np.asarray(event['fit_pool_ids'])
        counts = np.asarray(event['fit_pool_access_counts'])
        assert np.all(ids >= 0) and np.all(ids < right)
        assert event['retained_ids_before'] == previous_retained
        assert event['fit_pool_ids'] == previous_retained + list(range(left, right))
        previous_retained = event['retained_ids_after']
        assert len(set(previous_retained)) == len(previous_retained)
        assert all(0 <= value < right for value in previous_retained)
        visits = int(counts.sum())
        account = event['fit']
        assert visits == account['total_fit_visits'] == event['event_label_accesses']['fit']
        assert account['generated_permutations'] == math.ceil(visits / len(ids))
        assert account['permutation_proxy'] == 4 * len(ids) * account['generated_permutations']
        sizes = [cfg['batch']] * account['full_minibatches']
        if account['terminal_budget_batch_size']:
            sizes.append(account['terminal_budget_batch_size'])
        assert len(sizes) == account['steps']
        assert sum(sizes) == visits
        d, k = cfg['d'], event['width']
        p = (d+1)*k+1
        # Independent evaluation of frozen declared coefficients, not its helper.
        work = sum(n*(6*d*k+12*k+2*d+12)+30*p+4*d*k+4*k+32 for n in sizes)
        assert work == account['gradient_adam_proxy']
        assert sum(event['costs'].values()) == event['total_event_proxy']
        assert event['budget'] >= event['total_event_proxy']
        assert event['unused_event_proxy'] == event['budget'] - event['total_event_proxy']
        assert event['memory']['model_optimizer_arrays_bytes'] == 4*p*8
        np.add.at(aggregate['fit'], ids, counts)
        aggregate['retention'][left:right] += 1
        if r['method'] == 'historical_moment':
            aggregate['moment'][left:right] += 1
            expected = (np.einsum('n,ni,nj->ij', y[:right], x[:right], x[:right])
                        - np.sum(y[:right])*np.eye(d))/2
            np.testing.assert_allclose(event['historical_moment_sum'], expected, atol=1e-11)
            if event['proposal'] is not None:
                p0 = f.unpack(event['before_parameters'])
                w0, a0 = p0['w'], p0['a']
                residual = expected/right - sum(a0[j]*np.outer(w0[:, j], w0[:, j])
                                                for j in range(len(a0)))
                proposal = event['proposal']
                np.testing.assert_allclose(proposal['matrix'], residual, atol=1e-11)
                direction = np.asarray(proposal['direction'])
                rayleigh = direction @ residual @ direction
                np.testing.assert_allclose(abs(rayleigh), np.max(abs(np.linalg.eigvalsh(residual))), atol=1e-11)
        for key, cost in event['costs'].items():
            component_sum[key] += cost
    assert component_sum == r['cumulative_costs']
    assert sum(component_sum.values()) == r['total_proxy']
    for key, count in aggregate.items():
        np.testing.assert_array_equal(count, r['label_access_counts'][key])


def main():
    cfg = small_config()
    seed, task = cfg['seeds'][0], 'indefinite_noisy'
    hashes_before = f.code_hashes()
    x, y, spec = f.generate(seed, task, cfg)
    all_results = {}
    for age in cfg['age_modes']:
        for budget in cfg['budgets']:
            for method in cfg['methods']:
                key = (method, age, budget)
                result = f.learner(seed, task, method, age, budget, cfg, x, y)
                assert_ledger(result, cfg, x, y)
                all_results[key] = result
    # Reversed loop order must not change parameters, proposals, counts, or costs.
    for key in reversed(list(all_results)):
        method, age, budget = key
        again = f.learner(seed, task, method, age, budget, cfg, x, y)
        assert normalized(again) == normalized(all_results[key])
    # Future X and y change, while every already-observed event remains bitwise identical.
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[22:] += 50
    changed_y[22:] -= 200
    for method in cfg['methods']:
        changed = f.learner(seed, task, method, 'global', 20000, cfg, changed_x, changed_y)
        original = all_results[method, 'global', 20000]
        assert normalized(changed['events'][:2]) == normalized(original['events'][:2])
    # Replay choices and random birth directions share the prescribed streams.
    for budget in cfg['budgets']:
        h = all_results['historical_moment', 'global', budget]
        r = all_results['random_growth128', 'global', budget]
        r138 = all_results['random_growth138', 'global', budget]
        for he, re in zip(h['events'], r['events']):
            assert he['retained_ids_after'] == re['retained_ids_after']
        for re, re138 in zip(r['events'][1:], r138['events'][1:]):
            assert re['proposal']['direction'] == re138['proposal']['direction']
        for age in cfg['age_modes']:
            assert all_results['random_growth128', age, budget]['initial_parameters'] == r['initial_parameters']
    # Fixed-width local ages all match the global age, hence the trajectories agree.
    for budget in cfg['budgets']:
        global_final = all_results['fixed7', 'global', budget]['final_parameters']
        local_final = all_results['fixed7', 'local', budget]['final_parameters']
        for key in global_final:
            np.testing.assert_allclose(global_final[key], local_final[key], atol=1e-12, rtol=1e-12)
    assert all_results['historical_moment', 'global', 20000]['final_parameters'] != all_results['historical_moment', 'local', 20000]['final_parameters']
    # Exact quadrature is independent of the runner's Frobenius-norm evaluator.
    grid, probabilities = gaussian_grid(cfg['d'])
    mean = np.einsum('ni,ij,nj->n', grid, spec['matrix'], grid) - np.trace(spec['matrix'])
    risk_errors = []
    for result in all_results.values():
        evaluated = f.evaluate(copy.deepcopy(result), spec)
        p = f.unpack(result['final_parameters'])
        prediction = sum(p['a'][j]*((grid @ p['w'][:,j])**2-(p['w'][:,j]**2).sum())
                         for j in range(len(p['a']))) + p['c']
        independent = float(probabilities @ (prediction-mean)**2)
        np.testing.assert_allclose(independent, evaluated['final_excess_risk'], atol=1e-12, rtol=1e-12)
        risk_errors.append(abs(independent-evaluated['final_excess_risk']))
    # Observe the run_one orchestration and evaluator barrier on one tiny dataset.
    call_log = []
    actual_learner, actual_evaluate = f.learner, f.evaluate
    def observe_learner(*args, **kwargs):
        result = actual_learner(*args, **kwargs)
        call_log.append('train')
        return result
    def observe_evaluate(*args, **kwargs):
        assert call_log.count('train') == len(all_results)
        call_log.append('evaluate')
        return actual_evaluate(*args, **kwargs)
    with tempfile.TemporaryDirectory(prefix='audit_micro_', dir=ROOT/'reviews') as directory:
        directory = Path(directory)
        (directory/'data').mkdir()
        (directory/'results').mkdir()
        with patch.object(f, 'learner', side_effect=observe_learner), patch.object(f, 'evaluate', side_effect=observe_evaluate):
            first = f.run_one(seed, task, cfg, directory, hashes_before, None)
        before_bytes = (directory/first['result_file']).read_bytes()
        try:
            f.run_one(seed, task, cfg, directory, hashes_before, None)
        except FileExistsError:
            pass
        else:
            raise AssertionError('existing result was not protected')
        assert (directory/first['result_file']).read_bytes() == before_bytes
        resumed = f.run_one(seed, task, cfg, directory, hashes_before, None, resume=True)
        assert resumed['status'] == 'verified_existing'
        altered_cfg = copy.deepcopy(cfg)
        altered_cfg['lr'] = .123
        try:
            f.run_one(seed, task, altered_cfg, directory, hashes_before, None, resume=True)
        except ValueError:
            pass
        else:
            raise AssertionError('changed config accepted by resume')
    hashes_after = f.code_hashes()
    result = dict(status='passed', tiny_shape=list(x.shape), independent_fixture_seed=seed,
                  trajectory_combinations=len(all_results), repeated_reverse_order=len(all_results),
                  future_tamper_methods=len(cfg['methods']), exact_quadrature_risk_max_error=max(risk_errors),
                  run_one_barrier_calls=call_log, code_hashes_before=hashes_before,
                  code_hashes_after=hashes_after, source_unchanged_during_check=hashes_before==hashes_after,
                  verified=['budget caps', 'cost sums', 'full-batch tail carry', 'label ledgers',
                            'historical moment and selected eigenvector', 'future-data isolation',
                            'named RNG order invariance', 'replay and proposal stream pairing',
                            'local/global actual behavior', 'exact risk including intercept',
                            'per-seed evaluation barrier', 'output no-overwrite', 'resume config identity'])
    (ROOT/'reviews/runner_audit_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

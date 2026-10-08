"""Independent analytic fixtures for Phase 6 analysis; no experimental outcomes."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
import frontier

spec = importlib.util.spec_from_file_location('phase6_analysis_audited', ROOT/'analysis/analyze.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def analytical_fixture():
    budgets = np.asarray(analysis.GRID)
    logb = np.log(7*budgets)
    affine = 0.13 + 0.2*(logb-logb[0])
    exact = 0.13 + 0.1*(logb[-1]-logb[0])
    np.testing.assert_allclose(analysis.auc(affine, budgets), exact, atol=1e-14)
    np.testing.assert_allclose(analysis.auc(affine, budgets, arrivals=1), exact, atol=1e-14)
    # Large common teacher variation cancels in the pair; it cannot inflate N.
    n = 256
    z = np.linspace(-1, 1, n)
    risk_h = .2 + .1*z[:, None] + .01*np.arange(5)[None, :]
    matrices = {analysis.METHODS[0]: risk_h}
    for j, method in enumerate(analysis.METHODS[1:]):
        matrices[method] = risk_h + (j+1)*.01 + .001*z[:, None]
    ds = types.SimpleNamespace(seeds=tuple(range(n)), budgets=tuple(budgets), arrivals=7)
    ds.matrix = lambda task, age, method: matrices[method]
    group = analysis.analyze_group(ds, 'indefinite_noisy', 'global', primary=True)
    assert group['iut']['success'] is True
    assert group['iut']['n'] == n
    for index, row in enumerate(group['contrasts']):
        expected_diff = -(index+1)*.01 - .001*z
        se = np.std(expected_diff, ddof=1)/np.sqrt(n)
        upper = expected_diff.mean() + t.ppf(.95, n-1)*se
        p = t.cdf(expected_diff.mean()/se, n-1)
        assert row['n'] == n
        np.testing.assert_allclose(row['mean'], expected_diff.mean(), atol=1e-14)
        np.testing.assert_allclose(row['se'], se, atol=1e-14)
        np.testing.assert_allclose(row['upper95_one_sided'], upper, atol=1e-14)
        np.testing.assert_allclose(row['p_one_sided_historical_lower'], p, atol=1e-14)
    # One adverse comparator must prevent the conjunction, even if two pass.
    matrices[analysis.METHODS[-1]] = risk_h - .01
    failed = analysis.analyze_group(ds, 'indefinite_noisy', 'global', primary=True)
    assert failed['iut']['success'] is False
    assert failed['iut']['iut_p'] > .99
    # Secondary groups cannot become the primary success route.
    matrices[analysis.METHODS[-1]] = risk_h + .01
    secondary = analysis.analyze_group(ds, 'indefinite_noisy', 'local', primary=False)
    assert secondary['iut']['success'] is False
    # Protocol explicitly uses the first evaluated grid success, even if risk rebounds.
    target_function = getattr(analysis, 'first_grid_target_index', None)
    if target_function is None:
        target_function = getattr(analysis, 'first_target_index', None)
    if target_function is None:
        target_function = getattr(analysis, 'stable_target_index')
    assert target_function([.02, .005, .03]) == 1, 'Protocol mismatch: a later rebound cannot erase the first grid success'
    assert target_function([.02, .03, .04]) is None
    return dict(affine_log_budget_integral=exact, independent_units=n,
                comparisons_checked=3, one_adverse_comparator_blocks_iut=True,
                secondary_cannot_declare_primary_success=True,
                nonmonotone_first_grid_target_preserved=True)


def dataset_integrity_fixture():
    cfg = frontier.default_config()
    cfg.update(d=4, arrivals=2, block=7, batch=4, budgets=[5000, 10000],
               historical_capacity=3, replay_capacity=4, seeds=[703199])
    hashes = frontier.code_hashes()
    checks = []
    with tempfile.TemporaryDirectory(prefix='analysis_integrity_', dir=ROOT/'reviews') as directory:
        directory = Path(directory)
        (directory/'data').mkdir()
        (directory/'results').mkdir()
        manifest = dict(config=cfg, config_sha256=frontier.jsonhash(cfg), code_sha256=hashes,
                        environment={'workers': 1})
        (directory/'manifest.json').write_text(json.dumps(manifest))
        row = frontier.run_one(703199, 'indefinite_noisy', cfg, directory, hashes, None)
        target = directory/row['result_file']
        pristine = target.read_bytes()
        index = directory/'completion_audit_fixture.json'
        index.write_text(json.dumps({'runs': [row], 'complete': True}))
        ds = analysis.Dataset(directory)
        assert len(ds.records) == 8
        for name, mutate in [
            ('missing_trajectory', lambda raw: raw['trajectories'].pop()),
            ('duplicate_trajectory', lambda raw: raw['trajectories'].append(copy.deepcopy(raw['trajectories'][0]))),
            ('nonfinite_risk', lambda raw: raw['trajectories'][0].update(final_excess_risk=float('inf'))),
            ('unplanned_seed', lambda raw: raw.update(seed=703198)),
            ('source_identity', lambda raw: raw.update(code_sha256={'frontier.py': 'bad'})),
            ('cap_violation', lambda raw: raw['trajectories'][0]['events'][0].update(total_event_proxy=10**9)),
            ('risk_parameter_mismatch', lambda raw: raw['trajectories'][0].update(
                final_risk=raw['trajectories'][0]['final_risk']+.1,
                final_excess_risk=raw['trajectories'][0]['final_excess_risk']+.1)),
        ]:
            raw = json.loads(pristine)
            mutate(raw)
            target.write_text(json.dumps(raw))
            # Maintain outer byte identity so the structural check is exercised.
            changed_row = dict(row, result_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            index.write_text(json.dumps({'runs': [changed_row], 'complete': True}))
            try:
                analysis.Dataset(directory)
            except ValueError:
                checks.append(name)
            else:
                raise AssertionError(f'invalid fixture accepted: {name}')
            target.write_bytes(pristine)
            index.write_text(json.dumps({'runs': [row], 'complete': True}))
        target.write_bytes(pristine+b' ')
        try:
            analysis.Dataset(directory)
        except ValueError:
            checks.append('indexed_result_bytes_hash')
        else:
            raise AssertionError('changed result bytes with stale indexed hash accepted')
        target.write_bytes(pristine)
        # The actual config bytes and raw data must be checked, not only claimed digests.
        manifest['config']['lr'] = .03
        (directory/'manifest.json').write_text(json.dumps(manifest))
        try:
            analysis.Dataset(directory)
        except ValueError:
            checks.append('actual_manifest_config_hash')
        else:
            raise AssertionError('edited manifest config with stale hash was accepted')
        manifest['config']['lr'] = .015
        (directory/'manifest.json').write_text(json.dumps(manifest))
        rawfile = directory/'data'/'indefinite_noisy_703199.raw.npz'
        with rawfile.open('ab') as file:
            file.write(b'audit-only-corruption')
        try:
            analysis.Dataset(directory)
        except ValueError:
            checks.append('actual_raw_data_hash')
        else:
            raise AssertionError('edited raw data with stale hash was accepted')
    return dict(rejected_corruptions=checks, complete_fixture_trajectories=8)


def main():
    before = hashlib.sha256((ROOT/'analysis/analyze.py').read_bytes()).hexdigest()
    result = dict(status='passed', mathematical=analytical_fixture(),
                  integrity=dataset_integrity_fixture(), analysis_sha256=before,
                  runner_hashes=frontier.code_hashes())
    after = hashlib.sha256((ROOT/'analysis/analyze.py').read_bytes()).hexdigest()
    assert before == after, 'analysis changed during audit; rerun on a stable source'
    (ROOT/'reviews/statistics_audit_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

"""Frozen-protocol, budget-capped quadratic learner comparison.

Work means the declared algorithmic proxy below, NEVER hardware FLOPs.  The
learner accepts only the sequential x/y stream; teachers enter only generation
and a separate evaluator after every trajectory for a dataset has completed.
Audit serialization/access counters are excluded from timed algorithm blocks.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from phase5_reference import effective, grad, init, pack, predict, unpack

METHODS = ('historical_moment', 'random_growth128', 'random_growth138', 'fixed7')
TASKS = ('indefinite_noisy', 'null', 'lowrank_noiseless')
AGE_MODES = ('global', 'local')
COST_VERSION = 'declared_algorithmic_work_v1'


def default_config():
    return dict(schema_version=1, d=12, arrivals=7, block=1024, batch=128,
                lr=.015, gradient_clip=20., budgets=[3200000, 5000000, 6400000, 8000000, 10000000],
                methods=list(METHODS), age_modes=['global'], tasks=['indefinite_noisy'],
                seeds=list(range(256)), historical_capacity=128, replay_capacity=138,
                hard_wall_seconds=1200., randomize_order=True, method_lrs={})


def digest(data):
    return hashlib.sha256(data).hexdigest()


def jsonhash(obj):
    return digest(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def code_hashes():
    return {p.name: digest(p.read_bytes()) for p in
            (Path(__file__).resolve(), Path(__file__).resolve().with_name('phase5_reference.py'))}


def named_seed(seed, task, namespace, *extras):
    """Stable named, mutually separated random streams; independent of loop order."""
    raw = json.dumps([int(seed), task, namespace, *extras], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:16], 'little')


def generator(seed, task, namespace, *extras):
    return np.random.default_rng(named_seed(seed, task, namespace, *extras))


def generate(seed, task, cfg):
    d = cfg['d']
    if task not in TASKS:
        raise ValueError(f'unknown task {task}')
    q, r = np.linalg.qr(generator(seed, task, 'teacher_rotation').normal(size=(d, d)))
    # Fix QR's arbitrary column signs; the rotation still varies independently per seed.
    q *= np.where(np.diag(r) < 0., -1., 1.)
    vals = np.zeros(d)
    if task == 'indefinite_noisy':
        vals[:4] = [.65, -.5, .3, -.2]
    elif task == 'lowrank_noiseless':
        vals[:2] = [.8, .45]
    sigma = {'indefinite_noisy': .3, 'null': .5, 'lowrank_noiseless': 0.}[task]
    matrix = (q * vals) @ q.T
    x = generator(seed, task, 'data_inputs').normal(size=(cfg['arrivals'] * cfg['block'], d))
    noise = generator(seed, task, 'data_noise').normal(size=len(x))
    y = np.sum((x @ matrix) * x, axis=1) - np.trace(matrix) + sigma * noise
    return x, y, dict(matrix=matrix, rotation=q, spectrum=vals, noise=sigma)


def parameter_count(d, k):
    return (d + 1) * k + 1


def fit_batch_work(n, d, k):
    """Includes gradient/forward/gather and clipping/Adam/ages/control, excludes permutations."""
    p = parameter_count(d, k)
    return int(n * (6*d*k + 12*k + 2*d + 12) + 30*p + 4*d*k + 4*k + 32)


def accounting_definition():
    return {
        'version': COST_VERSION,
        'unit': 'declared algorithmic work proxy; not measured or exact hardware FLOPs',
        'P': '(d+1)*k+1; all floating arrays float64 and ID/age arrays int64',
        'fit_gradient_adam': 'n*(6*d*k+12*k+2*d+12) + 30*P+4*d*k+4*k+32 per actual minibatch',
        'fit_explanation': 'row term covers forward, gradient and gathers; fixed term covers clipping, Adam, age updates, finite checks and batch control; conservative fixed coefficients',
        'permutation': '4*fit_pool_rows per generated permutation, including any carried epoch tails',
        'moment_update': '2*n*d*d+n*d+n+4*d*d',
        'historical_residual_matrix': 'd*k+d*d*(2*k-1)+2*d*d (k before birth)',
        'eigendecomposition': '10*d**3',
        'historical_direction': '4*d',
        'random_direction': '8*d (normal draw, norm and normalization)',
        'reservoir_update': 'n*(d+2+8), conservatively charging every incoming row a write plus Algorithm R scalar/RNG work even when discarded',
        'fit_pool_copy': 'fit_pool_rows*(d+2)',
        'arrival_id_construction': 'n',
        'model_initialization': '8*d*k+k+5*P',
        'retention_allocation': 'capacity*(d+2) plus d*d for historical moment initialization',
        'model_birth': '4*P_after+5*(d+1), including parameter/moment/variance/age copies and appended entries',
        'event_control': '64 per event for counters, bounds and loop control',
        'budget_policy': 'reserve all named non-fit costs before fitting; choose largest affordable batch <= batch; only the final budget-limited batch may be small',
        'excluded': 'data generation, post-trajectory evaluator, file I/O, audit snapshots and access-counter bookkeeping',
        'wall_clock': 'sum measured learner algorithm blocks; audit instrumentation excluded; timer overhead included; process contention may affect wall times',
        'memory': 'explicit NumPy array byte inventory and conservative working-array estimate; excludes Python objects, BLAS/eigh internals, audit archives and data owner; not OS peak memory',
        'retention_comparison': 'historical128:128*14+144=1936 8-byte words; replay138:138*14=1932 words at d12, a four-word difference; random128 isolates proposal direction at equal replay rows',
        'rng_namespaces': ['teacher_rotation', 'data_inputs', 'data_noise', 'init', 'fit', 'proposal', 'reservoir', 'execution_order'],
    }


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('configured hard wall-clock limit exceeded; no seed was dropped')


def assert_finite(s):
    for group in ('p', 'm', 'v'):
        for key, value in s[group].items():
            if not np.isfinite(value).all():
                raise FloatingPointError(f'nonfinite {group}.{key}')


def new_state(seed, task, d, k, age_mode):
    if age_mode not in AGE_MODES:
        raise ValueError(age_mode)
    s = init(named_seed(seed, task, 'init', k), d, k)
    s['age'] = {key: np.zeros_like(value, dtype=np.int64) for key, value in s['p'].items()}
    return s


def grow(s, w):
    """Exactly function-preserving zero-output birth; old ages remain unchanged."""
    out = {'step': s['step']}
    for group in ('p', 'm', 'v', 'age'):
        out[group] = {
            'w': np.column_stack((s[group]['w'], w if group == 'p' else np.zeros_like(w, dtype=s[group]['w'].dtype))),
            'a': np.r_[s[group]['a'], np.asarray(0, dtype=s[group]['a'].dtype)],
            'c': s[group]['c'].copy(),
        }
    return out


def adam(s, g, lr, clip, age_mode):
    if any(not np.isfinite(value).all() for value in g.values()):
        raise FloatingPointError('nonfinite gradient')
    norm = np.sqrt(sum(float(np.sum(value * value)) for value in g.values()))
    if not np.isfinite(norm):
        raise FloatingPointError('nonfinite gradient norm')
    scale = min(1., clip / max(norm, 1e-300))
    s['step'] += 1
    for key in s['p']:
        gg = g[key] * scale
        s['m'][key] = .9 * s['m'][key] + .1 * gg
        s['v'][key] = .999 * s['v'][key] + .001 * gg * gg
        s['age'][key] += 1
        age = s['step'] if age_mode == 'global' else s['age'][key]
        s['p'][key] -= lr * s['m'][key] / (1 - .9**age) / (np.sqrt(s['v'][key] / (1 - .999**age)) + 1e-8)
    assert_finite(s)
    return int(scale < 1.)


class Replay:
    """Preallocated Algorithm R; random choices depend on arrival IDs only."""
    def __init__(self, capacity, d, rng):
        self.capacity, self.rng, self.seen, self.size = capacity, rng, 0, 0
        self.x = np.empty((capacity, d))
        self.y = np.empty(capacity)
        self.ids = np.empty(capacity, dtype=np.int64)

    def update(self, x, y, ids):
        for xx, yy, ii in zip(x, y, ids):
            self.seen += 1
            if self.size < self.capacity:
                j = self.size
                self.size += 1
            else:
                j = int(self.rng.integers(self.seen))
            if j < self.capacity:
                self.x[j], self.y[j], self.ids[j] = xx, yy, ii

    def nbytes(self):
        return self.x.nbytes + self.y.nbytes + self.ids.nbytes


def fit(s, x, y, ids, budget, rng, cfg, age_mode, deadline=None):
    """Carry tails between permutations, charging both optimizer and shuffle work."""
    if len(y) == 0 or len(ids) != len(y):
        raise ValueError('nonempty aligned fit pool required')
    counts = np.zeros(len(y), dtype=np.int64)
    spent = gradient_work = permutation_work = steps = full = small = clipped = visits = 0
    minimum, maximum, permutations = None, 0, 0
    order, pos = np.empty(0, dtype=np.int64), 0
    instrumentation_seconds = 0.
    started = time.perf_counter()
    d, k, pool = cfg['d'], len(s['p']['a']), len(y)
    row_work = 6*d*k + 12*k + 2*d + 12
    fixed_work = fit_batch_work(0, d, k)
    while True:
        check_deadline(deadline)
        remain = int(budget) - spent
        take = min(cfg['batch'], max(0, (remain - fixed_work) // row_work))
        available = len(order) - pos
        while take:
            required = (max(0, take - available) + pool - 1) // pool
            work = fit_batch_work(take, d, k) + required * 4 * pool
            if work <= remain:
                break
            take -= 1
        if not take:
            break
        pieces, needed = [], take
        new_permutations = 0
        while needed:
            if pos == len(order):
                order = rng.permutation(pool)
                pos = 0
                new_permutations += 1
            amount = min(needed, len(order) - pos)
            pieces.append(order[pos:pos+amount])
            pos += amount
            needed -= amount
        ix = np.concatenate(pieces)
        clipped += adam(s, grad(x[ix], y[ix], s['p']), cfg['lr'], cfg['gradient_clip'], age_mode)
        gw, pw = fit_batch_work(take, d, k), new_permutations * 4 * pool
        spent += gw + pw
        gradient_work += gw
        permutation_work += pw
        permutations += new_permutations
        steps += 1
        visits += take
        minimum = take if minimum is None else min(minimum, take)
        maximum = max(maximum, take)
        if take == cfg['batch']:
            full += 1
        else:
            if small:
                raise AssertionError('more than one terminal small batch')
            small = take
        audit_started = time.perf_counter()
        np.add.at(counts, ix, 1)
        instrumentation_seconds += time.perf_counter() - audit_started
    elapsed = time.perf_counter() - started - instrumentation_seconds
    if spent > budget:
        raise AssertionError('fit budget overrun')
    return dict(proxy=int(spent), gradient_adam_proxy=int(gradient_work), permutation_proxy=int(permutation_work),
                steps=steps, full_minibatches=full, terminal_budget_batch_size=small,
                min_batch_size=minimum, max_batch_size=maximum, total_fit_visits=visits,
                generated_permutations=permutations, clipped_steps=clipped,
                remainder=int(budget-spent), learner_seconds=elapsed,
                instrumentation_seconds=instrumentation_seconds), counts


def nonfit_costs(method, event, cfg, capacity, old_rows):
    d, n = cfg['d'], cfg['block']
    fixed = method == 'fixed7'
    k = cfg.get('max_width', cfg['arrivals']) if fixed else min(event+1, cfg.get('max_width', cfg['arrivals']))
    birth_now = event > 0 and not fixed and event < cfg.get('max_width', cfg['arrivals'])
    historical = method == 'historical_moment'
    costs = dict(event_control=64, arrival_id_construction=n,
                 fit_pool_copy=(old_rows+n)*(d+2), reservoir_update=n*(d+2+8),
                 moment_update=(2*n*d*d+n*d+n+4*d*d) if historical else 0,
                 historical_residual_matrix=0, eigendecomposition=0, direction_construction=0,
                 model_birth=0, model_initialization=0, retention_allocation=0)
    if event == 0:
        costs['model_initialization'] = 8*d*k+k+5*parameter_count(d, k)
        costs['retention_allocation'] = capacity*(d+2)+(d*d if historical else 0)
    if birth_now:
        previous_k = k-1
        if historical:
            costs['historical_residual_matrix'] = d*previous_k+d*d*(2*previous_k-1)+2*d*d
            costs['eigendecomposition'] = 10*d**3
            costs['direction_construction'] = 4*d
        else:
            costs['direction_construction'] = 8*d
        costs['model_birth'] = 4*parameter_count(d, k)+5*(d+1)
    return costs


def optimizer_snapshot(s):
    return dict(m=pack(s['m']), v=pack(s['v']), age=pack(s['age']), step=int(s['step']))


def array_inventory(s, replay, moment, fitx, fity, fitids, n, batch, proposal):
    d, k = s['p']['w'].shape
    p = parameter_count(d, k)
    model = sum(value.nbytes for group in ('p', 'm', 'v', 'age') for value in s[group].values())
    retained = replay.nbytes() + (0 if moment is None else moment.nbytes)
    pool = fitx.nbytes + fity.nbytes + fitids.nbytes
    # Explicit live-array estimate; this is not a verified allocation peak.
    # A carried tail keeps the old permutation alive while drawing a new one.
    permutation_words = len(fity)*(1+(max(0, batch-1)+len(fity)-1)//len(fity))
    fit_work = 8 * (permutation_words + batch*(d+2+3*k+4) + 8*p + 3*d*k)
    proposal_work = 8 * (6*d*d + n*d + 4*d) if moment is not None else 8*3*d
    birth_work = model if proposal is not None else 0
    return dict(retained_arrays_bytes=int(retained), retained_allocated_replay_bytes=int(replay.nbytes()),
                retained_active_replay_bytes=int(replay.size*(d+2)*8),
                retained_moment_bytes=0 if moment is None else int(moment.nbytes),
                model_optimizer_arrays_bytes=int(model), model_parameters_bytes=int(sum(v.nbytes for v in s['p'].values())),
                optimizer_m_v_age_bytes=int(model-sum(v.nbytes for v in s['p'].values())),
                working_fit_pool_arrays_bytes=int(pool), working_current_input_view_bytes=int(n*(d+1)*8),
                working_fit_array_byte_estimate=int(fit_work), working_proposal_array_byte_estimate=int(proposal_work),
                working_birth_array_byte_estimate=int(birth_work),
                working_array_bytes_estimate=int(pool+8*n+max(fit_work, proposal_work, birth_work)),
                note='NumPy arrays/estimated explicit temporaries, not OS peak. Current input is a borrowed view. Python, BLAS/eigh internals, full raw-data owner and audit archives excluded.')


def learner(seed, task, method, age_mode, budget, cfg, xstream, ystream, deadline=None):
    """No teacher or evaluator argument is accepted by this function."""
    if method not in METHODS or age_mode not in AGE_MODES:
        raise ValueError((method, age_mode))
    effective_lr = cfg.get('method_lrs', {}).get(method, cfg['lr'])
    cfg = dict(cfg, lr=effective_lr)
    d, rounds = cfg['d'], cfg['arrivals']
    if xstream.shape != (rounds*cfg['block'], d) or ystream.shape != (len(xstream),):
        raise ValueError('stream shape does not match configuration')
    capacity = cfg['historical_capacity'] if method in ('historical_moment', 'random_growth128') else cfg['replay_capacity']
    first_costs = nonfit_costs(method, 0, cfg, capacity, 0)
    if sum(first_costs.values()) > budget:
        raise ValueError(f'event 0 non-fit work exceeds cap: {sum(first_costs.values())}>{budget}')
    check_deadline(deadline)
    started = time.perf_counter()
    width = cfg.get('max_width', rounds) if method == 'fixed7' else 1
    s = new_state(seed, task, d, width, age_mode)
    replay = Replay(capacity, d, generator(seed, task, 'reservoir'))
    moment = np.zeros((d, d)) if method == 'historical_moment' else None
    setup_seconds = time.perf_counter()-started
    initial = pack(s['p'])
    initial_optimizer = optimizer_snapshot(s)
    events, elapsed = [], 0.
    cumulative_costs = {key: 0 for key in first_costs}
    cumulative_costs.update(fit_gradient_adam=0, fit_permutation=0)
    total_counts = {key: np.zeros(len(ystream), dtype=np.int64) for key in ('fit', 'moment', 'retention')}
    for event in range(rounds):
        check_deadline(deadline)
        left, right = event*cfg['block'], (event+1)*cfg['block']
        before = pack(s['p'])
        before_opt = optimizer_snapshot(s)
        before_retained = replay.ids[:replay.size].tolist()
        costs = nonfit_costs(method, event, cfg, capacity, replay.size)
        nonfit = sum(costs.values())
        if nonfit > budget:
            raise ValueError(f'event {event} non-fit work exceeds cap: {nonfit}>{budget}')
        event_started = time.perf_counter()
        xx, yy = xstream[left:right], ystream[left:right]
        ids = np.arange(left, right, dtype=np.int64)
        fitx = np.concatenate((replay.x[:replay.size], xx), axis=0)
        fity = np.concatenate((replay.y[:replay.size], yy))
        fitids = np.concatenate((replay.ids[:replay.size], ids))
        if moment is not None:
            moment += ((xx.T * yy) @ xx - yy.sum()*np.eye(d)) / 2
            if not np.isfinite(moment).all():
                raise FloatingPointError('nonfinite historical moment')
        proposal = None
        if costs['model_birth']:
            if moment is not None:
                matrix = moment/right - effective(s['p'])
                values, vectors = np.linalg.eigh(matrix)
                chosen = int(np.argmax(np.abs(values)))
                direction = vectors[:, chosen].copy()
                proposal = dict(estimator='historical Gaussian quadratic moment minus current effective matrix',
                                eigenvalues=values, selected_value=float(values[chosen]), selected_index=chosen,
                                direction=direction, matrix=matrix)
            else:
                direction = generator(seed, task, 'proposal', event).normal(size=d)
                direction /= np.linalg.norm(direction)
                proposal = dict(estimator='isotropic random direction', direction=direction)
            s = grow(s, direction)
        replay.update(xx, yy, ids)
        account, fit_counts = fit(s, fitx, fity, fitids, int(budget)-nonfit,
                                 generator(seed, task, 'fit', event), cfg, age_mode, deadline)
        event_seconds = time.perf_counter()-event_started-account['instrumentation_seconds']
        if event == 0:
            event_seconds += setup_seconds
        elapsed += event_seconds
        # Everything below is archival instrumentation, outside the learner timer.
        np.add.at(total_counts['fit'], fitids, fit_counts)
        total_counts['retention'][ids] += 1
        if moment is not None:
            total_counts['moment'][ids] += 1
        costs['fit_gradient_adam'] = account['gradient_adam_proxy']
        costs['fit_permutation'] = account['permutation_proxy']
        spent = sum(costs.values())
        if spent > budget or fitids.max() >= right:
            raise AssertionError('budget overrun or future data access')
        for key, value in costs.items():
            cumulative_costs[key] += int(value)
        if proposal is not None:
            proposal = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in proposal.items()}
            proposal['proxy'] = costs['historical_residual_matrix']+costs['eigendecomposition']+costs['direction_construction']
        accesses = dict(fit=int(fit_counts.sum()), moment=len(ids) if moment is not None else 0, retention=len(ids), proposal_direct=0)
        item = dict(t=event, arrival_range=[left, right], budget=int(budget), width=len(s['p']['a']),
                    parameters=parameter_count(d, len(s['p']['a'])), before_parameters=before,
                    after_parameters=pack(s['p']), before_optimizer=before_opt, after_optimizer=optimizer_snapshot(s),
                    proposal=proposal, fit=account, costs=costs, nonfit_proxy=nonfit,
                    total_event_proxy=spent, unused_event_proxy=int(budget)-spent,
                    retained_ids_before=before_retained, retained_ids_after=replay.ids[:replay.size].tolist(),
                    fit_pool_ids=fitids.tolist(), fit_pool_access_counts=fit_counts.tolist(),
                    event_label_accesses=accesses,
                    event_unique_fit_labels=int(np.count_nonzero(fit_counts)),
                    learner_seconds=event_seconds, cumulative_learner_seconds=elapsed,
                    memory=array_inventory(s, replay, moment, fitx, fity, fitids, len(yy), cfg['batch'], proposal))
        if moment is not None:
            item['historical_moment_sum'] = moment.tolist()
        events.append(item)
        # Archival values above are independent Python lists. Release event
        # arrays before allocating the next fit pool, avoiding two live pools.
        cleanup_started = time.perf_counter()
        del fitx, fity, fitids, fit_counts, xx, yy, ids
        if costs['model_birth']:
            del direction
            if moment is not None:
                del matrix, values, vectors
        cleanup_seconds = time.perf_counter()-cleanup_started
        elapsed += cleanup_seconds
        item['learner_seconds'] += cleanup_seconds
        item['cumulative_learner_seconds'] = elapsed
    return dict(method=method, age_mode=age_mode, budget=int(budget), effective_lr=float(effective_lr), initial_parameters=initial,
                initial_optimizer=initial_optimizer, events=events, final_parameters=pack(s['p']),
                final_optimizer=optimizer_snapshot(s), learner_seconds=elapsed, cumulative_costs=cumulative_costs,
                total_proxy=sum(cumulative_costs.values()), label_access_counts={key: val.tolist() for key, val in total_counts.items()},
                rng_keys={'init': str(named_seed(seed, task, 'init', width)),
                          'reservoir': str(named_seed(seed, task, 'reservoir')),
                          'fit': [str(named_seed(seed, task, 'fit', t)) for t in range(rounds)],
                          'proposal': [str(named_seed(seed, task, 'proposal', t)) for t in range(1, rounds)]})


def evaluate(trajectory, spec):
    """Called only after all learner trajectories for this dataset are complete."""
    def risk(parameters):
        p = unpack(parameters)
        value = spec['noise']**2 + 2*np.sum((effective(p)-spec['matrix'])**2) + float(p['c'])**2
        if not np.isfinite(value):
            raise FloatingPointError('nonfinite evaluator risk')
        return float(value)
    trajectory['risk_kind'] = 'exact_centered_gaussian_quadratic_population'
    trajectory['noise_variance'] = float(spec['noise']**2)
    trajectory['initial_risk'] = risk(trajectory['initial_parameters'])
    trajectory['final_risk'] = risk(trajectory['final_parameters'])
    trajectory['final_excess_risk'] = trajectory['final_risk']-float(spec['noise']**2)
    for event in trajectory['events']:
        event['risk_before'] = risk(event['before_parameters'])
        event['risk_after'] = risk(event['after_parameters'])
        event['excess_risk_after'] = event['risk_after']-float(spec['noise']**2)
    return trajectory


def exclusive_commit(path, writer):
    """Commit a complete file atomically without replacing an existing result.

    Hard-link publication is atomic and fails if the destination exists. A hard
    process kill may leave an ignored .tmp file, never a partial final artifact.
    """
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}-{time.time_ns()}')
    try:
        with temporary.open('xb') as out:
            writer(out)
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_exclusive(path, obj, compact=False):
    encoded = (json.dumps(obj, sort_keys=True, indent=None if compact else 2,
                         separators=(',', ':') if compact else None, allow_nan=False)+'\n').encode()
    exclusive_commit(path, lambda out: out.write(encoded))


def write_dataset(path, x, y, spec, resume):
    payload = dict(x=x, y=y, teacher_matrix=spec['matrix'], teacher_rotation=spec['rotation'],
                   teacher_spectrum=spec['spectrum'], noise=np.asarray(spec['noise']))
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        with np.load(path, allow_pickle=False) as previous:
            if set(previous.files) != set(payload) or any(not np.array_equal(previous[key], val) for key, val in payload.items()):
                raise ValueError(f'existing raw dataset fails exact regeneration: {path}')
    else:
        exclusive_commit(path, lambda out: np.savez_compressed(out, **payload))
    return digest(path.read_bytes())


def validate_config(cfg):
    expected = set(default_config())
    required = expected - {'method_lrs'}
    if not required <= set(cfg) or set(cfg)-expected:
        raise ValueError(f'config keys mismatch; missing={sorted(required-set(cfg))}, extra={sorted(set(cfg)-expected)}')
    if cfg['schema_version'] != 1 or cfg['d'] != 12 or cfg['arrivals'] != 7:
        raise ValueError('protocol requires schema_version1, d12 and seven arrivals')
    for key in ('block', 'batch', 'historical_capacity', 'replay_capacity'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(key)
    if cfg['block'] != 1024 or cfg['batch'] != 128 or cfg['gradient_clip'] != 20:
        raise ValueError('P5 block/batch/clip are frozen at 1024/128/20')
    if cfg['lr'] not in (.0075, .015, .03):
        raise ValueError('learning rate must belong to the preregistered development grid')
    method_lrs = cfg.get('method_lrs', {})
    if not isinstance(method_lrs, dict) or not set(method_lrs) <= set(METHODS) or any(lr not in (.0075, .015, .03) for lr in method_lrs.values()):
        raise ValueError('method_lrs must contain known methods and preregistered rates')
    if cfg['historical_capacity'] != 128 or cfg['replay_capacity'] != 138:
        raise ValueError('retention capacities frozen at 128 and 138')
    if not cfg['budgets'] or any(type(b) is not int or b <= 0 for b in cfg['budgets']) or len(set(cfg['budgets'])) != len(cfg['budgets']):
        raise ValueError('budgets must be distinct positive integers')
    for key, choices in [('methods', METHODS), ('tasks', TASKS), ('age_modes', AGE_MODES)]:
        if not cfg[key] or len(set(cfg[key])) != len(cfg[key]) or not set(cfg[key]) <= set(choices):
            raise ValueError(key)
    if not cfg['seeds'] or any(type(seed) is not int or seed < 0 for seed in cfg['seeds']) or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('seeds must be distinct nonnegative integers')
    if not isinstance(cfg['randomize_order'], bool) or not np.isfinite(cfg['hard_wall_seconds']) or cfg['hard_wall_seconds'] <= 0:
        raise ValueError('invalid execution control')
    for method in cfg['methods']:
        cap = cfg['historical_capacity'] if method in ('historical_moment', 'random_growth128') else cfg['replay_capacity']
        for event in range(cfg['arrivals']):
            costs = nonfit_costs(method, event, cfg, cap, min(cap, event*cfg['block']))
            if sum(costs.values()) > min(cfg['budgets']):
                raise ValueError('smallest budget cannot pay declared non-fit work')


def run_one(seed, task, cfg, outdir, hashes, deadline, resume=False):
    """One independent dataset, all methods/budgets, then evaluation, then one JSON."""
    outdir = Path(outdir)
    resultfile = outdir/'results'/f'{task}_{seed}.json'
    datafile = outdir/'data'/f'{task}_{seed}.raw.npz'
    check_deadline(deadline)
    if resultfile.exists():
        if not resume:
            raise FileExistsError(resultfile)
        previous = json.loads(resultfile.read_text())
        if previous.get('config_sha256') != jsonhash(cfg) or previous.get('code_sha256') != hashes or previous.get('data_sha256') != digest(datafile.read_bytes()):
            raise ValueError(f'resume hash mismatch: {resultfile}')
        expected_runs = len(cfg['methods'])*len(cfg['budgets'])*len(cfg['age_modes'])
        if previous.get('complete') is not True or len(previous.get('trajectories', [])) != expected_runs:
            raise ValueError(f'incomplete existing result: {resultfile}')
        return dict(seed=seed, task=task, status='verified_existing', result_file=str(resultfile.relative_to(outdir)),
                    result_sha256=digest(resultfile.read_bytes()), data_sha256=previous['data_sha256'],
                    learner_seconds=previous['learner_seconds'])
    started = time.perf_counter()
    x, y, spec = generate(seed, task, cfg)
    datahash = write_dataset(datafile, x, y, spec, resume)
    jobs = [(method, age, budget) for age in cfg['age_modes'] for budget in cfg['budgets'] for method in cfg['methods']]
    if cfg['randomize_order']:
        indices = generator(seed, task, 'execution_order').permutation(len(jobs))
        jobs = [jobs[int(index)] for index in indices]
    trajectories = []
    for method, age, budget in jobs:
        check_deadline(deadline)
        try:
            trajectories.append(learner(seed, task, method, age, budget, cfg, x, y, deadline))
        except Exception as error:
            raise type(error)(f'seed={seed}, task={task}, method={method}, age={age}, budget={budget}: {error}') from error
    # All trajectories finished; the teacher first enters result processing here.
    for trajectory in trajectories:
        evaluate(trajectory, spec)
    out = dict(complete=True, seed=int(seed), task=task, config_sha256=jsonhash(cfg), code_sha256=hashes,
               data_file=str(datafile.relative_to(outdir)), data_sha256=datahash,
               teacher_rotation_seed=str(named_seed(seed, task, 'teacher_rotation')),
               execution_order=[dict(method=m, age_mode=a, budget=int(b)) for m, a, b in jobs],
               trajectories=trajectories, learner_seconds=sum(t['learner_seconds'] for t in trajectories),
               complete_seed_seconds=time.perf_counter()-started,
               evaluation_order='all trajectories completed before any population evaluation')
    write_json_exclusive(resultfile, out, compact=True)
    return dict(seed=int(seed), task=task, status='completed', result_file=str(resultfile.relative_to(outdir)),
                result_sha256=digest(resultfile.read_bytes()), data_sha256=datahash,
                learner_seconds=out['learner_seconds'], complete_seed_seconds=out['complete_seed_seconds'])


def environment_info(workers):
    return dict(python=sys.version, numpy=np.__version__, platform=platform.platform(), command=sys.argv,
                workers=workers, wall_time_interpretation='process-contended' if workers > 1 else 'serial learner wall clock',
                threads={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    validate_config(cfg)
    if args.workers < 1:
        raise ValueError('workers must be positive')
    outdir = args.output_dir.resolve()
    hashes = code_hashes()
    manifest = dict(config=cfg, config_sha256=jsonhash(cfg), code_sha256=hashes,
                    accounting=accounting_definition(), environment=environment_info(args.workers))
    if outdir.exists():
        if not args.resume:
            raise FileExistsError(f'output already exists; use a new directory or --resume: {outdir}')
        previous = json.loads((outdir/'manifest.json').read_text())
        if previous['config_sha256'] != manifest['config_sha256'] or previous['code_sha256'] != hashes:
            raise ValueError('resume requires the identical configuration and both source-code hashes')
        if previous['environment']['workers'] != args.workers:
            raise ValueError('resume requires the original worker count for timing comparability')
    else:
        if args.resume:
            raise FileNotFoundError('--resume requires an existing initialized output directory')
        outdir.mkdir(parents=True)
        (outdir/'data').mkdir()
        (outdir/'results').mkdir()
        write_json_exclusive(outdir/'manifest.json', manifest)
    started = time.monotonic()
    deadline = started + cfg['hard_wall_seconds']
    invocation = str(time.time_ns())
    write_json_exclusive(outdir/f'invocation_{invocation}.json', dict(environment=environment_info(args.workers), resume=args.resume))
    jobs = [(seed, task) for task in cfg['tasks'] for seed in cfg['seeds']]
    rows = []
    try:
        if args.workers == 1:
            for seed, task in jobs:
                row = run_one(seed, task, cfg, outdir, hashes, deadline, args.resume)
                rows.append(row)
                print(json.dumps(dict(progress=len(rows), planned=len(jobs), **row), sort_keys=True), flush=True)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(run_one, seed, task, cfg, outdir, hashes, deadline, args.resume): (seed, task) for seed, task in jobs}
                try:
                    for future in concurrent.futures.as_completed(futures):
                        row = future.result()
                        rows.append(row)
                        print(json.dumps(dict(progress=len(rows), planned=len(jobs), **row), sort_keys=True), flush=True)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
        write_json_exclusive(outdir/f'completion_{invocation}.json', dict(complete=True, completed=len(rows), planned=len(jobs),
                             elapsed_seconds=time.monotonic()-started, runs=rows))
        print(json.dumps(dict(status='COMPLETE', completed=len(rows), planned=len(jobs), elapsed_seconds=time.monotonic()-started)), flush=True)
    except BaseException as error:
        write_json_exclusive(outdir/f'failure_{invocation}.json', dict(complete=False, completed=len(rows), planned=len(jobs),
                             elapsed_seconds=time.monotonic()-started, error_type=type(error).__name__, error=str(error), runs=rows))
        raise


if __name__ == '__main__':
    main()

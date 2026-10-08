"""Observed-data structural direction experiment, NumPy CPU.

No gate, no supplied directions, no claim that moment/spectral methods are new.
All architectures learn hidden vectors and output coefficients after each birth.
The model consumes sequential blocks; population evaluation is performed only
after every parameter/action in a complete trajectory has been saved.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, platform, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = {'null': 0, 'lowrank_noiseless': 1, 'indefinite_noisy': 2, 'nonlinear': 3}
METHODS = ['historical_moment', 'reservoir_residual', 'fresh_residual',
           'learned_probe', 'random_replay128', 'random_matched138', 'fixed7_replay138']

def digest(b): return hashlib.sha256(b).hexdigest()
def jsonhash(x): return digest(json.dumps(x, sort_keys=True, separators=(',', ':')).encode())
def pack(p): return {k: v.tolist() for k, v in p.items()}
def unpack(p): return {k: np.asarray(v, dtype=float) for k, v in p.items()}
def nparams(p): return sum(v.size for v in p.values())
def effective(p): return (p['w'] * p['a']) @ p['w'].T
def predict(x, p):
    z = x @ p['w']; q = z*z - np.sum(p['w']*p['w'], axis=0)
    return q @ p['a'] + p['c']
def grad(x, y, p):
    z = x @ p['w']; q = z*z - np.sum(p['w']*p['w'], axis=0)
    e = q @ p['a'] + p['c'] - y
    return {'w': 4 * p['a'] * ((x.T @ (e[:, None]*z))/len(y) - e.mean()*p['w']),
            'a': 2*q.T @ e/len(y), 'c': np.asarray(2*e.mean())}
def init(seed, d, k):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(d, k)); w /= np.linalg.norm(w, axis=0)
    p = {'w': w, 'a': rng.normal(0, .01, k), 'c': np.asarray(0.)}
    return {'p': p, 'm': {a: np.zeros_like(v) for a, v in p.items()},
            'v': {a: np.zeros_like(v) for a, v in p.items()}, 'step': 0}
def adam(s, g, lr, clip):
    norm = np.sqrt(sum(float(np.sum(v*v)) for v in g.values()))
    scale = min(1., clip/max(norm, 1e-300)); s['step'] += 1; t = s['step']
    for key in s['p']:
        gg = g[key]*scale
        s['m'][key] = .9*s['m'][key] + .1*gg
        s['v'][key] = .999*s['v'][key] + .001*gg*gg
        s['p'][key] -= lr*s['m'][key]/(1-.9**t)/(np.sqrt(s['v'][key]/(1-.999**t))+1e-8)
    return int(scale < 1.)
def birth(s, w):
    out = copy.deepcopy(s)
    for group in ['p', 'm', 'v']:
        out[group]['w'] = np.column_stack([out[group]['w'], w if group == 'p' else np.zeros_like(w)])
        out[group]['a'] = np.r_[out[group]['a'], 0.]
    return out

class Ledger:
    def __init__(self, n):
        self.counts = {k: np.zeros(n, dtype=np.int32) for k in ['fit', 'proposal', 'moment']}
        self.proxy = {'fit': 0, 'proposal': 0, 'moment': 0}
        self.extra = {'eigendecomposition': 0, 'reservoir_seen': 0, 'clipped_steps': 0}
    def add(self, ids, kind, proxy):
        np.add.at(self.counts[kind], ids, 1); self.proxy[kind] += int(proxy)
    def report(self):
        return {'proxy': self.proxy.copy(), 'extra': self.extra.copy(),
                'label_accesses': {k: int(v.sum()) for k, v in self.counts.items()},
                'unique_label_accesses': {k: int(np.count_nonzero(v)) for k, v in self.counts.items()}}

def fit(s, x, y, ids, budget, rng, ledger, cfg):
    # A transparent arithmetic proxy: 3*d*k+4*k+3 per sample for a full step.
    # It excludes hardware/BLAS implementation constants; logs do not call it FLOPs.
    per = 3*cfg['d']*len(s['p']['a']) + 4*len(s['p']['a']) + 3
    spent = 0; steps = 0; order = rng.permutation(len(y)); pos = 0
    while budget-spent >= per:
        if pos == len(order): order = rng.permutation(len(y)); pos = 0
        take = min(cfg['batch'], len(y)-pos, (budget-spent)//per)
        ix = order[pos:pos+take]; pos += take
        ledger.add(ids[ix], 'fit', take*per)
        ledger.extra['clipped_steps'] += adam(s, grad(x[ix], y[ix], s['p']), cfg['lr'], cfg['gradient_clip'])
        spent += take*per; steps += 1
    assert all(np.isfinite(v).all() for v in s['p'].values()), 'nonfinite fitted parameters'
    return {'proxy': int(spent), 'steps': steps, 'remainder': int(budget-spent)}

class Reservoir:
    def __init__(self, capacity, d, rng):
        self.capacity = capacity; self.d = d; self.rng = rng; self.seen = 0
        self.x = np.empty((0, d)); self.y = np.empty(0); self.ids = np.empty(0, dtype=np.int64)
    def update(self, x, y, ids):
        # Algorithm R; replacement is independent of values and shared across arms.
        for xx, yy, ii in zip(x, y, ids):
            self.seen += 1
            if len(self.y) < self.capacity:
                self.x = np.vstack([self.x, xx]); self.y = np.r_[self.y, yy]; self.ids = np.r_[self.ids, ii]
            else:
                j = int(self.rng.integers(self.seen))
                if j < self.capacity: self.x[j] = xx; self.y[j] = yy; self.ids[j] = ii

def empirical_residual_matrix(x, y, ids, p, ledger, cfg):
    r = y-predict(x, p); d = cfg['d']; k = len(p['a'])
    ledger.add(ids, 'proposal', len(y)*(d*d+d*k+3*k+d))
    return ((x.T*r) @ x/len(y) - r.mean()*np.eye(d))/2

def direction(method, s, old, fresh, moment, nseen, rng, ledger, cfg):
    d = cfg['d']; start = sum(ledger.proxy.values()) + ledger.extra['eigendecomposition']
    candidates = None; value = None
    if method == 'historical_moment':
        matrix = moment/nseen - effective(s['p'])
        eig, vec = np.linalg.eigh(matrix); j = int(np.argmax(np.abs(eig))); w = vec[:, j]
        value = float(eig[j]); ledger.extra['eigendecomposition'] += 10*d**3
        info = {'eigenvalues': eig.tolist(), 'estimator': 'S_n minus current A, population Gaussian inverse'}
    elif method in ['fresh_residual', 'reservoir_residual']:
        xx, yy, ii = fresh if method == 'fresh_residual' else old
        matrix = empirical_residual_matrix(xx, yy, ii, s['p'], ledger, cfg)
        eig, vec = np.linalg.eigh(matrix); j = int(np.argmax(np.abs(eig))); w = vec[:, j]
        value = float(eig[j]); ledger.extra['eigendecomposition'] += 10*d**3
        info = {'eigenvalues': eig.tolist(), 'source_size': len(yy), 'estimator': 'empirical residual moment'}
    elif method == 'learned_probe':
        xx, yy, ii = old; count = cfg['probes']
        wmat = rng.normal(size=(d, count)); wmat /= np.linalg.norm(wmat, axis=0)
        initial = wmat.copy()
        for _ in range(cfg['probe_steps']):
            ix = rng.choice(len(yy), min(cfg['batch'], len(yy)), replace=False)
            x = xx[ix]; r = yy[ix]-predict(x, s['p']); z = x@wmat
            q = z*z - np.sum(wmat*wmat, axis=0)
            cov = np.mean(r[:, None]*q, axis=0); den = np.mean(q*q, axis=0)+1e-3
            dc = 2*(x.T@(r[:, None]*z)/len(ix) - r.mean()*wmat)
            dd = 4*(x.T@(q*z)/len(ix) - q.mean(axis=0)*wmat)
            gg = (2*cov/den)*dc - (cov*cov/(den*den))*dd
            wmat += cfg['probe_lr']*gg
            wmat /= np.maximum(np.linalg.norm(wmat, axis=0), 1e-12)
            ledger.add(ii[ix], 'proposal', len(ix)*(5*d*count+d*len(s['p']['a'])+8*count))
        r = yy-predict(xx, s['p']); z = xx@wmat; q = z*z - np.sum(wmat*wmat, axis=0)
        scores = np.mean(r[:, None]*q, axis=0)**2/(np.mean(q*q, axis=0)+1e-3)
        ledger.add(ii, 'proposal', len(yy)*(d*count+d*len(s['p']['a'])+5*count))
        chosen = int(np.argmax(scores)); w = wmat[:, chosen]; value = float(scores[chosen])
        info = {'initial_vectors': initial.tolist(), 'candidate_vectors': wmat.tolist(), 'scores': scores.tolist(), 'chosen': chosen}
    else:
        w = rng.normal(size=d); w /= np.linalg.norm(w); info = {'estimator': 'random direction'}
    spent = sum(ledger.proxy.values())+ledger.extra['eigendecomposition']-start
    return w, {'method': method, 'direction': w.tolist(), 'selected_value': value, 'proxy': int(spent), **info}

def teacher(task, d):
    rng = np.random.default_rng([571902, TASKS[task]]); q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    vals = np.zeros(d)
    if task == 'lowrank_noiseless': vals[:2] = [.8, .45]
    elif task == 'indefinite_noisy': vals[:4] = [.65, -.5, .3, -.2]
    noise = {'null': .5, 'lowrank_noiseless': 0., 'indefinite_noisy': .3, 'nonlinear': .2}[task]
    return {'matrix': (q*vals)@q.T, 'rotation': q, 'noise': noise}
def teacher_mean(x, task, spec):
    if task == 'nonlinear':
        z = x@spec['rotation']; return .8*np.tanh(1.5*z[:, 0])*np.tanh(1.5*z[:, 1]) + .4*np.sin(z[:, 2])
    a = spec['matrix']; return np.sum((x@a)*x, axis=1)-np.trace(a)
def generate(seed, task, cfg):
    rng = np.random.default_rng([seed, TASKS[task], 918]); spec = teacher(task, cfg['d'])
    x = rng.normal(size=(cfg['arrivals']*cfg['block'], cfg['d']))
    y = teacher_mean(x, task, spec)+spec['noise']*rng.normal(size=len(x))
    erng = np.random.default_rng([seed, TASKS[task], 1129]); ex = erng.normal(size=(cfg['neval'], cfg['d']))
    eq = teacher_mean(ex, task, spec)
    return x, y, ex, eq, spec

def learner(seed, task, method, cfg, xstream, ystream):
    fixed = method.startswith('fixed'); k0 = cfg['max_width'] if fixed else 1
    s = init([seed, TASKS[task], 409, k0], cfg['d'], k0)
    capacity = cfg['base_reservoir'] if method in ['historical_moment', 'random_replay128'] else cfg['matched_reservoir']
    reservoir = Reservoir(capacity, cfg['d'], np.random.default_rng([seed, TASKS[task], 500]))
    moment = np.zeros((cfg['d'], cfg['d'])) if method == 'historical_moment' else None
    ledger = Ledger(len(ystream)); events = []; params0 = pack(s['p']); started = time.perf_counter()
    for t in range(cfg['arrivals']):
        left = t*cfg['block']; right = left+cfg['block']; ids = np.arange(left, right)
        xx = xstream[left:right]; yy = ystream[left:right]
        previous_counts = {k: v.copy() for k, v in ledger.counts.items()}
        before = ledger.report(); old = (reservoir.x.copy(), reservoir.y.copy(), reservoir.ids.copy())
        # Fitting has access only to the newly arrived block and retained old rows.
        fitx = np.vstack([old[0], xx]); fity = np.r_[old[1], yy]; fitids = np.r_[old[2], ids]
        proposal = None; moment_cost = 0
        if moment is not None:
            moment += ((xx.T*yy)@xx - yy.sum()*np.eye(cfg['d']))/2
            moment_cost = len(yy)*(cfg['d']**2+cfg['d'])
            ledger.add(ids, 'moment', moment_cost)
        before_p = copy.deepcopy(s['p']); before_state = copy.deepcopy(s)
        if t and not fixed:
            w, proposal = direction(method, s, old, (xx, yy, ids), moment, right,
                    np.random.default_rng([seed, TASKS[task], 701, t]), ledger, cfg)
            s = birth(s, w)
            assert np.max(np.abs(predict(xx, before_p)-predict(xx, s['p']))) < 1e-10
        proposal_cost = 0 if proposal is None else proposal['proxy']
        remain = cfg['event_budget']-moment_cost-proposal_cost
        assert remain > 0, 'candidate work exceeded event budget'
        account = fit(s, fitx, fity, fitids, remain,
                      np.random.default_rng([seed, TASKS[task], 809, t]), ledger, cfg)
        reservoir.update(xx, yy, ids); ledger.extra['reservoir_seen'] += len(ids)
        event_counts = {k: ledger.counts[k]-previous_counts[k] for k in ledger.counts}
        assert all(not np.any(v[right:]) for v in event_counts.values()), 'future label access'
        event = {'t': t, 'arrival_range': [left, right], 'action': 'continue' if fixed or not t else 'scheduled_growth',
                 'width': len(s['p']['a']), 'parameters': nparams(s['p']), 'before_parameters': pack(before_p),
                 'after_parameters': pack(s['p']),
                 'before_optimizer': {'m': pack(before_state['m']), 'v': pack(before_state['v']), 'step': before_state['step']},
                 'proposal': proposal, 'fit': account,
                 'moment_proxy': moment_cost, 'total_event_proxy': moment_cost+proposal_cost+account['proxy'],
                 'fit_pool_ids': fitids.tolist(), 'retained_ids_after': reservoir.ids.tolist(),
                 'event_label_accesses': {k: int(v.sum()) for k, v in event_counts.items()},
                 'event_unique_labels': {k: int(np.count_nonzero(v)) for k, v in event_counts.items()},
                 'persistent_model_optimizer_words': 3*nparams(s['p']),
                 'persistent_retention_words': len(reservoir.y)*(cfg['d']+2)+(cfg['d']**2 if moment is not None else 0),
                 'current_input_words': len(yy)*(cfg['d']+1),
                 'fit_pool_workspace_words': len(fity)*(cfg['d']+2),
                 'fit_gradient_workspace_words': nparams(s['p']) + cfg['batch']*(2*len(s['p']['a'])+2),
                 'candidate_workspace_word_bound': (4*cfg['d']**2 + len(yy)*(cfg['d']+2) + 10*cfg['d']*cfg['probes']),
                 'memory_accounting_note': 'Float64/int64-equivalent array words; 6 additional controller/RNG words; eigensolver, BLAS, Python, logging/archive workspace excluded and not claimed matched.',
                 'ledger': ledger.report(), 'elapsed_seconds': time.perf_counter()-started}
        if moment is not None: event['historical_moment'] = (moment/right).tolist()
        events.append(event)
    return {'method': method, 'initial_parameters': params0, 'events': events, 'final_parameters': pack(s['p']),
            'ledger': ledger.report(), 'elapsed_seconds': time.perf_counter()-started}, ledger.counts

def evaluate(result, task, spec, ex, eq):
    def risk(packed):
        p = unpack(packed)
        if task != 'nonlinear': return float(2*np.sum((effective(p)-spec['matrix'])**2)+float(p['c'])**2+spec['noise']**2)
        return float(np.mean((predict(ex, p)-eq)**2)+spec['noise']**2)
    result['initial_risk'] = risk(result['initial_parameters']); result['final_risk'] = risk(result['final_parameters'])
    result['risk_kind'] = 'exact_gaussian_population' if task != 'nonlinear' else 'independent_input_monte_carlo_noise_integrated'
    for e in result['events']:
        e['risk_before'] = risk(e['before_parameters']); e['risk_after'] = risk(e['after_parameters'])
        if e['proposal'] is not None and task != 'nonlinear':
            p = unpack(e['before_parameters']); w = np.asarray(e['proposal']['direction'])
            residual = spec['matrix']-effective(p)
            value = float(w@residual@w); oracle = float(np.max(np.abs(np.linalg.eigvalsh(residual))))
            e['evaluator_direction_coefficient'] = value
            e['evaluator_best_rank1_coefficient_magnitude'] = oracle
            e['evaluator_direction_optimal_fraction'] = value*value/(oracle*oracle) if oracle > 1e-14 else None
    return result

def paired_checkpoint_diagnostics(seed, task, cfg, reference, x, y, spec, ex, eq):
    """Post-trajectory analysis; never enters any learner's control flow.

    Use the random_matched138 incumbent/optimizer and its retained rows for all
    alternatives, then give all alternatives identical fitting randomness/work.
    Candidate costs are additional diagnostic work, not silently matched.
    """
    rows = []
    def risk(p):
        if task != 'nonlinear': return float(2*np.sum((effective(p)-spec['matrix'])**2)+float(p['c'])**2+spec['noise']**2)
        return float(np.mean((predict(ex, p)-eq)**2)+spec['noise']**2)
    for event in reference['events'][1:]:
        t = event['t']; left, right = event['arrival_range']
        oldids = np.asarray(reference['events'][t-1]['retained_ids_after'], dtype=int)
        freshids = np.arange(left, right); old = (x[oldids], y[oldids], oldids)
        fresh = (x[left:right], y[left:right], freshids)
        pooled_ids = np.r_[oldids, freshids]; xx = x[pooled_ids]; yy = y[pooled_ids]
        p = unpack(event['before_parameters']); opt = event['before_optimizer']
        start = {'p': p, 'm': unpack(opt['m']), 'v': unpack(opt['v']), 'step': opt['step']}
        hist = ((x[:right].T*y[:right])@x[:right]-y[:right].sum()*np.eye(cfg['d']))/2
        choices = {}
        for method in ['historical_moment', 'reservoir_residual', 'fresh_residual', 'learned_probe', 'random_matched138']:
            led = Ledger(len(y)); w, meta = direction(method, start, old, fresh, hist, right,
                 np.random.default_rng([seed, TASKS[task], 701, t]), led, cfg)
            grown = birth(start, w)
            acc = fit(grown, xx, yy, pooled_ids, cfg['event_budget'],
                      np.random.default_rng([seed, TASKS[task], 809, t]), led, cfg)
            item = {'proposal': meta, 'trained_parameters': pack(grown['p']), 'risk_after_identical_refit': risk(grown['p']),
                    'fit_account': acc, 'total_diagnostic_proxy': sum(led.proxy.values())+led.extra['eigendecomposition']}
            if task != 'nonlinear':
                residual = spec['matrix']-effective(p); val = float(w@residual@w)
                oracle = float(np.max(np.abs(np.linalg.eigvalsh(residual))))
                item.update(population_optimal_output_coefficient=val,
                            population_best_gain_along_direction=2*val*val,
                            population_oracle_rank1_gain=2*oracle*oracle)
            choices[method] = item
        rows.append({'t': t, 'checkpoint_method': 'random_matched138', 'risk_before': risk(p),
                     'before_parameters': pack(p), 'old_replay_ids': oldids.tolist(),
                     'available_through_exclusive': right, 'choices': choices})
    return rows

def run_one(seed, task, cfg, outdir):
    outdir = Path(outdir).resolve(); start = time.perf_counter(); x, y, ex, eq, spec = generate(seed, task, cfg)
    data_dir = outdir.parent/'data'; data_dir.mkdir(parents=True, exist_ok=True)
    datafile = data_dir/f'{task}_{seed}.npz'
    np.savez_compressed(datafile, x=x, y=y, evaluator_x=ex, evaluator_mean=eq,
                        teacher_matrix=spec['matrix'], teacher_rotation=spec['rotation'], noise=spec['noise'])
    results = {}; counts = {}
    for method in cfg['methods']:
        r, cnt = learner(seed, task, method, cfg, x, y)
        results[method] = evaluate(r, task, spec, ex, eq)
        for name, val in cnt.items(): counts[method+'_'+name] = val
    paired = paired_checkpoint_diagnostics(seed, task, cfg, results['random_matched138'], x, y, spec, ex, eq)
    countfile = data_dir/f'{task}_{seed}_access.npz'; np.savez_compressed(countfile, **counts)
    out = {'seed': seed, 'task': task, 'config': cfg, 'methods': results, 'paired_checkpoint_diagnostics': paired,
           'runner_sha256': digest(Path(__file__).read_bytes()), 'config_sha256': jsonhash(cfg),
           'data_file': str(datafile.relative_to(ROOT)), 'data_sha256': digest(datafile.read_bytes()),
           'access_file': str(countfile.relative_to(ROOT)), 'access_sha256': digest(countfile.read_bytes()),
           'wall_seconds': time.perf_counter()-start}
    path = outdir/f'{task}_{seed}.json'; path.write_text(json.dumps(out, indent=2))
    return {'seed': seed, 'task': task, 'file': str(path.relative_to(ROOT)),
            'sha256': digest(path.read_bytes()), 'wall_seconds': out['wall_seconds'],
            'risks': {m: r['final_risk'] for m, r in results.items()}}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True)
    ap.add_argument('--output-dir', required=True); ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args(); cfg = json.loads(Path(args.config).read_text()); outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    env = {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
           'command': sys.argv, 'code_sha256': digest(Path(__file__).read_bytes()), 'config_sha256': jsonhash(cfg),
           'threads': {k: os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']}}
    (outdir/'environment.json').write_text(json.dumps(env, indent=2)); rows = []; start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs = [executor.submit(run_one, s, t, cfg, outdir) for t in cfg['tasks'] for s in cfg['seeds']]
        for job in as_completed(jobs):
            row = job.result(); rows.append(row); print(json.dumps(row), flush=True)
            (outdir/'run_index.json').write_text(json.dumps({'runs': rows, 'completed': len(rows),
                 'planned': len(jobs), 'wall_seconds': time.perf_counter()-start}, indent=2))
    print('COMPLETE', len(rows), time.perf_counter()-start, flush=True)

if __name__ == '__main__': main()

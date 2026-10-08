"""Tiny array-lifetime checks; does not edit the runner or run experiments.

Run with a NumPy-enabled Python and PYTHONDONTWRITEBYTECODE=1.
Only explicit, disjoint NumPy buffers are counted; this is a lower bound.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import weakref


def root_array(array, np):
    while isinstance(array.base, np.ndarray):
        array = array.base
    return array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frontier', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'code/frontier.py')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.frontier.resolve().parent))
    spec = importlib.util.spec_from_file_location('frontier_microcheck', args.frontier)
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    np = f.np
    cfg = f.default_config()
    real_concatenate = np.concatenate
    transitions = []
    allocation_lifetimes = []
    previous_fitx = None

    def concatenate(*arguments, **keywords):
        nonlocal previous_fitx
        frame = inspect.currentframe().f_back
        is_fitx = (frame.f_code.co_name == 'learner'
                   and arguments[0][0].ndim == 2)
        if is_fitx:
            allocation_lifetimes.append({
                'event': frame.f_locals['event'],
                'previous_fitx_alive_before_allocation': (
                    previous_fitx is not None and previous_fitx() is not None),
            })
        result = real_concatenate(*arguments, **keywords)
        if is_fitx:
            previous_fitx = weakref.ref(result)
        if (frame.f_code.co_name == 'learner' and result.ndim == 2
                and 'fitx' in frame.f_locals):
            arrays = {name: frame.f_locals[name]
                      for name in ('fitx', 'fity', 'fitids', 'ids')}
            arrays['new_fitx'] = result
            transitions.append({
                'event': frame.f_locals['event'],
                'known_live_unique_array_bytes': sum(a.nbytes for a in arrays.values()),
                'components': {key: value.nbytes for key, value in arrays.items()},
                'pairwise_disjoint': all(
                    not np.shares_memory(a, b)
                    for i, a in enumerate(arrays.values())
                    for j, b in enumerate(arrays.values()) if i < j),
            })
        return result

    x = np.zeros((cfg['arrivals'] * cfg['block'], cfg['d']))
    y = np.zeros(len(x))
    np.concatenate = concatenate
    try:
        trajectory = f.learner(0, 'null', 'random_growth138', 'global',
                               100000, cfg, x, y)
    finally:
        np.concatenate = real_concatenate
    for transition in transitions:
        memory = trajectory['events'][transition['event']]['memory']
        reported = memory.get('working_array_bytes_upper_estimate',
                              memory.get('working_array_bytes_estimate'))
        transition['reported_working_array_estimate'] = reported
        transition['difference'] = transition['known_live_unique_array_bytes'] - reported

    gradient_snapshots = []

    def trace(frame, event, argument):
        if frame.f_code is f.grad.__code__ and event == 'line' and 'e' in frame.f_locals:
            fit_frame = frame.f_back
            buffers = {}
            names = {}
            arrays = {name: frame.f_locals[name] for name in ('x', 'y', 'z', 'q', 'e')}
            arrays.update({name: fit_frame.f_locals[name] for name in ('order', 'ix')})
            arrays.update({f'piece_{i}': value
                           for i, value in enumerate(fit_frame.f_locals['pieces'])})
            for name, array in arrays.items():
                root = root_array(array, np)
                buffers[id(root)] = root.nbytes
                names.setdefault(id(root), []).append(name)
            gradient_snapshots.append({
                'known_live_unique_array_bytes': sum(buffers.values()),
                'buffers': [{'names': names[key], 'nbytes': value}
                            for key, value in buffers.items()],
            })
        return trace

    pool, width = 1162, 2
    state = f.new_state(0, 'null', cfg['d'], width, 'global')
    fitx, fity, ids = np.zeros((pool, cfg['d'])), np.zeros(pool), np.arange(pool)
    replay = f.Replay(cfg['replay_capacity'], cfg['d'], np.random.default_rng(0))
    reported_fit = f.array_inventory(state, replay, None, fitx, fity, ids,
                                    cfg['block'], cfg['batch'], None)['working_fit_array_byte_estimate']
    sys.settrace(trace)
    try:
        fit_account, _ = f.fit(state, fitx, fity, ids, 400000,
                               np.random.default_rng(0), cfg, 'global')
    finally:
        sys.settrace(None)
    maximum_fit = max(gradient_snapshots,
                      key=lambda item: item['known_live_unique_array_bytes'])
    maximum_fit['reported_working_fit_array_byte_estimate'] = reported_fit
    maximum_fit['difference'] = maximum_fit['known_live_unique_array_bytes'] - reported_fit
    output = {
        'frontier': str(args.frontier.resolve()),
        'frontier_sha256': hashlib.sha256(args.frontier.read_bytes()).hexdigest(),
        'note': 'Lower bounds exclude all persistent/model/audit arrays and vendor workspaces.',
        'pool_transition': transitions,
        'fitx_allocation_lifetimes': allocation_lifetimes,
        'cross_epoch_gradient_snapshot': maximum_fit,
        'fit_account': fit_account,
    }
    encoded = json.dumps(output, indent=2) + '\n'
    if args.output:
        args.output.write_text(encoded)
    print(encoded)


if __name__ == '__main__':
    main()

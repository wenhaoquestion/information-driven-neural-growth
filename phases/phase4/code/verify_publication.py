"""Validate delivered historical summaries without imports, RNG, or training."""
from pathlib import Path
from collections import Counter
import ast
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]

def csvrows(name):
    with (ROOT / name).open(newline='') as f:
        return list(csv.DictReader(f))

def main():
    runs = csvrows('experiments/results/run_summary.csv')
    events = csvrows('experiments/results/events.csv')
    pairs = csvrows('experiments/results/paired_contrasts.csv')
    config = json.loads((ROOT/'experiments/config.json').read_text())
    assert len(runs) == 336 and len(events) == 8064 and len(pairs) == 128
    lookup = {(r['task'], int(r['seed']), r['method']): r for r in runs}
    assert len(lookup) == 336
    expected = {(t, s, m) for t in config['tasks'] for s in config['seeds'] for m in config['methods']}
    assert set(lookup) == expected
    counts = Counter((e['task'], int(e['seed']), e['method']) for e in events)
    assert set(counts) == expected and set(counts.values()) == {24}
    for e in events:
        if int(e['event']) == 24:
            r = lookup[e['task'], int(e['seed']), e['method']]
            assert float(e['retained_risk']) == float(r['final_risk'])
            assert int(e['width']) == int(r['final_width'])
    weights = {
      'release_residual': [('released_residual',1),('withheld_residual',-1)],
      'release_random': [('released_random',1),('withheld_random',-1)],
      'residual_minus_random_released': [('released_residual',1),('released_random',-1)],
      'released_residual_minus_fixed2': [('released_residual',1),('fixed2_online',-1)],
      'released_residual_minus_fixed26': [('released_residual',1),('fixed26_online',-1)],
      'released_residual_minus_fixed26_gated': [('released_residual',1),('fixed26_gated',-1)],
      'fixed26_gated_minus_online': [('fixed26_gated',1),('fixed26_online',-1)],
      'release_by_proposal_interaction': [('released_residual',1),('withheld_residual',-1),('released_random',-1),('withheld_random',1)]}
    maxerr = 0.0
    for p in pairs:
        values = [sum(w * float(lookup[p['task'], s, m][p['metric']]) for m,w in weights[p['contrast']]) for s in config['seeds']]
        mean = sum(values)/len(values)
        sd = math.sqrt(sum((x-mean)**2 for x in values)/(len(values)-1))
        se = sd/math.sqrt(len(values))
        maxerr=max(maxerr,abs(mean-float(p['mean'])),abs(sd-float(p['sd'])),abs(se-float(p['se'])))
        assert maxerr < 1e-12
        assert abs((float(p['ci95_high'])+float(p['ci95_low']))/2-mean)<1e-12
    manifest = json.loads((ROOT/'RAW_ARTIFACTS.json').read_text())
    rawchecked=0
    for row in manifest['files']:
        if row['included']:
            f=ROOT/row['path']
            assert f.stat().st_size==row['bytes']
            assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256']
            rawchecked+=1
    freeze=json.loads((ROOT/'experiments/protocol_frozen.json').read_text())
    for rel,h in freeze['files'].items():
        assert hashlib.sha256((ROOT/'experiments'/rel).read_bytes()).hexdigest()==h
    base=json.loads((ROOT/'results/retention_summary.json').read_text())
    extra=json.loads((ROOT/'results/retention_supplement_summary.json').read_text())
    assert len(base)==153 and len(extra)==96
    for row in base+extra:
        assert row['replicates']==1000
        assert row['rate']==row['accepted']/row['replicates']
        assert 0<=row['accepted']<=1000
    for p in ROOT.rglob('*.py'):
        ast.parse(p.read_text(),filename=str(p.relative_to(ROOT)))
    report=dict(status='passed',scope='Static syntax, saved summary arithmetic and included-file hashes; no experiment, RNG, or training executed.',
      runs=len(runs),events=len(events),paired_contrasts=len(pairs),maximum_mean_sd_se_error=maxerr,
      retention_groups=len(base),post_hoc_groups=len(extra),raw_hashes_checked=rawchecked,
      original_frozen_files_checked=len(freeze['files']))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()

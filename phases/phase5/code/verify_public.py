"""Read-only public-record consistency check: standard library, no RNG/training."""
from pathlib import Path
from collections import defaultdict
import ast
import csv
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]

def read_csv(relative):
    with (ROOT/relative).open(newline='') as stream:
        return list(csv.DictReader(stream))

def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=2e-13),(a,b)

def mean(values):
    values=list(values)
    return math.fsum(values)/len(values)

manifest=json.loads((ROOT/'PUBLICATION_MANIFEST.json').read_text())
for row in manifest['files']:
    path=ROOT/row['path']
    assert path.is_file(),row['path']
    assert path.stat().st_size==row['bytes'],row['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],row['path']
python_count=0
for path in ROOT.rglob('*.py'):
    ast.parse(path.read_text(),filename=str(path.relative_to(ROOT)))
    python_count+=1
for path in ROOT.rglob('*.json'):
    json.loads(path.read_text())
report=dict(status='PASS',scope='Included public files and deterministic saved-table consistency only; no raw-data replay, RNG, bootstrap or training',
            release_files=len(manifest['files']),python_sources_parsed=python_count)

endpoints=read_csv('results/published_seed_endpoints.csv')
assert len(endpoints)==768
assert len({(r['study'],r['task'],r['seed'],r['method']) for r in endpoints})==768
assert sum(int(r['events']) for r in endpoints)==5376
for study,output,expected in [('confirmation','quadratic',336),('adaptive_confirmation','adaptive',240),('fullbatch_confirmation','fullbatch',192)]:
    group=[r for r in endpoints if r['study']==study]
    assert len(group)==expected
    for row in read_csv('results/'+output+'/method_summary.csv'):
        values=[float(r['final_risk']) for r in group if r['task']==row['task'] and r['method']==row['method']]
        assert len(values)==12
        close(mean(values),row['mean'])
    for row in read_csv('results/'+output+'/paired_contrasts.csv'):
        by={(r['seed'],r['method']):float(r['final_risk']) for r in group if r['task']==row['task']}
        seeds=sorted({k[0] for k in by})
        close(mean(by[s,row['first']]-by[s,row['second']] for s in seeds),row['mean'])
branches=read_csv('results/published_checkpoint_branches.csv')
assert len(branches)==1440
assert len({(r['task'],r['seed'],r['event'],r['method']) for r in branches})==1440
for freeze in (ROOT/'experiments').glob('*frozen.json'):
    for relative,expected in json.loads(freeze.read_text())['files'].items():
        path=ROOT/relative.removeprefix('phase5/')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,relative
original_moment=json.loads((ROOT/'results/moments/execution.json').read_text())
assert hashlib.sha256((ROOT/'results/moments/runner_snapshot.py').read_bytes()).hexdigest()==original_moment['source_sha256']
stages=[json.loads(line) for line in (ROOT/'results/moments/raw.jsonl').read_text().splitlines()]
assert len(stages)==1024
assert len({(r['sigma'],r['seed']) for r in stages})==128
for row in json.loads((ROOT/'results/moments/summary.json').read_text()):
    close(mean(float(r[row['method']]) for r in stages if r['sigma']==row['sigma'] and r['stage']==row['stage']),row['mean'])
followup=read_csv('results/fullbatch/paired_contrasts.csv')
assert len(followup)==16 and all(float(r['ci95_low'])<=0<=float(r['ci95_high']) for r in followup)
for name in ['empirical_audit.json','adaptive_empirical_audit.json','fullbatch_audit.json','empirical_replay_check.json']:
    assert not json.loads((ROOT/'reviews'/name).read_text()).get('errors')
report.update(neural_endpoints=768,neural_events=5376,checkpoint_branches=1440,estimator_stages=1024,
              fullbatch_intervals_containing_zero=16)
print(json.dumps(report,indent=2))

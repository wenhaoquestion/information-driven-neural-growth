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

records=read_csv('analysis/confirmation/all_seed_results.csv')
assert len(records)==7040
keys=['task','seed','age_mode','method','event_budget']
key=lambda r:tuple(r[k] for k in keys)
assert len({key(r) for r in records})==7040
costs=defaultdict(int)
for row in read_csv('analysis/confirmation/all_seed_costs.csv'):
    costs[key(row)]+=int(row['proxy_work'])
groups=defaultdict(list)
for row in records:
    assert costs[key(row)]==int(row['actual_total_work'])
    assert int(row['actual_total_work'])<=int(row['allocated_total_work'])
    close(float(row['final_risk'])-float(row['noise_variance']),row['final_excess_risk'])
    assert int(row['reaches_epsilon_001'])==int(float(row['final_excess_risk'])<=.01)
    groups[(row['task']+'_'+row['age_mode'],row['seed'],row['method'])].append(row)
computed={}
for k,rows in groups.items():
    rows.sort(key=lambda r:int(r['event_budget']))
    assert [int(r['event_budget']) for r in rows]==[3200000,5000000,6400000,8000000,10000000]
    x=[math.log(int(r['allocated_total_work'])) for r in rows]
    y=[float(r['final_excess_risk']) for r in rows]
    computed[k]=math.fsum((x[j+1]-x[j])*(y[j]+y[j+1])/2 for j in range(4))/(x[-1]-x[0])
for row in read_csv('analysis/confirmation/seed_auc.csv'):
    close(computed[row['group'],row['seed'],row['method']],row['auc'])
for row in read_csv('analysis/confirmation/auc_summary.csv'):
    close(mean(v for (g,s,m),v in computed.items() if g==row['group'] and m==row['method']),row['mean'])
for row in read_csv('analysis/confirmation/contrasts.csv'):
    seeds=sorted({s for g,s,m in computed if g==row['group']})
    close(mean(computed[row['group'],s,'historical_moment']-computed[row['group'],s,row['comparator']] for s in seeds),row['mean'])
primary='indefinite_noisy_global'
seeds=sorted({s for g,s,m in computed if g==primary})
assert len(seeds)==256
wins=sum(computed[primary,s,'historical_moment']<computed[primary,s,'fixed7'] for s in seeds)
assert wins==118
summary=json.loads((ROOT/'analysis/confirmation/analysis_summary.json').read_text())
frozen_runner=json.loads((ROOT/'experiments/confirmation_primary/manifest.json').read_text())['code_sha256']
for name,expected in frozen_runner.items():
    assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==expected,name
assert summary['primary']['n']==256 and summary['primary']['success'] is True
close(summary['primary']['iut_p'],0.0031173493423169085)
nonattainers=[r for r in read_csv('analysis/confirmation/target_seeds.csv') if r['group']==primary and r['method']=='fixed7' and r['attained']=='False']
assert len(nonattainers)==8
report.update(confirmation_sensitivity_trajectories=7040,reconstructed_auc_rows=len(computed),
              independent_primary_seeds=256,historical_wins_against_fixed7=wins,fixed7_target_nonattainers=8,
              archived_primary_iut_p=summary['primary']['iut_p'])
print(json.dumps(report,indent=2))

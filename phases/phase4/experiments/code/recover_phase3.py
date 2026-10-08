"""Read-only original-result recovery; all new output is kept in phase4."""
from pathlib import Path
import hashlib, json, os, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'phase3/code'))
from betting_growth import run
OUT=ROOT/'phase4/experiments/results'

def compare(a,b,path='run'):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,set(a)^set(b))
        for k in a:
            if k not in ['elapsed_seconds','config']:compare(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}[{i}]')
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        assert np.isclose(a,b,rtol=1e-9,atol=1e-11),(path,a,b)
    else:assert a==b,(path,a,b)

def main():
    source=ROOT/'phase3/results/betting_followup4096.json'
    data=json.loads(source.read_text())
    old=next(r for r in data['runs'] if r['task']=='interaction' and r['seed']==63000)
    begin=time.perf_counter();new=run(old['seed'],old['task'],data['config'])
    compare(old,new)
    (OUT/'phase3_recovered_run.json').write_text(json.dumps(new,indent=2))
    result=dict(status='matched',source=str(source.relative_to(ROOT)),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        original_runner_sha256=hashlib.sha256((ROOT/'phase3/code/betting_growth.py').read_bytes()).hexdigest(),
        task=old['task'],seed=old['seed'],methods=list(new['methods']),
        scope='One entire original interaction trajectory, all methods, branches, tails and controls; numerical tolerance rtol=1e-9 atol=1e-11, timings excluded.',
        wall_seconds=time.perf_counter()-begin)
    (OUT/'phase3_recovery.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()

"""Replay one full seven-method confirmation pair into a separate directory."""
from pathlib import Path
import argparse,json,time
import numpy as np
import recycling as r

def compare(a,b,path='methods'):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,set(a)^set(b))
        for k in a:
            if k!='elapsed_seconds':compare(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}[{i}]')
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        assert np.isclose(a,b,rtol=1e-10,atol=1e-12),(path,a,b)
    else:assert a==b,(path,a,b)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',default='interaction');ap.add_argument('--seed',type=int,default=750000)
    ap.add_argument('--output-dir',default=str(r.BASE/'replay/results'));args=ap.parse_args()
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    assert not (out/f'{args.task}_{args.seed}.json').exists(),'Use a new output directory to preserve earlier replays.'
    original=r.BASE/'results/confirmation'/f'{args.task}_{args.seed}.json';old=json.loads(original.read_text())
    begin=time.perf_counter();r.run_one(args.seed,args.task,old['config'],out)
    new=json.loads((out/f'{args.task}_{args.seed}.json').read_text());compare(old['methods'],new['methods'])
    report=dict(status='matched',task=args.task,seed=args.seed,methods=list(old['methods']),events=old['config']['events']*len(old['methods']),
        source_sha256=r.sha(original.read_bytes()),runner_sha256=r.sha(Path(r.__file__).read_bytes()),
        tolerance='rtol=1e-10 atol=1e-12; elapsed time excluded',wall_seconds=time.perf_counter()-begin)
    (out/'replay_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()

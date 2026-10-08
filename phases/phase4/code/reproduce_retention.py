"""Replay the unchanged executed scripts in a new directory and compare data.

The destination must not already exist. Example from the project root:
python phases/phase4/code/reproduce_retention.py --output-dir phase4/replays/retention
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);args=p.parse_args()
    source=Path(__file__).resolve().parents[1]
    dest=Path(args.output_dir).resolve()
    if dest.exists():raise SystemExit('Refusing to overwrite an existing replay directory: '+str(dest))
    (dest/'code').mkdir(parents=True)
    (dest/'research').mkdir();(dest/'tmp').mkdir();(dest/'figures').mkdir();(dest/'results').mkdir()
    scripts=['retention_experiment.py','retention_supplement.py','plot_retention.py','audit_retention.py']
    for name in scripts:shutil.copy2(source/'code'/name,dest/'code'/name)
    for name in ['RETENTION_EXPERIMENT_PROTOCOL.md','RETENTION_PROTOCOL_ADDENDUM.md']:
        shutil.copy2(source/'research'/name,dest/'research'/name)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',
                                    PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(dest/'tmp/mpl'))
    for name in scripts:
        with (dest/'results'/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(dest/'code'/name)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    names=['retention_raw.jsonl','retention_summary.json','retention_supplement_raw.jsonl',
           'retention_supplement_summary.json']
    checks={}
    for name in names:
        original=source/'results'/name
        if original.exists():
            a=hashlib.sha256(original.read_bytes()).hexdigest()
        else:
            manifest=json.loads((source/'RAW_ARTIFACTS.json').read_text())
            a=next(row['sha256'] for row in manifest['files'] if row['path']=='results/'+name)
        b=hashlib.sha256((dest/'results'/name).read_bytes()).hexdigest()
        checks[name]={'source_sha256':a,'replay_sha256':b,'identical':a==b}
        assert a==b,name
    import numpy as np
    with np.load(source/'results/retention_counts.npz') as a,np.load(dest/'results/retention_counts.npz') as b:
        assert a.files==b.files
        assert all(np.array_equal(a[name],b[name]) for name in a.files)
        count=len(a.files)
    report={'checks':checks,'raw_count_arrays_identical':count,'timing_fields_compared':False,
            'scope':'Full seeded replay of R1/R2 and post-hoc supplementary methods; scripts copied byte-for-byte.'}
    (dest/'replay_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()

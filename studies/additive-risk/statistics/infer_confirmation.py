"""Inference module; reads counts/public inputs, never simulator reference."""
import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent/'theory'))
from honest_delta import (Interval, PARTITIONS, count_rectangle, infer_delta,
                          report_error_bounds)
from local_loss_oracle import Interior2048Loss


def encode(x):
    if isinstance(x,Q):
        return {'rational':str(x), 'decimal':float(x)}
    if isinstance(x,Interval):
        return {'lo':encode(x.lo), 'hi':encode(x.hi)}
    if isinstance(x,dict):
        return {str(k):encode(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):
        return [encode(v) for v in x]
    return x


def run_inference(public_counts):
    allowed = {'candidate','counts','public_p','m'}
    if set(public_counts) != allowed:
        raise ValueError('Only candidate/counts/public_p/m are allowed inference inputs')
    if public_counts['candidate'] != 'preselected interior_v1 n=2048':
        raise ValueError('Only the preregistered candidate is supported')
    counts = tuple(tuple(v) for v in public_counts['counts'])
    p = tuple(Q(v) for v in public_counts['public_p'])
    if sum(n for n,k in counts) != public_counts['m'] or public_counts['m'] > 10**6:
        raise ValueError('Inconsistent or excessive sample count')
    if p != (Q(1,4),)*4:
        raise ValueError('Public p differs from preregistration')
    loss = Interior2048Loss()
    result = infer_delta(counts,p,loss)
    rectangle,center = count_rectangle(counts)
    reports = []
    for part in PARTITIONS:
        fitted = tuple(sum(p[i]*center[i] for i in b)/sum(p[i] for i in b) for b in part)
        reports.append({'partition':part, 'sample_fitted_reports':fitted,
                        'population_report_error_interval':
                            report_error_bounds(part,fitted,rectangle,p,loss)})
    return {'candidate':public_counts['candidate'], 'm':public_counts['m'],
            'counts':counts, 'public_p':p, 'alpha':Q(1,20),
            'null_threshold':Q(1,2000), 'delta_95_interval':result['delta'],
            'posterior_simultaneous_95_rectangle':rectangle,
            'reject_null':result['reject_null'],
            'hierarchy_95_bounds':result['hierarchy_bounds'],
            'all_15_partition_report_error_bounds':reports,
            'visible_input_contract':'counts/public_p/candidate/m only; no reference posterior'}


def run_files(counts_path,output_path):
    counts_path,output_path=Path(counts_path),Path(output_path)
    started=datetime.now(timezone.utc).isoformat()
    clock=time.perf_counter()
    result=run_inference(json.loads(counts_path.read_text()))
    result['runtime']={'started_utc':started,
                       'finished_utc':datetime.now(timezone.utc).isoformat(),
                       'wall_seconds':time.perf_counter()-clock,
                       'python':platform.python_version(),
                       'executable':sys.executable,
                       'peak_memory_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*
                                               (1 if sys.platform=='darwin' else 1024))}
    paths=[Path(__file__).resolve(),HERE/'honest_delta.py',HERE.parent/'theory'/'local_loss_oracle.py']
    result['source_sha256']={str(x.relative_to(HERE.parent)):
                             hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
    result['counts_sha256']=hashlib.sha256(counts_path.read_bytes()).hexdigest()
    output_path.write_text(json.dumps(encode(result),indent=2)+'\n')
    print(json.dumps({'reject_null':result['reject_null'],
                       'delta_95_interval':[float(result['delta_95_interval'].lo),
                                            float(result['delta_95_interval'].hi)]}))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--counts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run_files(args.counts,args.output)


if __name__=='__main__':
    main()

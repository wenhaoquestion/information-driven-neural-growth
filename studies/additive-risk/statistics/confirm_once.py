"""Prepared one-shot confirmation.  Do not execute before the recorded review gate.

The simulator reads the reference; the inference module receives only public
count-file paths, never theta. One process is used, as frozen in the protocol.
An exclusive lock prevents reruns.
"""
import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute-authorized',action='store_true',
                        help='Use only after independent review and the recorded study gate')
    args=parser.parse_args()
    if not args.execute_authorized:
        parser.error('Prepared only. Explicit execution authorization has not been supplied.')
    import numpy as np
    reference_path=HERE/'CONFIRMATION_REFERENCE.json'
    ref=json.loads(reference_path.read_text())
    if ref['seed']!=2026100723 or ref['sample_budget']!=10**6:
        raise ValueError('Frozen seed/sample budget changed')
    if ref['state_probabilities']!=['1/4']*4 or ref['reference_posterior']!=['1/8','3/8','5/8','7/8']:
        raise ValueError('Frozen reference changed')
    certificate=json.loads((HERE/'INTERIOR_2048_POWER_CERTIFICATE.json').read_text())
    if not certificate['certified']:
        raise ValueError('Analytic power gate is not certified')
    for relative,digest in certificate['source_sha256'].items():
        if hashlib.sha256((HERE.parent/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError('Analytic certificate source hash mismatch: '+relative)
    destination=HERE/'confirmation'
    destination.mkdir(exist_ok=True)
    lock=destination/'ONE_SHOT_STARTED.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    started=datetime.now(timezone.utc).isoformat()
    with os.fdopen(fd,'w') as f:
        f.write('One confirmation attempt began at '+started+'. Never remove to retry.\n')
    clock=time.perf_counter()
    source_paths=[Path(__file__).resolve(),reference_path,HERE/'infer_confirmation.py',
                  HERE/'honest_delta.py',HERE/'certify_interior2048.py',
                  HERE.parent/'theory'/'local_loss_oracle.py',
                  HERE/'INTERIOR_2048_POWER_CERTIFICATE.json']
    hashes={str(x.relative_to(HERE.parent)):hashlib.sha256(x.read_bytes()).hexdigest()
            for x in source_paths}
    rng=np.random.default_rng(ref['seed'])
    n=rng.multinomial(ref['sample_budget'],[float(Q(x)) for x in ref['state_probabilities']])
    k=[int(rng.binomial(int(ni),float(Q(theta))))
       for ni,theta in zip(n,ref['reference_posterior'])]
    public={'candidate':ref['candidate'],
            'counts':[[int(ni),ki] for ni,ki in zip(n,k)],
            'public_p':ref['state_probabilities'], 'm':ref['sample_budget']}
    counts_path=destination/'COUNTS_PUBLIC.json'
    counts_path.write_text(json.dumps(public,indent=2)+'\n')
    from infer_confirmation import run_files
    run_files(counts_path,destination/'INFERENCE.json')
    rss_self=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    runtime={'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),
             'wall_seconds':time.perf_counter()-clock,'python':platform.python_version(),
             'executable':sys.executable,'numpy':np.__version__,
             'bit_generator':type(rng.bit_generator).__name__,
             'seed':ref['seed'],'sample_budget':ref['sample_budget'],
             'peak_memory_bytes':int(rss_self*(1 if sys.platform=='darwin' else 1024)),
             'peak_memory_scope':'single process ru_maxrss',
             'single_process':True,
             'source_sha256':hashes,
             'interpretation':'One confirmation realization, not an estimate of power. Power is analytic.'}
    (destination/'RUNTIME.json').write_text(json.dumps(runtime,indent=2)+'\n')
    (destination/'COMPLETED.json').write_text(json.dumps({'completed':True,
        'finished_utc':runtime['finished_utc'],'counts_sha256':hashlib.sha256(counts_path.read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps({'completed':True,'output_directory':str(destination),
                      'wall_seconds':runtime['wall_seconds'],
                      'peak_memory_bytes':runtime['peak_memory_bytes']}))


if __name__=='__main__':
    main()

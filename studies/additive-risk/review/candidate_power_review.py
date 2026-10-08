"""Read-only, deterministic pre-label review of candidate-specific power."""
from fractions import Fraction as Q
from pathlib import Path
from itertools import combinations, product
import hashlib
import json
import signal
import sys
import time

signal.alarm(120)
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PILOT=HERE.parent
sys.path.insert(0,str(PILOT/'statistics'))
sys.path.insert(0,str(PILOT/'theory'))
from honest_delta import certify_power, infer_delta
from local_loss_oracle import Interior2048Loss

start=time.monotonic()
loss=Interior2048Loss()
theta=(Q(1,8),Q(3,8),Q(5,8),Q(7,8));p=(Q(1,4),)*4
out=certify_power(theta,p,loss,entropy_width=loss.entropy_width,slope_width=loss.slope_width)
assert out['certified'] and out['power_lower_if_certified']==Q(4,5)
assert out['inference_lower_bound_on_power_event']>Q(1,2000)
saved=json.loads((PILOT/'statistics/INTERIOR_2048_POWER_CERTIFICATE.json').read_text())
assert out['inference_lower_bound_on_power_event']==Q(saved['inference_lower_bound_on_power_event']['rational'])
for rel,digest in saved['source_sha256'].items():
    assert hashlib.sha256((PILOT/rel).read_bytes()).hexdigest()==digest

env=out['envelope'];rect=env['rectangle'];cellchecks=[]
for size in range(1,5):
    for block in combinations(range(4),size):
        lo=sum(rect[i].lo for i in block)/size
        hi=sum(rect[i].hi for i in block)/size
        argslo=loss._args(lo)[1];argshi=loss._args(hi)[1]
        unions=[(min(a[0],b[0]),max(a[1],b[1])) for a,b in zip(argslo,argshi)]
        assert all(loss._safe(u) for u in unions)
        for q in (lo,(lo+hi)/2,hi):
            h,s=loss.entropy(q),loss.slope(q)
            assert h.hi-h.lo<=loss.entropy_width
            assert s.hi-s.lo<=loss.slope_width
        cellchecks.append({'block':list(block),'whole_mean_interval_safe':True})

# Deterministic extreme integer count inputs; not label sampling.
n=250000
choices=[]
for r in env['estimator_ranges']:
    low=n*r.lo;high=n*r.hi
    choices.append((low.numerator//low.denominator+1,high.numerator//high.denominator))
lower=[]
for counts in product(*choices):
    result=infer_delta([(n,k) for k in counts],p,loss)
    assert result['reject_null']
    assert result['delta'].lo>=out['inference_lower_bound_on_power_event']
    lower.append(result['delta'].lo)

fallback=[]
for n,k in ((0,0),(1000,0),(1000,1000),(1000,500),(1000,360),(1000,640)):
    # All four empirical posteriors coincide. A constant posterior in this
    # rectangle has exactly Delta=0, so any valid enclosure must include zero.
    result=infer_delta([(n,k)]*4,p,loss)
    assert result['delta'].lo<=0<=result['delta'].hi
    assert not result['reject_null']
    fallback.append({'N':n,'K':k,'contains_zero_and_does_not_reject':True})

report={'candidate':'preselected interior_v1 n2048','sampled_labels':0,
 'power_lower_bound':str(out['power_lower_if_certified']),
 'actual_inference_lower_on_power_event_exact':str(out['inference_lower_bound_on_power_event']),
 'actual_inference_lower_on_power_event_decimal':float(out['inference_lower_bound_on_power_event']),
 'saved_power_certificate_exactly_reproduced':True,'saved_source_hashes_verified':True,
 'all15_whole_mean_interval_checks':cellchecks,
 'moving_center_count_cases':len(lower),'minimum_checked_actual_lower':float(min(lower)),
 'fallback_checks':fallback,'all_assertions_passed':True,
 'elapsed_seconds':time.monotonic()-start,
 'reviewed_hashes':{rel:hashlib.sha256((PILOT/rel).read_bytes()).hexdigest()
  for rel in ['statistics/honest_delta.py','statistics/DERIVATION.md',
              'statistics/INTERIOR_2048_POWER_CERTIFICATE.json',
              'theory/local_loss_oracle.py','theory/LOCAL_GRADIENT_SUPPORT.md']}}
(HERE/'candidate_power_review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in
 ['actual_inference_lower_on_power_event_exact','all15_whole_mean_interval_checks','reviewed_hashes']},indent=2))

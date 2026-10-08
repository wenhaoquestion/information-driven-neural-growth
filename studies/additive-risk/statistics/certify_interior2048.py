"""Recompute pre-label power certificate; no random numbers or sampling."""
import hashlib
import json
import sys
from fractions import Fraction as Q
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent / 'theory'))
from honest_delta import Interval, certify_power
from local_loss_oracle import Interior2048Loss


def encode(x):
    if isinstance(x, Q):
        return {'rational':str(x), 'decimal':float(x)}
    if isinstance(x, Interval):
        return {'lo':encode(x.lo), 'hi':encode(x.hi)}
    if isinstance(x, dict):
        return {str(k):encode(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)):
        return [encode(v) for v in x]
    return x


def main():
    loss = Interior2048Loss()
    result = certify_power((Q(1,8),Q(3,8),Q(5,8),Q(7,8)), (Q(1,4),)*4,
                           loss, m=10**6, entropy_width=loss.entropy_width,
                           slope_width=loss.slope_width)
    inputs = [HERE/'honest_delta.py', HERE/'certify_interior2048.py',
              HERE.parent/'theory'/'local_loss_oracle.py']
    result['source_sha256'] = {str(x.relative_to(HERE.parent)):
                               hashlib.sha256(x.read_bytes()).hexdigest() for x in inputs}
    result['sampling_performed'] = False
    result['candidate'] = 'preselected interior_v1 n=2048'
    for name,data in [('INTERIOR_2048_POWER_ENVELOPE.json',result['envelope']),
                      ('INTERIOR_2048_POWER_CERTIFICATE.json',result)]:
        (HERE/name).write_text(json.dumps(encode(data),indent=2)+'\n')
    print(json.dumps({'certified':result['certified'],
          'minimum_inference_lower_bound_on_power_event':
              float(result['inference_lower_bound_on_power_event']),
          'power_lower_bound':float(result['power_lower_if_certified']),
          'sample_budget':10**6, 'sampling_performed':False},indent=2))


if __name__ == '__main__':
    main()

"""Finite-distribution exact checks of the bounded-mean betting gate."""
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom
from betting_growth import betting_comparison

records=[]
for tau in [0.,.0005]:
    for lo,hi in [(-1.,1.),(-1.,.2),(-.1,.1)]:
        prob=(tau-lo)/(hi-lo)
        for n in [12,60,200]:
            lam=np.array([.05,.1,.2,.5,1/(1+tau)])
            counts=np.arange(n+1)
            factors_lo=1+lam*(lo-tau);factors_hi=1+lam*(hi-tau)
            assert np.min(factors_lo)>=-1e-14
            logs=np.empty((n+1,len(lam)))
            for j in range(len(lam)):
                for k in counts:
                    logs[k,j]=(k*math.log(factors_hi[j]) if k else 0)+( (n-k)*math.log(factors_lo[j]) if n>k and factors_lo[j]>0 else (-math.inf if n>k else 0))
            pmf=binom.pmf(counts,n,prob)
            log_e=logsumexp(logs,axis=1)-math.log(len(lam))
            expected=float(np.sum(pmf*np.exp(log_e)))
            assert abs(expected-1)<2e-12,expected
            alpha=.05
            error=float(np.sum(pmf[log_e>=-math.log(alpha)]))
            assert error<=alpha+1e-13,error
            records.append(dict(tau=tau,lo=lo,hi=hi,n=n,expected_e=expected,false_accept=error))
# Test numerical endpoint factors through actual loss arrays in [0,1].
for tau in [0.,.0005]:
    r=betting_comparison(np.array([0.,1.,.4]),np.array([1.,0.,.3]),.05,tau)
    assert np.isfinite(r['log_e'])
result={'evidence_type':'exact finite-binomial expectation and probability sums, not Monte Carlo',
        'max_expectation_error':max(abs(r['expected_e']-1) for r in records),'records':records}
path=Path(__file__).resolve().parents[1]/'results/betting_gate_checks.json'
path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'max_expectation_error':result['max_expectation_error'],'checks':len(records)}))

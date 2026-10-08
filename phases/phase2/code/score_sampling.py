"""Finite iid observation tests of a raw local split score on noisy parity.
Multinomial counts generate exactly the iid sufficient-statistic distribution;
no importance weights or training on population labels are used.
The comparator is given the true parity feature (oracle-informed dictionary).
"""
from pathlib import Path
import json,math
import numpy as np
import mpmath as mp
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1]

def main():
    rng=np.random.default_rng(731090); d=6; eta=.15; b=.5; reps=1000
    sign=float(mp.sign(mp.diff(mp.tanh,mp.mpf('.5'),d)))
    j=np.arange(d+1); V=(2*j-d)/np.sqrt(d); parity=sign*(-1.)**(d-j)
    source=np.array([math.comb(d,int(k))/2**d for k in j]); q=.5+eta*parity
    joint=np.stack([source*(1-q),source*q],axis=1).ravel()
    Y=np.tile([0,1],d+1); par=np.repeat(parity,2)
    rows=[]
    for t in [.1,.2,.4,.8]:
        H=np.tanh(b+t*V)-np.tanh(b)-t*(1-np.tanh(b)**2)*V
        obs=(Y-.5)*np.repeat(H,2); oracle=(Y-.5)*par
        mu=float(joint@obs); variance=float(joint@(obs**2)-mu**2)
        for n in [128,1024,8192,65536,1048576]:
            counts=rng.multinomial(n,joint,size=reps)
            scores=counts@obs/n; oracles=counts@oracle/n
            rows.append({'t':t,'n':n,'reps':reps,'mean_score_population':mu,'score_variance_population':variance,
                         'unit_snr_sample_proxy':variance/mu**2,'normal_sign_error_prediction':float(norm.cdf(-mu*np.sqrt(n/variance))),
                         'raw_score_nonpositive_count':int(np.sum(scores<=0)),
                         'parity_feature_nonpositive_count':int(np.sum(oracles<=0)),
                         'raw_scores':scores.tolist(),'parity_feature_scores':oracles.tolist()})
    payload={'status':'completed','seed':731090,'d':d,'eta':eta,'b':b,
             'controller_information':'Raw score uses fixed activation direction; comparator is supplied true parity feature. Neither uses q or eta in its estimate.',
             'sampling':'Ordinary iid finite observations represented exactly by multinomial category counts, not importance sampling',
             'population_information_nats':float(np.log(2)+(.5+eta)*np.log(.5+eta)+(.5-eta)*np.log(.5-eta)),
             'rows':rows}
    (ROOT/'results/score_sampling.json').write_text(json.dumps(payload,indent=2))
    print([{k:r[k] for k in ['t','n','raw_score_nonpositive_count','parity_feature_nonpositive_count','unit_snr_sample_proxy']} for r in rows])
if __name__=='__main__':main()

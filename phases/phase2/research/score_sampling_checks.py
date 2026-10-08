"""Read-only independent population/raw-array checks for score_sampling.py."""
import csv
import itertools
import json
import math
from pathlib import Path
import mpmath as mp
from high_order_independent import calculate, tanh_derivative


def main():
    mp.mp.dps=100
    root=Path(__file__).resolve().parents[1]
    saved=json.loads((root/'results/score_sampling.json').read_text())
    d=saved['d'];eta=mp.mpf(str(saved['eta']));b=mp.mpf(str(saved['b']))
    orientation=mp.sign(tanh_derivative(d,b))
    checks=[]
    for t0 in [.1,.2,.4,.8]:
        t=mp.mpf(str(t0));means=[];seconds=[]
        for state in itertools.product([-1,1],repeat=d):
            V=(orientation*state[0]+sum(state[1:]))/mp.sqrt(d)
            chi=math.prod(state)
            h=mp.tanh(b+t*V)-mp.tanh(b)-t*(1-mp.tanh(b)**2)*V
            means.append(eta*chi*h);seconds.append(h*h/4)
        mean=mp.fsum(means)/2**d
        variance=mp.fsum(seconds)/2**d-mean**2
        row=next(row for row in saved['rows'] if row['t']==t0)
        relmean=float(abs(mp.mpf(row['mean_score_population'])/mean-1))
        relvar=float(abs(mp.mpf(row['score_variance_population'])/variance-1))
        assert relmean<1e-8 and relvar<1e-12
        checks.append(dict(t=t0,full_cube_mean=mp.nstr(mean,35),
                           relative_double_mean_error=relmean,
                           relative_double_variance_error=relvar))
    for row in saved['rows']:
        assert row['raw_score_nonpositive_count']==sum(x<=0 for x in row['raw_scores'])
        assert row['parity_feature_nonpositive_count']==sum(x<=0 for x in row['parity_feature_scores'])
        assert len(row['raw_scores'])==len(row['parity_feature_scores'])==row['reps']
        assert abs(row['unit_snr_sample_proxy']-row['score_variance_population']/row['mean_score_population']**2)<1e-4
    d10=[]
    for row in csv.DictReader((root/'results/high_order_population.csv').open()):
        if int(row['d'])!=10 or float(row['t']) not in [.3,.01] or row['method']!='vanishing':
            continue
        t=mp.mpf(row['t']);d=10
        B=abs(tanh_derivative(d,b))/mp.mpf(d)**(mp.mpf(d)/2)
        K=tanh_derivative(2,b)**2*(3-mp.mpf(2)/d)/32
        p=eta*B/(2*K)*t**(d-4)
        risk,_,_=calculate(d,t,p)
        relative=abs(risk/mp.mpf(row['risk_change'])-1)
        assert relative<mp.mpf('1e-58')
        d10.append(dict(t=float(t),full_cube_risk_change=mp.nstr(risk,60),
                        grouped_relative_error=mp.nstr(relative,20),
                        normalized=row['normalized']))
    out=dict(full_cube_population_checks=checks,
             all_saved_counts_match_raw_arrays=True,
             minimum_absolute_sample_score=min(abs(v) for r in saved['rows'] for v in r['raw_scores']),
             information_nats=mp.nstr(mp.log(2)+(mp.mpf('.5')+eta)*mp.log(mp.mpf('.5')+eta)+(mp.mpf('.5')-eta)*mp.log(mp.mpf('.5')-eta),35),
             d10_checks=d10)
    Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()

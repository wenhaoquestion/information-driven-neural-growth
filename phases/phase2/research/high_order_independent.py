"""Independent hypercube enumeration and derivative-polynomial verification."""
from itertools import product
from pathlib import Path
import csv
import json
import mpmath as mp


def tanh_derivative(order, x):
    # Integer polynomial recurrence P_(k+1)(z)=(1-z^2)P'_k(z).
    coefficients = [0, 1]
    for _ in range(order):
        derivative = [j*coefficients[j] for j in range(1,len(coefficients))]
        updated = [0]*(len(derivative)+2)
        for j,c in enumerate(derivative):
            updated[j] += c
            updated[j+2] -= c
        coefficients = updated
    z=mp.tanh(x)
    return sum(c*z**j for j,c in enumerate(coefficients))


def calculate(d,t,p):
    b=mp.mpf('.5'); eta=mp.mpf('.15')
    sd=tanh_derivative(d,b); s2=tanh_derivative(2,b)
    direction=[mp.sign(sd)/mp.sqrt(d)]+[1/mp.sqrt(d)]*(d-1)
    direct=[]; score=[]; scoresecond=[]
    for state in product([-1,1],repeat=d):
        V=sum(x*v for x,v in zip(state,direction))
        parity=mp.mpf(1)
        for x in state: parity*=x
        q=mp.mpf('.5')+eta*parity
        logit=p*mp.tanh(b+t*V)+(1-p)*mp.tanh(b-p*t*V/(1-p))-mp.tanh(b)
        # Compute conditional Bernoulli log loss without the cosh identity.
        direct.append(q*mp.log(1+mp.exp(-logit))+(1-q)*mp.log(1+mp.exp(logit))-mp.log(2))
        h=mp.tanh(b+t*V)-mp.tanh(b)-t*tanh_derivative(1,b)*V
        score.append(eta*parity*h)
        scoresecond.append(h*h/4)
    mean=mp.fsum(score)/2**d
    return (mp.fsum(direct)/2**d,mean,mp.fsum(scoresecond)/2**d-mean*mean)


def main():
    mp.mp.dps=100
    root=Path(__file__).resolve().parents[1]
    grouped={}
    for row in csv.DictReader((root/'results/high_order_population.csv').open()):
        grouped[(int(row['d']),float(row['t']),row['method'])]=mp.mpf(row['risk_change'])
    rows=[]; comparisons=[]
    for d in [6,8]:
        b=mp.mpf('.5'); eta=mp.mpf('.15')
        B=abs(tanh_derivative(d,b))/mp.mpf(d)**(mp.mpf(d)/2)
        K=tanh_derivative(2,b)**2*(3-mp.mpf(2)/d)/32
        kappa=eta*B/(2*K)
        for radius in ['.2','.1','.03','.01']:
            t=mp.mpf(radius)
            paths=[('fixed_0.5',mp.mpf('.5'),4,K),
                   ('fixed_0.2',mp.mpf('.2'),4,K/16)]
            for factor in [mp.mpf('.5'),mp.mpf(1),mp.mpf(3)]:
                k=factor*kappa
                paths.append(('vanishing_'+str(factor),k*t**(d-4),2*d-4,K*k*k-eta*B*k))
            for name,p,power,coefficient in paths:
                risk,mean,variance=calculate(d,t,p)
                row=dict(d=d,t=float(t),path=name,p=mp.nstr(p,40),risk_change=mp.nstr(risk,70),
                         coefficient=mp.nstr(coefficient,40),normalized=mp.nstr(risk/(coefficient*t**power),40),
                         raw_score_mean_normalized=mp.nstr(mean/(eta*B*t**d),30),
                         raw_score_variance_normalized=mp.nstr(variance/(2*K*t**4),30))
                rows.append(row)
                method={'fixed_0.5':'balanced','vanishing_1.0':'vanishing'}.get(name)
                key=(d,float(t),method)
                if key in grouped:
                    err=abs(risk/grouped[key]-1)
                    assert err<mp.mpf('1e-58')
                    comparisons.append(dict(d=d,t=float(t),method=method,relative_error=mp.nstr(err,20)))
                if t==mp.mpf('.01'):
                    assert abs(risk/(coefficient*t**power)-1)<mp.mpf('.02')
    out=dict(setting='Independent full hypercube states, direct Bernoulli log loss, tanh derivative integer polynomial recurrence',
             decimal_precision=100,rows=rows,grouped_crosschecks=comparisons)
    (Path(__file__).with_name('high_order_independent.json')).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'max_grouped_relative_error':max(float(x['relative_error']) for x in comparisons),
                      't_.01_rows':[r for r in rows if r['t']==.01]},indent=2))


if __name__=='__main__': main()

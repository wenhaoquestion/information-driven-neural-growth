"""Population high-order split calculations; not neural training.
All hypercube states are enumerated (grouped by signed-coordinate sum).
mpmath precision resolves differences far below binary64 precision.
"""
from pathlib import Path
import math,json,csv,time
import mpmath as mp

ROOT=Path(__file__).resolve().parents[1]

def calculate(d,t,dps=100):
    mp.mp.dps=dps; t=mp.mpf(str(t)); b=mp.mpf('.5'); eta=mp.mpf('.15')
    deriv=mp.diff(mp.tanh,b,d); sigma2=mp.diff(mp.tanh,b,2)
    orientation=mp.sign(deriv)
    # v=(orientation,1,...,1)/sqrt(d); sum distribution remains binomial.
    B=abs(deriv)/mp.mpf(d)**(mp.mpf(d)/2)
    moment4=mp.mpf(3)-mp.mpf(2)/d
    K=sigma2**2*moment4/32
    kappa=eta*B/(2*K)
    p=kappa*t**(d-4)
    rows=[]
    for name,weight in [('balanced',mp.mpf('.5')),('vanishing',p)]:
        if weight>=mp.mpf('.5') and name=='vanishing': continue
        dr=signal=penalty=zero=scoremean=scoresecond=mp.mpf(0)
        for j in range(d+1):
            mass=mp.mpf(math.comb(d,j))/2**d
            V=(2*j-d)/mp.sqrt(d)
            parity=orientation*(-1)**(d-j)
            f=weight*mp.tanh(b+t*V)+(1-weight)*mp.tanh(b-weight/(1-weight)*t*V)-mp.tanh(b)
            sg=eta*parity*f
            pe=mp.log(mp.cosh(f/2))
            signal+=mass*sg;penalty+=mass*pe
            dr+=mass*(pe-sg)
            zero+=mass*f
        pred=K if name=='balanced' else -eta**2*B**2/(4*K)
        power=4 if name=='balanced' else 2*d-4
        rows.append(dict(d=d,t=float(t),method=name,p=mp.nstr(weight,40),risk_change=mp.nstr(dr,60),
                         convexity_penalty=mp.nstr(penalty,60),task_signal=mp.nstr(signal,60),
                         predicted_power=power,scaled_change=mp.nstr(dr/t**power,40),limit_coefficient=mp.nstr(pred,40),
                         normalized=mp.nstr(dr/(pred*t**power),30),B=mp.nstr(B,40),K=mp.nstr(K,40),kappa=mp.nstr(kappa,40),
                         dps=dps,source_mass=mp.nstr(mp.mpf(2)**(-d),20),output_l1=1,max_incoming_norm=float(t),
                         information_nats=float(mp.log(2)+(mp.mpf('.5')+eta)*mp.log(mp.mpf('.5')+eta)+(mp.mpf('.5')-eta)*mp.log(mp.mpf('.5')-eta))))
    return rows

def main():
    started=time.perf_counter();rows=[]
    for d in [6,8,10]:
        for t in [.3,.2,.1,.05,.02,.01,.005,.002,.001,.0003,.0001]:
            rows.extend(calculate(d,t))
    # Genuine numerical crosscheck: +70 digits, saved 60 decimal significant digits.
    precision=[]
    for d in [6,8,10]:
        low=calculate(d,.0001,100);hi=calculate(d,.0001,170)
        for a,z in zip(low,hi):
            err=abs(mp.mpf(a['risk_change'])/mp.mpf(z['risk_change'])-1)
            assert err<mp.mpf('1e-30'),(d,err)
            precision.append({'d':d,'method':a['method'],'relative_error':mp.nstr(err,10)})
    # At sufficiently small radii, assert signs and asymptotic approach, not global monotonicity.
    for row in rows:
        if row['t']<=.001:
            assert (mp.mpf(row['risk_change'])>0)==(row['method']=='balanced')
            assert abs(mp.mpf(row['normalized'])-1)<mp.mpf('.01')
    result={'setting':'Exact population; grouped exhaustive hypercube, no samples or training',
            'b':.5,'eta':.15,'rows':rows,'precision_checks':precision,'seconds':time.perf_counter()-started}
    (ROOT/'results/high_order_population.json').write_text(json.dumps(result,indent=2))
    with (ROOT/'results/high_order_population.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    print(json.dumps({'rows':len(rows),'precision':precision,'seconds':result['seconds']},indent=2))
    for row in rows:
        if row['t']==.1:print({k:row[k] for k in ['d','method','p','risk_change','normalized']})

if __name__=='__main__': main()

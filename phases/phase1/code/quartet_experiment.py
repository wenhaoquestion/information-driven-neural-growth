"""Exhaustive high-precision posterior-quantization and hierarchy calculations.
Run from the project root: .venv/bin/python code/quartet_experiment.py
No fitting, sampling, or selected optimizer starts: all 15 set partitions and
all 18 distinct 3-to-2 refinement chains are evaluated.
"""
from pathlib import Path
import csv, json, math, os, platform
from itertools import combinations
import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]

def partitions(items):
    if not items:
        yield ()
        return
    first, *rest = items
    for tail in partitions(rest):
        yield ((first,),) + tail
        for j in range(len(tail)):
            yield tail[:j] + (tuple(sorted((first,) + tail[j])),) + tail[j+1:]

def canonical(p):
    return tuple(sorted(tuple(sorted(b)) for b in p))

PARTITIONS = sorted(set(canonical(p) for p in partitions(list(range(4)))))
P2 = [p for p in PARTITIONS if len(p)==2]
P3 = [p for p in PARTITIONS if len(p)==3]
CHAINS = [(p2,p3) for p2 in P2 for p3 in P3
          if all(any(set(b)<=set(c) for c in p2) for b in p3)]
assert len(PARTITIONS)==15 and len(P2)==7 and len(P3)==6 and len(CHAINS)==18

def h(t):
    if t==0 or t==1:
        return mp.mpf(0)
    return -t*mp.log(t)-(1-t)*mp.log1p(-t)

def label(p):
    names='ACDB'
    return '|'.join(''.join(names[i] for i in b) for b in p)

def cost(part, weights, posterior, loss='log'):
    total=mp.mpf(0)
    for block in part:
        mass=sum(weights[i] for i in block)
        mean=sum(weights[i]*posterior[i] for i in block)/mass
        if loss=='log':
            total += mass*h(mean)-sum(weights[i]*h(posterior[i]) for i in block)
        elif loss=='brier':
            total += sum(weights[i]*(posterior[i]-mean)**2 for i in block)
        else:
            raise ValueError(loss)
    return total

def evaluate(L, multiplier=1, keep_all=False, extra_dps=0):
    # Smallest posterior is O(exp(-2L)); evaluating entropy differences needs this precision.
    mp.mp.dps = max(80,math.ceil(2*L/math.log(10))+90)+extra_dps
    Lm=mp.mpf(str(L))
    w=mp.exp(-Lm); e=w*w; c=(1-2*w)/2
    delta=mp.mpf(str(multiplier))*mp.sqrt(mp.log(2)/Lm)
    assert e<delta<mp.mpf('0.5')
    weights=[c,w,w,c]; posterior=[e,delta,1-delta,1-e]
    costs={p:cost(p,weights,posterior) for p in PARTITIONS}
    opt2=min(costs[p] for p in P2); opt3=min(costs[p] for p in P3)
    scores={chain:max(costs[chain[0]]/opt2,costs[chain[1]]/opt3) for chain in CHAINS}
    best=min(scores,key=scores.get); price=scores[best]
    a=cost(((0,1),(2,),(3,)),weights,posterior)
    r=cost(((0,),(1,2),(3,)),weights,posterior)
    t=cost(((0,1,2),(3,)),weights,posterior)
    formula=min(a/r,t/(2*a))
    assert mp.almosteq(opt2,2*a,rel_eps=mp.mpf("1e-40")) and mp.almosteq(opt3,r,rel_eps=mp.mpf("1e-40")), (L,multiplier)
    assert mp.almosteq(price,formula,rel_eps=mp.mpf("1e-40")), (L,multiplier)
    baseline=sum(weights[i]*h(posterior[i]) for i in range(4))
    total_scores={chain:max((baseline+costs[chain[0]])/(baseline+opt2),
                            (baseline+costs[chain[1]])/(baseline+opt3)) for chain in CHAINS}
    total_price=min(total_scores.values())
    points=[((baseline+costs[p2])/(baseline+opt2),
             (baseline+costs[p3])/(baseline+opt3)) for p2,p3 in CHAINS]
    randomized=total_price
    # The minimum of max(x,y) on this 2D hull is a vertex or a diagonal crossing of an edge.
    for z1,z2 in combinations(points,2):
        d1=z1[0]-z1[1]; d2=z2[0]-z2[1]
        if d1*d2<0:
            mix=-d2/(d1-d2)
            randomized=min(randomized,mix*z1[0]+(1-mix)*z2[0])
    total_A=(baseline+a)/(baseline+r)
    total_B=(baseline+t)/(baseline+2*a)
    random_formula=1+(total_A-1)*(total_B-1)/(total_A+total_B-2)
    assert mp.almosteq(randomized,random_formula,rel_eps=mp.mpf('1e-40'))
    # Feasible stochastic 4->3->2 representation; half of D joins C.
    channel=[[1,0,0],[0,1,0],[0,mp.mpf('0.5'),mp.mpf('0.5')],[0,0,1]]
    def channel_risk(K):
        ans=mp.mpf(0)
        for j in range(len(K[0])):
            mass=sum(weights[i]*K[i][j] for i in range(4))
            if mass:
                mean=sum(weights[i]*K[i][j]*posterior[i] for i in range(4))/mass
                ans+=mass*h(mean)
        return ans
    coarse=[[row[0]+row[1],row[2]] for row in channel]
    stochastic_r2=channel_risk(coarse); stochastic_r3=channel_risk(channel)
    stochastic_upper=max(stochastic_r2/(baseline+opt2),stochastic_r3/(baseline+opt3))
    assert stochastic_upper+mp.mpf('1e-40')>=randomized
    bcost={p:cost(p,weights,posterior,'brier') for p in PARTITIONS}
    b2=min(bcost[p] for p in P2); b3=min(bcost[p] for p in P3)
    bp=min(max(bcost[p2]/b2,bcost[p3]/b3) for p2,p3 in CHAINS)
    pred=mp.sqrt(Lm)/(2*mp.sqrt(mp.log(2)))
    row=dict(L=L,multiplier=multiplier,dps=mp.mp.dps,delta=float(delta),
             log10_rare_mass=float(mp.log10(w)),opt2_over_w=float(opt2/w),
             opt3_over_w=float(opt3/w),near3_ratio=float(a/r),rare2_ratio=float(t/(2*a)),
             price=float(price),asymptotic=float(pred),normalized_price=float(price/pred),
             brier_price=float(bp),total_price=float(total_price),randomized_total_price=float(randomized),
             stochastic_upper=float(stochastic_upper),baseline_over_opt3=float(baseline/opt3),
             best2=label(best[0]),best3=label(best[1]))
    full={"parameters":{k:mp.nstr(v,60) for k,v in dict(epsilon=e,w=w,c=c,delta=delta).items()},
          "row":row,"opt2":mp.nstr(opt2,60),"opt3":mp.nstr(opt3,60),
          "price":mp.nstr(price,60),"total_price":mp.nstr(total_price,60),"baseline":mp.nstr(baseline,60),"randomized_total_price":mp.nstr(randomized,60),
          "stochastic_upper":mp.nstr(stochastic_upper,60),
          "stochastic_r2":mp.nstr(stochastic_r2,60),"stochastic_r3":mp.nstr(stochastic_r3,60),
          "partitions":[dict(partition=label(p),states=len(p),cost=mp.nstr(costs[p],60),
                              cost_over_w=mp.nstr(costs[p]/w,60)) for p in PARTITIONS],
          "chains":[dict(level2=label(p2),level3=label(p3),
                         ratio=mp.nstr(scores[p2,p3],60),total_ratio=mp.nstr(total_scores[p2,p3],60),
                         total_level2=mp.nstr((baseline+costs[p2])/(baseline+opt2),60),
                         total_level3=mp.nstr((baseline+costs[p3])/(baseline+opt3),60)) for p2,p3 in CHAINS]}
    return row,full

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    (ROOT/'results').mkdir(exist_ok=True)
    Ls=[16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048]
    vals=[evaluate(L) for L in Ls]
    write_csv(ROOT/'results/quartet_scaling.csv',[v[0] for v in vals])
    (ROOT/'results/quartet_all_partitions.json').write_text(json.dumps([v[1] for v in vals],indent=2))
    # Purposeful sensitivity check around the asymptotically balanced choice.
    sensitivity=[evaluate(256,float(t))[0] for t in [0.35,0.5,0.65,0.8,1,1.2,1.5,2,2.5,3]]
    write_csv(ROOT/'results/quartet_sensitivity.csv',sensitivity)
    # Independent higher-precision recalculation, including the smallest losses.
    checks=[]
    for L in [16,64,256,2048]:
        low=next(v[1] for v in vals if v[0]['L']==L)
        hi=evaluate(L,extra_dps=70)[1]
        errs={}
        for key in ['price','total_price','randomized_total_price','stochastic_upper','stochastic_r2','stochastic_r3']:
            err=abs(mp.mpf(low[key])-mp.mpf(hi[key]))/mp.mpf(hi[key])
            assert low[key]==hi[key], (L,key,'60-digit disagreement')
            errs[key]=mp.nstr(err,8)
        checks.append(dict(L=L,relative_differences_of_stored_values=errs,stored_digits=60))
    report=dict(partitions=15,three_to_two_chains=18,scaling_cases=len(Ls),sensitivity_cases=len(sensitivity),
                precision_checks=checks,assertions='All passed',python=platform.python_version(),
                mpmath=mp.__version__,seed='None: deterministic exhaustive evaluation')
    (ROOT/'results/verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print('largest-L price:', vals[-1][0]['price'])

if __name__=='__main__':
    main()

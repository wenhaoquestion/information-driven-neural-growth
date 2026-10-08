#!/usr/bin/env python3
"""Adversarial computational checks of the new entropy perturbation/rate proof.
Run from repository root: python phases/phase4/publication/finite/research/sharp_rate_check.py
Independent instances are mathematical test cases, not empirical learner replicates.
"""
from pathlib import Path
import itertools,json,math,platform,time
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parent
SEED=20260914

def parts():
    ans=set()
    for labels in itertools.product(range(4),repeat=4):
        ans.add(tuple(sorted(tuple(i for i,l in enumerate(labels) if l==k) for k in set(labels))))
    return sorted(ans)
P=parts();P2=[p for p in P if len(p)==2];P3=[p for p in P if len(p)==3]
CHAINS=[(a,b) for a in P2 for b in P3 if all(any(set(c)<=set(d) for d in a) for c in b)]

def divergence(p,q):
    good=p>0
    return float(np.sum(p[good]*np.log(p[good]/q[good])))
def cost(weights,posterior,part):
    total=0.
    for cell in part:
        ids=np.array(cell);ws=weights[ids];ps=posterior[ids]
        mean=(ws[:,None]*ps).sum(0)/ws.sum()
        total+=sum(float(weights[i])*divergence(posterior[i],mean) for i in ids)
    return max(total,0.)
def ent(mu,u):
    z=u*u;mean=mu@z
    if mean==0:return 0.
    good=z>0
    return max(0.,float(np.sum(mu[good]*z[good]*np.log(z[good]/mean))))

def main():
    start=time.monotonic();rng=np.random.default_rng(SEED)
    rows=[];liprows=[];maxlip=0.;maxgrad=0.;maxmove=0.;maxprice=0.
    for trial in range(4000):
        n=2+trial%15
        mu=np.exp(rng.uniform(-30,0,n));mu/=mu.sum()
        u=np.exp(rng.uniform(-30,3,n));v=np.exp(rng.uniform(-30,3,n))
        if trial%5==0:u[rng.random(n)<.2]=0.
        if trial%7==0:v[:]=rng.uniform(0,2)
        if trial%101==0:u[:]=0.
        if trial%103==0:v[:]=0.
        c=2+np.log(1/mu.min());fu=ent(mu,u);fv=ent(mu,v);norm=float(mu@((u-v)**2))
        gap=(np.sqrt(fu)-np.sqrt(fv))**2
        assert gap<=c*norm+1e-11*max(1,fu,fv),(trial,gap,c*norm)
        frac=gap/(c*norm) if norm>1e-20 else None
        maxlip=max(maxlip,frac or 0.)
        mean=mu@(u*u);grad=None
        if fu>1e-12 and mean>0:
            good=u>0;r=u[good]**2/mean
            grad=float(np.sum(mu[good]*u[good]**2*np.log(r)**2)/fu/c)
            assert grad<=1+1e-8,(trial,grad)
            maxgrad=max(maxgrad,grad)
        liprows.append(dict(trial=trial,dimension=n,mu=mu.tolist(),u=u.tolist(),v=v.tolist(),lip_fraction=frac,gradient_fraction=grad))
    for trial in range(1204):
        m=2+trial%7
        weights=np.exp(rng.uniform(-25,0,4));weights/=weights.sum()
        posterior=np.exp(rng.uniform(-35,0,(4,m)))
        if trial%3==0:
            posterior[rng.random(posterior.shape)<.25]=0.;posterior[:,0]+=1e-12
        posterior/=posterior.sum(1)[:,None]
        if trial==1200:posterior[:]=posterior[0]
        if trial==1201:posterior[:]=0.;posterior[:,0]=1.
        if trial==1202:posterior[1]=posterior[0]
        if trial==1203:posterior[2]=posterior[3]
        costs={p:cost(weights,posterior,p) for p in P}
        best2=min(P2,key=costs.get);best3=min(P3,key=costs.get)
        a=costs[best2];r=costs[best3];gamma=weights.min();c=2+math.log(1/gamma);upper=2+2*math.sqrt(1+c)
        pair=next(cell for cell in best3 if len(cell)==2);i,j=sorted(pair,key=lambda k:weights[k])
        ci=next(cell for cell in best2 if i in cell);cj=next(cell for cell in best2 if j in cell)
        move=None;frac=None
        if ci!=cj:
            moved=tuple(sorted([tuple(k for k in ci if k!=i),tuple(sorted(cj+(i,)))]));moved=tuple(x for x in moved if x)
            # The raw move may have one cell; a split into unions of fine cells only improves it.
            move=cost(weights,posterior,moved)
            uppermove=(math.sqrt(2*a)+math.sqrt(2*c*r))**2
            assert move<=uppermove+1e-12,(trial,move,uppermove)
            frac=move/uppermove if uppermove>1e-16 else None;maxmove=max(maxmove,frac or 0.)
        price=None
        if r>1e-13 and a>1e-13:
            price=min(max(costs[p2]/a,costs[p3]/r) for p2,p3 in CHAINS)
            assert price<=upper+1e-10,(trial,price,upper)
            maxprice=max(maxprice,price/upper)
        rows.append(dict(trial=trial,weights=weights.tolist(),posterior=posterior.tolist(),optimal2=best2,minimum_pair=pair,opt2=a,opt3=r,raw_moved_cost=move,move_bound_fraction=frac,hierarchy_price=price,new_upper_bound=upper))
    for name,data in [('sharp_rate_raw.jsonl',rows),('entropy_lipschitz_raw.jsonl',liprows)]:
        with (ROOT/name).open('w') as f:
            for row in data:f.write(json.dumps(row,separators=(',',':'))+'\n')
    # Higher-precision independent entropy-coordinate calculation on selected extreme and zero cases.
    mp.mp.dps=90;mpchecks=[]
    for ix in [0,1,7,21,50,103,207,305,777,1199,1200,1201,1202,1203]:
        row=rows[ix];ws=list(map(lambda x:mp.mpf(str(x)),row['weights']));ws=[w/sum(ws) for w in ws]
        ps=[[mp.mpf(str(x)) for x in r] for r in row['posterior']];ps=[[x/sum(r) for x in r] for r in ps]
        def mpcost(part):
            result=mp.mpf(0)
            for cell in part:
                mass=sum(ws[i] for i in cell)
                for k in range(len(ps[0])):
                    z=sum(ws[i]*ps[i][k] for i in cell)
                    if z:
                        result+=sum(ws[i]*ps[i][k]*mp.log(ps[i][k]*mass/z) for i in cell if ps[i][k])
            return max(result,mp.mpf(0))
        vals={p:mpcost(p) for p in P};a=min(vals[p] for p in P2);r=min(vals[p] for p in P3)
        c=2+mp.log(1/min(ws));bound=2+2*mp.sqrt(1+c)
        if a>mp.mpf('1e-70') and r>mp.mpf('1e-70'):
            price=min(max(vals[p2]/a,vals[p3]/r) for p2,p3 in CHAINS);assert price<=bound
        else:price=None
        mpchecks.append(dict(trial=ix,opt2=mp.nstr(a,60),opt3=mp.nstr(r,60),price=None if price is None else mp.nstr(price,60),bound=mp.nstr(bound,60)))
    (ROOT/'sharp_rate_high_precision.json').write_text(json.dumps(mpchecks,indent=2)+'\n')
    summary=dict(status='all assertions passed',seed=SEED,python=platform.python_version(),numpy=np.__version__,mpmath=mp.__version__,scalar_vector_tests=4000,hierarchy_instances=1204,high_precision_rechecks=len(mpchecks),max_lipschitz_bound_fraction=maxlip,max_gradient_bound_fraction=maxgrad,max_regrouping_bound_fraction=maxmove,max_hierarchy_price_over_bound=maxprice,seconds=time.monotonic()-start,note='Randomized mathematical diagnostics; not a proof or trained-learning benchmark. Near-zero floating ratios omitted; explicit zero cases retained and checked at high precision.')
    (ROOT/'sharp_rate_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

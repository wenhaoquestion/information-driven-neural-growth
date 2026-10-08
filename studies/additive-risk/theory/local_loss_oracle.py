"""Certified full-domain loss oracle for the single interior n=2048 candidate.

Only Fraction/stdlib arithmetic. Its narrow tail certificate uses the public
loss parameters and fixed tail radius, not unknown posterior input. Outside
the certified tail region it returns conservative but valid intervals.
"""
from fractions import Fraction as Q
from math import isqrt
from pathlib import Path
import json
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"statistics"))
from honest_delta import Interval


def down(x, digits=30):
    return Q((x*10**digits).numerator//(x*10**digits).denominator,10**digits)


def up(x, digits=30):
    return -down(-x,digits)


def sqrtlo(x, digits=35):
    return Q(isqrt(x.numerator*10**(2*digits)//x.denominator),10**digits)


def sqrthi(x, digits=35):
    return sqrtlo(x,digits)+Q(1,10**digits)


def expneg_upper(x, terms=200):
    assert x >= 0
    total=term=Q(1)
    for j in range(1,terms+1):
        term=term*x/j
        total+=term
    return up(1/total,50)


def div_interval(lo,hi,alo,ahi):
    pts=(lo/alo,lo/ahi,hi/alo,hi/ahi)
    return min(pts),max(pts)


class Interior2048Loss:
    variation=Q(2)

    def __init__(self):
        l2=2*sum((Q(1,(2*j+1)*3**(2*j+1)) for j in range(40)),Q())
        u2=l2+Q(2,81*3**81)/(1-Q(1,9))
        self.el=down(16*11*l2/2048)
        self.eu=up(16*11*u2/2048)
        self.delta=up(2/(2048**5*11*l2))
        self.alo=down(4+64*self.el**2-2*self.delta)
        self.ahi=up(4+64*self.eu**2)
        self.r=self.el-Q(1,40)
        self.eta=Q(1,200)
        assert 0 < self.eta < self.r < self.el < self.eu < Q(1,16)
        u=(self.el**2-self.eta**2)/16
        vcore=sqrtlo(2*u/(1+u/2))
        vtail=sqrthi((self.eu**2-self.r**2)/8)
        assert vcore > vtail
        self.exponent=down(4096*(vcore-vtail))
        self.central=up(2*(self.eu-self.r)/self.eta*expneg_upper(self.exponent))
        self.outer=up(8/self.eta*expneg_upper(down(4096*vcore)))
        self.tail=up(self.central+self.outer)
        self.hinge_error=up((self.eu-self.r)*self.central+4*self.outer)
        self.global_hinge_error=up(self.eu/2+self.delta)
        de=self.eu-self.el
        de2=self.eu**2-self.el**2
        self.entropy_width=up((2*de+16*de2+2*self.hinge_error)/self.alo
             +(2+16*self.eu**2)*(1/self.alo-1/self.ahi))
        self.slope_width=up(4*(2*self.tail+16*de2)/self.alo
             +4*(1+16*self.eu**2)*(1/self.alo-1/self.ahi))
        assert self.entropy_width < Q(2,10**8)
        assert self.slope_width < Q(3,10**6)

    def _args(self,q):
        q=Q(q)
        if not 0 <= q <= 1:
            raise ValueError("Report must be in full [0,1] domain")
        z=4*q-Q(3,2)
        return z,((-z-self.eu,-z-self.el),(z-1-self.eu,z-1-self.el))

    def _safe(self,arg):
        lo,hi=arg
        return hi <= -self.r or lo >= self.r

    def locally_narrow(self,q):
        _,args=self._args(q)
        return all(self._safe(a) for a in args)

    def entropy(self,q):
        z,args=self._args(q)
        lo=sum(max(a,0) for a,b in args)+4*self.el**2*(z-Q(1,2))**2
        hi=sum(max(b,0) for a,b in args)+4*self.eu**2*(z-Q(1,2))**2
        hi+=sum(self.hinge_error if self._safe(a) else self.global_hinge_error for a in args)
        fl,fu=div_interval(lo,hi,self.alo,self.ahi)
        return Interval(-fu,-fl)

    def slope(self,q):
        z,args=self._args(q)
        cdfs=[]
        for lo,hi in args:
            if hi <= -self.r:
                cdfs.append((Q(0),self.tail))
            elif lo >= self.r:
                cdfs.append((1-self.tail,Q(1)))
            else:
                cdfs.append((Q(0),Q(1)))
        quad=(8*self.el**2*(z-Q(1,2)),8*self.eu**2*(z-Q(1,2)))
        lo=-cdfs[0][1]+cdfs[1][0]+min(quad)
        hi=-cdfs[0][0]+cdfs[1][1]+max(quad)
        fl,fu=div_interval(4*lo,4*hi,self.alo,self.ahi)
        return Interval(-fu,-fl)


def enc(x):
    return {"rational":str(x),"decimal":str(float(x))}


def main():
    from itertools import combinations
    loss=Interior2048Loss()
    p=Path(__file__).resolve().parent
    source=p.parent/"statistics"/"INTERIOR_2048_POWER_ENVELOPE.json"
    data=json.loads(source.read_text())
    rect=[(Q(r['lo']['rational']),Q(r['hi']['rational'])) for r in data['rectangle']]
    truth=(Q(1,8),Q(3,8),Q(5,8),Q(7,8))
    assert all(t-Q(1,160) <= lo <= hi <= t+Q(1,160) for t,(lo,hi) in zip(truth,rect))
    cells=[]
    for size in range(1,5):
        for block in combinations(range(4),size):
            lo=sum(rect[i][0] for i in block)/size
            hi=sum(rect[i][1] for i in block)/size
            assert loss.locally_narrow(lo) and loss.locally_narrow(hi)
            # Entire cell-mean interval, not merely its endpoints, is narrow.
            argslo=loss._args(lo)[1]
            argshi=loss._args(hi)[1]
            assert all(loss._safe((min(a[0],b[0]),max(a[1],b[1])))
                       for a,b in zip(argslo,argshi))
            for q in (lo,hi):
                h,s=loss.entropy(q),loss.slope(q)
                assert h.hi-h.lo <= loss.entropy_width
                assert s.hi-s.lo <= loss.slope_width
            cells.append({"block":block,"mean":[enc(lo),enc(hi)],
               "hprime_interval":[enc(loss.slope(hi).lo),enc(loss.slope(lo).hi)]})
    constants={name:enc(getattr(loss,name)) for name in
       ('el','eu','delta','alo','ahi','r','eta','exponent','central','outer','tail',
        'hinge_error','global_hinge_error','entropy_width','slope_width')}
    out={"candidate":"interior n=2048 only","constants":constants,"cells":cells,
         "full_domain_fallback":True,"all_15_cell_intervals_certified":True,
         "decimal_displays":"informational only; rational strings are certificates",
         "envelope_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
         "oracle_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (p/'LOCAL_ORACLE_CERTIFICATE.json').write_text(json.dumps(out,indent=2)+'\n')
    for k in ('tail','hinge_error','entropy_width','slope_width'):
        print(k,constants[k])
    print('All 15 cell intervals and uniform oracle-width assertions passed.')
    print('Full report domain [0,1]; exact rational arithmetic; no quadrature.')


if __name__=='__main__':
    main()

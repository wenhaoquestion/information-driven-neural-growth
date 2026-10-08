"""Independent direct checks of the two-task neural theorem; stdlib only."""
import json
import math
from pathlib import Path


def risk(s, b, eta, incoming, outgoing, intercept):
    values = []
    for j in range(2):
        q = .5 + (1 if j == 0 else -1) * s * eta
        for sign in [-1, 1]:
            f = intercept + sum(a * math.tanh(b + sign * w[j])
                                for w, a in zip(incoming, outgoing))
            values.append(math.log1p(math.exp(f)) - q * f)
    return sum(values) / 4


def main():
    b, eta = .5, .15
    z = math.tanh(b)
    phip = 1 - z*z
    phipp = -2*z*phip
    stability_threshold = phip**2/(4*abs(phipp))
    discrepancies = []
    complement_sums = []
    for angle in range(101):
        v = (math.cos(angle*.091), math.sin(angle*.091))
        for eps in [.01, .1, .4, 1, 4]:
            w = tuple(eps * vi for vi in v)
            ds = [z - (math.tanh(b+wj)+math.tanh(b-wj))/2 for wj in w]
            direct = []
            for s in [-1, 1]:
                delta = risk(s, b, eta, [w, tuple(-x for x in w)],
                             [.5,.5], -z)-math.log(2)
                closed = .5*sum(math.log(math.cosh(dj/2)) for dj in ds) + s*eta*(ds[0]-ds[1])/2
                discrepancies.append(abs(delta-closed))
                direct.append(delta)
            complement_sums.append(sum(direct))
    step = 1e-4
    hessian_direct = []
    hessian_exact = []
    for j in range(2):
        w = tuple(step if k == j else 0 for k in range(2))
        hessian_direct.append(2*(risk(1,b,eta,[w],[1],-z)-math.log(2))/step**2)
        hessian_exact.append(phip**2/8 + phipp * (-eta/2 if j == 0 else eta/2))
    gains = []
    for eps in [.01,.1,.4,1,4]:
        direct = math.log(2)-risk(1,b,eta,[(0,eps),(0,-eps)],[.5,.5],-z)
        d = z-(math.tanh(b+eps)+math.tanh(b-eps))/2
        bound = eta*d/2-d*d/16
        gains.append(dict(epsilon=eps,gain=direct,lower_bound=bound,d=d))
    sample_rows=[]
    for n in [4,8,16,32,64,128]:
        # Equal-prior error; an exact zero sum is randomized fairly.
        p=.5+eta
        error=sum(math.comb(n,k)*p**k*(1-p)**(n-k) for k in range((n+1)//2))
        if n%2==0:
            error+=.5*math.comb(n,n//2)*(p*(1-p))**(n//2)
        lower=(1-4*eta*eta)**n/4
        upper=math.exp(-2*n*eta*eta)
        assert lower <= error <= upper
        sample_rows.append(dict(n=n,exact_error=error,lower_bound=lower,upper_bound=upper))
    out={"parameters":{"b":b,"eta":eta},
         "stability_threshold":stability_threshold,
         "maximum_exact_formula_error":max(discrepancies),
         "minimum_two_task_sum":min(complement_sums),
         "hessian_direct":hessian_direct,"hessian_exact":hessian_exact,
         "split_gains":gains,"sample_test":sample_rows}
    assert max(discrepancies)<1e-14
    assert min(complement_sums)>-1e-14
    assert max(abs(x-y) for x,y in zip(hessian_exact,hessian_direct))<1e-7
    assert all(g["gain"]+1e-14>=g["lower_bound"] for g in gains)
    out_path=Path(__file__).with_name("theory_checks.json")
    out_path.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))


if __name__ == "__main__":
    main()

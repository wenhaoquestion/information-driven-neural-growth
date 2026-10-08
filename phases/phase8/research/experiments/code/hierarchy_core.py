"""Four-state Jensen costs without subtracting nearly equal generator values.

All weights are original source masses, not per-cell normalized masses.
Polynomial integration uses Gauss-Legendre order >= 2n+1, which integrates
degree-4n curvature times each linear Jensen-kernel segment exactly in exact
arithmetic. Floating-point accuracy is checked separately; no interval claim.
"""
from functools import lru_cache
from itertools import combinations
import numpy as np
from scipy.special import roots_legendre


def partitions(n, k):
    def rec(prefix):
        if len(prefix) == n:
            if max(prefix) + 1 == k:
                yield tuple(tuple(i for i, a in enumerate(prefix) if a == j)
                            for j in range(k))
            return
        for a in range(min(k - 1, max(prefix) + 1) + 1):
            yield from rec(prefix + [a])
    return list(rec([0]))


P2, P3 = partitions(4, 2), partitions(4, 3)
HIERARCHIES = [(i, j) for i, c in enumerate(P2) for j, f in enumerate(P3)
               if all(any(set(a) <= set(b) for b in c) for a in f)]
assert (len(P2), len(P3), len(HIERARCHIES)) == (7, 6, 18)


def kernel(t, x, w):
    """K(t)=sum w_i (x_i-t)+ - sum(w)*(mean-t)+, stable two-sided form."""
    t = np.asarray(t)
    total = np.sum(w)
    if total <= 0:
        return np.zeros_like(t)
    mean = np.dot(w, x) / total
    left = np.sum(w[:, None] * np.maximum(t[None, :] - x[:, None], 0), axis=0)
    right = np.sum(w[:, None] * np.maximum(x[:, None] - t[None, :], 0), axis=0)
    return np.where(t < mean, left, right)


def hinge_cell(x, w, hinges, quadratic):
    total = sum(w)
    if total <= 0:
        return 0.0
    mean = np.dot(w, x) / total
    return float(np.sum(kernel(np.asarray(hinges), x, w))
                 + quadratic * np.dot(w, (x - mean)**2))


@lru_cache(maxsize=6)
def gauss(order):
    return roots_legendre(order)


class NeedleCurvature:
    """v1 construction on z in [-3/2,5/2], no power-basis coefficients."""
    def __init__(self, n, epsilon, order=None):
        self.n, self.epsilon = int(n), float(epsilon)
        self.order = int(order or (2*n + 2))
        self.logpeak = self._logp(np.array([0.]))[0]
        u, w = gauss(self.order)
        self.z_scaled = float(4 * np.dot(w, self._scaled(4*u)))
        self.logZ = self.logpeak + np.log(self.z_scaled)

    def _logp(self, s):
        # Stable acosh(1+v): 2 asinh(sqrt(v/2)), avoiding rounding 1+v.
        v = (self.epsilon**2 - s*s) / 16
        out = np.empty_like(v)
        pos = v > 0
        a = self.n * (2 * np.arcsinh(np.sqrt(v[pos]/2)))
        out[pos] = 2 * (a + np.log1p(np.exp(-2*a)) - np.log(2))
        # acos(1+v)=2 asin(sqrt(-v/2)), also stable near zero.
        a = self.n * (2 * np.arcsin(np.sqrt(np.maximum(-v[~pos], 0)/2)))
        with np.errstate(divide='ignore'):
            out[~pos] = 2*np.log(np.abs(np.cos(a)))
        return out

    def _scaled(self, s):
        return np.exp(self._logp(s) - self.logpeak)

    def kappa(self, s):
        s = np.asarray(s)
        return np.where(np.abs(s) <= 4, self._scaled(s)/self.z_scaled, 0)

    def __call__(self, z):
        return self.kappa(-z-self.epsilon) + self.kappa(z-1-self.epsilon) + 8*self.epsilon**2

    def cell(self, x, w):
        w = np.asarray(w)
        active = w > 0
        x, w = np.asarray(x)[active], w[active]
        if len(x) <= 1:
            return 0.0
        mean = np.dot(x, w) / sum(w)
        extra = [a for a in getattr(self, 'breaks', []) if min(x) < a < max(x)]
        knots = sorted(set([*x, mean, *extra]))
        u, v = gauss(self.order)
        ans = 0.0
        for a, b in zip(knots[:-1], knots[1:]):
            t = (a+b)/2 + (b-a)*u/2
            ans += (b-a)/2*np.dot(v, kernel(t, x, w)*self(t))
        return float(ans)

    def max_upper(self):
        # Each squared-Chebyshev kernel is maximized at s=0 on [-4,4].
        # This rigorous analytic upper bound avoids mislabeling a sampled max.
        return 2/self.z_scaled + 8*self.epsilon**2


class EndpointCurvature(NeedleCurvature):
    """Degree-2n endpoint Chebyshev needles, evaluator restricted to [0,1].

    It must not be used to estimate the polynomial's maximum outside this
    interval: the clipped trigonometric evaluation is only valid on [0,1].
    """
    def __init__(self, n, delta, order=None):
        self.n, self.delta = int(n), float(delta)
        self.order = int(order or (n+2))
        self.logpeak = self._logp(np.array([0.]))[0]
        u,w = gauss(self.order)
        # Whole-[0,1] high-order roots have poor relative accuracy in the
        # endpoint weights at very large n. Separate the narrow spike first.
        self.breaks=[self.delta/4,self.delta,1-self.delta,1-self.delta/4]
        self.z_scaled = 0.
        knots=[0.,self.delta/4,self.delta,1.]
        for a,b in zip(knots[:-1],knots[1:]):
            self.z_scaled += (b-a)/2*np.dot(w,self._scaled((a+b)/2+(b-a)*u/2))
        self.logZ=self.logpeak+np.log(self.z_scaled)

    def _logp(self,t):
        v=2*(self.delta-t)/(1-self.delta)
        out=np.empty_like(v); pos=v>0
        a=self.n*2*np.arcsinh(np.sqrt(v[pos]/2))
        out[pos]=2*(a+np.log1p(np.exp(-2*a))-np.log(2))
        a=self.n*2*np.arcsin(np.sqrt(np.clip(-v[~pos]/2,0,1)))
        with np.errstate(divide='ignore'):out[~pos]=2*np.log(np.abs(np.cos(a)))
        return out

    def kappa(self,t):
        return self._scaled(np.asarray(t))/self.z_scaled

    def __call__(self,t):
        return self.kappa(t)+self.kappa(1-t)+self.delta**2

    def max_upper(self):
        return 2/self.z_scaled+self.delta**2


def evaluate(x, p, cell):
    x, p = np.asarray(x, float), np.asarray(p, float)
    cells = {tuple(c): cell(x[list(c)], p[list(c)])
             for k in range(1, 5) for c in combinations(range(4), k)}
    d2 = np.array([sum(cells[c] for c in part) for part in P2])
    d3 = np.array([sum(cells[c] for c in part) for part in P3])
    o2, o3 = float(min(d2)), float(min(d3))
    assert o2 > 0 and o3 > 0
    vectors = np.array([(d2[i]/o2, d3[j]/o3) for i, j in HIERARCHIES])
    scores = vectors.max(axis=1)
    best = int(np.argmin(scores))
    # Minimizing max over a 2D convex hull attains its optimum at a point
    # or on a segment between two of the finitely many input vertices.
    lower = float(min(scores))
    mix = (best, best, 1.0)
    for i, j in combinations(range(18), 2):
        a, b = vectors[i], vectors[j]
        den = a[0]-b[0]-a[1]+b[1]
        if abs(den) > 1e-15:
            t = (b[1]-b[0])/den
            if 0 <= t <= 1:
                value = float(np.max(t*a+(1-t)*b))
                if value < lower:
                    lower, mix = value, (i, j, float(t))
    return dict(opt2=o2, opt3=o3, rho_det=float(scores[best]),
                stochastic_convex_lower=lower, convex_mixture=mix,
                best_hierarchy=best, flat2=int(np.argmin(d2)), flat3=int(np.argmin(d3)),
                d2=d2.tolist(), d3=d3.tolist(), vectors=vectors.tolist(),
                cells={','.join(map(str, c)): v for c, v in cells.items()})


EXPLICIT_Q = np.array([[1.,0,0], [0,1,0], [0,.5,.5], [0,0,1]])
EXPLICIT_R = np.array([[1.,0], [1.,0], [0,1.]])


def channel(x, p, Q, R, cell, opt2, opt3):
    Q, R = np.asarray(Q), np.asarray(R)
    assert np.min(Q) >= 0 and np.min(R) >= 0
    assert np.max(np.abs(Q.sum(1)-1)) < 1e-12
    assert np.max(np.abs(R.sum(1)-1)) < 1e-12
    coarse = Q @ R
    d3 = sum(cell(x, p*Q[:, j]) for j in range(Q.shape[1]))
    d2 = sum(cell(x, p*coarse[:, j]) for j in range(coarse.shape[1]))
    return dict(d2=float(d2), d3=float(d3), ratio2=float(d2/opt2),
                ratio3=float(d3/opt3), upper=float(max(d2/opt2, d3/opt3)))

"""Exact-rational, sample-free building blocks for honest four-state inference.

No candidate posterior is an input to infer_delta.  Loss oracle outputs must
enclose the normalized Bayes entropy and its derivative over the FULL [0,1].
See DERIVATION.md for proofs, guarantees, and reference-only power calculations.
Only the Python standard library is used.  No random sampling is performed.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from math import isqrt
from typing import Protocol


@dataclass(frozen=True)
class Interval:
    lo: Q
    hi: Q

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError("Reversed interval")

    @staticmethod
    def point(x):
        return Interval(Q(x), Q(x))

    def __add__(self, other):
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other):
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def scale(self, x):
        x = Q(x)
        return Interval(min(x*self.lo, x*self.hi), max(x*self.lo, x*self.hi))


class CertifiedLoss(Protocol):
    # Upper bound on integral_0^1 normalized curvature, at most 2.
    variation: Q

    def entropy(self, x: Q) -> Interval: ...
    def slope(self, x: Q) -> Interval: ...


def exp_strictly_above(x: Q, target: Q, max_terms=256):
    """Certify exp(x)>target by a finite sum of its positive Taylor terms."""
    if x < 0:
        raise ValueError("Expected nonnegative exponent")
    term = total = Q(1)
    for j in range(1, max_terms + 1):
        term = term*x/j
        total += term
        if total > target:
            return True
    return False


def sqrt_upper(x: Q, bits=64):
    """Dyadic upper bound, exact even when input is not a square."""
    x = Q(x)
    if x < 0:
        raise ValueError("Negative square root")
    numerator = x.numerator << (2*bits)
    k = isqrt(numerator // x.denominator)
    if k*k*x.denominator < numerator:
        k += 1
    return Q(k, 1 << bits)


def bernstein_interval(n: int, t: Q, c=Q(1269, 250), bits=56):
    """Outward-rounded inversion of n(t-q)^2 <= c(2q(1-q)+2|t-q|/3).

    For empirical t=k/n and any q, exclusion probability <=2 exp(-c).
    Fractional t is also allowed for deterministic power envelopes.
    """
    t, c = Q(t), Q(c)
    if n < 0 or not 0 <= t <= 1 or c <= 0:
        raise ValueError("Invalid Bernstein arguments")
    if n == 0:
        return Interval(Q(0), Q(1))

    def polynomial(q):
        d = abs(t-q)
        return n*d*d - c*(2*q*(1-q) + Q(2, 3)*d)

    if polynomial(Q(0)) <= 0:
        lower = Q(0)
    else:
        outside, inside = Q(0), t
        for _ in range(bits):
            mid = (outside + inside)/2
            if polynomial(mid) > 0:
                outside = mid
            else:
                inside = mid
        lower = outside
    if polynomial(Q(1)) <= 0:
        upper = Q(1)
    else:
        inside, outside = t, Q(1)
        for _ in range(bits):
            mid = (inside + outside)/2
            if polynomial(mid) > 0:
                outside = mid
            else:
                inside = mid
        upper = outside
    return Interval(lower, upper)


def count_rectangle(counts, alpha=Q(1, 20), c=Q(1269, 250), bits=56):
    """counts=[(N_i,K_i)] only; simultaneous coverage >=1-alpha for all theta."""
    if len(counts) != 4 or not 0 < alpha < 1:
        raise ValueError("Expected four states and alpha in (0,1)")
    if not exp_strictly_above(c, Q(8)/alpha):
        raise ValueError("c does not certify 8 exp(-c)<alpha")
    rectangle, center = [], []
    for n, k in counts:
        if not isinstance(n, int) or not isinstance(k, int) or not 0 <= k <= n:
            raise ValueError("Counts must be integers, 0<=K_i<=N_i")
        t = Q(k, n) if n else Q(1, 2)
        center.append(t)
        rectangle.append(bernstein_interval(n, t, c, bits))
    return tuple(rectangle), tuple(center)


def partitions(n=4):
    def rec(i, blocks):
        if i == n:
            yield tuple(tuple(b) for b in blocks)
            return
        for j in range(len(blocks)):
            nxt = [b[:] for b in blocks]
            nxt[j].append(i)
            yield from rec(i+1, nxt)
        yield from rec(i+1, blocks + [[i]])
    return tuple(rec(0, []))


def refines(fine, coarse):
    return all(any(set(b) <= set(c) for c in coarse) for b in fine)


PARTITIONS = partitions()
P2 = tuple(p for p in PARTITIONS if len(p) <= 2)
P3 = tuple(p for p in PARTITIONS if len(p) <= 3)
HIERARCHIES = tuple((a, b) for a in P2 for b in P3 if refines(b, a))


def _validate_p(p):
    p = tuple(map(Q, p))
    if len(p) != 4 or min(p) <= 0 or sum(p) != 1:
        raise ValueError("p must be four strictly positive probabilities")
    return p


def _mean(block, p, values):
    w = sum(p[i] for i in block)
    return sum(p[i]*values[i] for i in block)/w


def fitted_risk(partition, theta, p, loss: CertifiedLoss):
    out = Interval.point(0)
    for block in partition:
        out += loss.entropy(_mean(block, p, theta)).scale(sum(p[i] for i in block))
    return out


def delta_rectangle(rectangle, p, loss: CertifiedLoss, center=None):
    """Enclose Delta on every theta in rectangle, enumerating all 15/39 objects.

    CertifiedLoss's mathematical/numerical contracts are the caller's duty.
    All arithmetic outside that oracle is exact rational arithmetic.
    """
    p = _validate_p(p)
    if len(rectangle) != 4 or any(not 0 <= a.lo <= a.hi <= 1 for a in rectangle):
        raise ValueError("Rectangle must be inside [0,1]^4")
    if not 0 < loss.variation <= 2:
        raise ValueError("Invalid normalized curvature variation bound")
    if center is None:
        center = tuple((a.lo+a.hi)/2 for a in rectangle)
    center = tuple(map(Q, center))
    if any(not a.lo <= t <= a.hi for a, t in zip(rectangle, center)):
        raise ValueError("Center must lie in rectangle")
    radii = tuple(max(t-a.lo, a.hi-t) for a, t in zip(rectangle, center))
    risks = {part: fitted_risk(part, center, p, loss) for part in PARTITIONS}
    unique_blocks = set(b for part in PARTITIONS for b in part)
    slope_ranges = {}
    for block in unique_blocks:
        low = _mean(block, p, [a.lo for a in rectangle])
        high = _mean(block, p, [a.hi for a in rectangle])
        # h' decreases because normalized curvature is nonnegative.
        slope_ranges[block] = Interval(loss.slope(high).lo, loss.slope(low).hi)
    member = {part: tuple(next(b for b in part if i in b) for i in range(4))
              for part in PARTITIONS}
    contrasts = {}
    for a in PARTITIONS:
        for b in PARTITIONS:
            if a == b:
                contrasts[a, b] = Interval.point(0)
                continue
            radius = Q(0)
            for i in range(4):
                ba, bb = member[a][i], member[b][i]
                if ba != bb:
                    diff = slope_ranges[ba] - slope_ranges[bb]
                    slope_bound = min(loss.variation, max(abs(diff.lo), abs(diff.hi)))
                    radius += p[i]*radii[i]*slope_bound
            j = risks[a] - risks[b]
            lo, hi = j.lo-radius, j.hi+radius
            if refines(a, b):
                hi = min(hi, Q(0))
            if refines(b, a):
                lo = max(lo, Q(0))
            contrasts[a, b] = Interval(lo, hi)
    excess = {}
    for budget, flat in ((2, P2), (3, P3)):
        for a in flat:
            excess[budget, a] = Interval(
                max(contrasts[a, b].lo for b in flat),
                max(contrasts[a, b].hi for b in flat))
    h_intervals = [Interval(max(excess[2, a].lo, excess[3, b].lo),
                            max(excess[2, a].hi, excess[3, b].hi))
                   for a, b in HIERARCHIES]
    delta = Interval(min(x.lo for x in h_intervals), min(x.hi for x in h_intervals))
    return {"delta": delta, "hierarchy_bounds": tuple(zip(HIERARCHIES, h_intervals)),
            "contrast_bounds": contrasts, "rectangle": tuple(rectangle)}


def infer_delta(counts, p, loss: CertifiedLoss, null_threshold=Q(1, 2000),
                alpha=Q(1, 20), c=Q(1269, 250)):
    """The actual statistical program: only counts, public p/loss, fixed settings."""
    rectangle, center = count_rectangle(counts, alpha, c)
    result = delta_rectangle(rectangle, p, loss, center)
    result["reject_null"] = result["delta"].lo > null_threshold
    result["alpha"] = alpha
    return result


def report_error_bounds(partition, reports, rectangle, p, loss: CertifiedLoss):
    """Risk of fixed, possibly data-fitted reports minus true population refits.

    Uniform on the same rectangle; summing cell maxima needs no extra alpha.
    reports contains one exact rational report per block, always in [0,1].
    """
    p = _validate_p(p)
    if len(reports) != len(partition):
        raise ValueError("One report per cell is required")
    upper = Q(0)
    for block, q in zip(partition, reports):
        q = Q(q)
        if not 0 <= q <= 1:
            raise ValueError("Report outside full report domain")
        a = _mean(block, p, [x.lo for x in rectangle])
        b = _mean(block, p, [x.hi for x in rectangle])
        endpoint_bounds = [(loss.entropy(q) + loss.slope(q).scale(t-q)
                            - loss.entropy(t)).hi for t in (a, b)]
        bound = min(Q(1), max(endpoint_bounds),
                    loss.variation*max(abs(a-q), abs(b-q)))
        upper += sum(p[i] for i in block)*max(Q(0), bound)
    return Interval(Q(0), upper)


def power_envelope(theta_reference, p, m=10**6, beta_count=Q(1, 10),
                   beta_labels=Q(1, 10), c_count=Q(3689, 1000),
                   c_labels=Q(4383, 1000), c_confidence=Q(1269, 250)):
    """REFERENCE ONLY: enclose the random confidence rectangle with >=1-beta.

    This function is never called by infer_delta.  Exact theta is allowed here
    only to prove power at a separately fixed candidate, before label sampling.
    """
    p = _validate_p(p)
    theta = tuple(map(Q, theta_reference))
    if len(theta) != 4 or any(not 0 <= x <= 1 for x in theta):
        raise ValueError("Reference posterior outside [0,1]^4")
    if not exp_strictly_above(c_count, Q(4)/beta_count):
        raise ValueError("Uncertified count failure budget")
    if not exp_strictly_above(c_labels, Q(8)/beta_labels):
        raise ValueError("Uncertified label failure budget")
    n_min, estimator_ranges, outer, deviations, ci_radii = [], [], [], [], []
    for pi, t in zip(p, theta):
        # P(N_i < m*pi-sqrt(2*m*pi*c_count)) <= exp(-c_count).
        cutoff = m*pi - sqrt_upper(2*m*pi*c_count)
        n = max(0, (cutoff.numerator + cutoff.denominator - 1)//cutoff.denominator)
        n_min.append(n)
        if n == 0:
            estimator_ranges.append(Interval(Q(0), Q(1)))
            outer.append(Interval(Q(0), Q(1)))
            deviations.append(Q(1))
            ci_radii.append(Q(1))
            continue
        # Endpoints are deterministic conditional on N_i, so exploit them.
        if t in (0, 1):
            radius = Q(0)
        else:
            radius = sqrt_upper(2*t*(1-t)*c_labels/n) + 2*c_labels/(3*n)
        possible = Interval(max(Q(0), t-radius), min(Q(1), t+radius))
        deviations.append(max(t-possible.lo, possible.hi-t))
        estimator_ranges.append(possible)
        # A separately rounded data CI can protrude beyond the rounded
        # envelope CI: do not assume bisection's rounding is monotone in t/n.
        # Enlarge by its full maximum outward error to cover that protrusion.
        rounding_pad = Q(1, 1 << 56)
        lo = max(Q(0), bernstein_interval(n, possible.lo, c_confidence).lo-rounding_pad)
        hi = min(Q(1), bernstein_interval(n, possible.hi, c_confidence).hi+rounding_pad)
        outer.append(Interval(lo, hi))
        vmax = Q(1, 4) if lo <= Q(1, 2) <= hi else max(lo*(1-lo), hi*(1-hi))
        offset = c_confidence/(3*n)
        # Add the outward bisection error of bernstein_interval (56 steps).
        ci_radii.append(min(Q(1), offset + sqrt_upper(offset*offset +
                          2*c_confidence*vmax/n) + Q(1, 1 << 56)))
    return {"rectangle": tuple(outer), "n_min": tuple(n_min),
            "estimator_ranges": tuple(estimator_ranges),
            "estimation_deviations": tuple(deviations),
            "confidence_radii": tuple(ci_radii),
            "power_lower_if_uniform_rejection": 1-beta_count-beta_labels}


def certify_power(theta_reference, p, loss: CertifiedLoss, m=10**6,
                  null_threshold=Q(1, 2000), entropy_width=Q(0), slope_width=Q(0)):
    """Reference-only certificate for THE ACTUAL infer_delta output.

    entropy_width and slope_width must be uniform upper bounds on the widths
    returned by the loss oracle over [0,1] (or a certified domain enclosing
    all calls).  They are proof obligations, not values inferred from a few calls.
    Merely bounding Delta over an outer rectangle does NOT certify that this
    possibly loose, moving-center inference implementation rejects.
    """
    p = _validate_p(p)
    theta = tuple(map(Q, theta_reference))
    env = power_envelope(theta, p, m)
    rect = env["rectangle"]
    risks = {part: fitted_risk(part, theta, p, loss) for part in PARTITIONS}
    slopes = {}
    for block in set(b for part in PARTITIONS for b in part):
        lo = _mean(block, p, [x.lo for x in rect])
        hi = _mean(block, p, [x.hi for x in rect])
        slopes[block] = Interval(loss.slope(hi).lo, loss.slope(lo).hi)
    member = {part: tuple(next(b for b in part if i in b) for i in range(4))
              for part in PARTITIONS}
    lower_contrasts = {}
    for a in PARTITIONS:
        for b in PARTITIONS:
            if a == b:
                lower_contrasts[a, b] = Q(0)
                continue
            penalty = 2*entropy_width
            for i in range(4):
                ba, bb = member[a][i], member[b][i]
                if ba != bb:
                    ds = slopes[ba]-slopes[bb]
                    M = min(loss.variation, max(abs(ds.lo), abs(ds.hi)))
                    M_actual = min(loss.variation, M+2*slope_width)
                    penalty += p[i]*(M*env["estimation_deviations"][i] +
                                    M_actual*env["confidence_radii"][i])
            lower = (risks[a]-risks[b]).lo - penalty
            if refines(b, a):
                lower = max(Q(0), lower)
            lower_contrasts[a, b] = lower
    excess = {(k, a): max(lower_contrasts[a, b] for b in flat)
              for k, flat in ((2, P2), (3, P3)) for a in flat}
    hierarchy_lower = [(a, b, max(excess[2, a], excess[3, b]))
                       for a, b in HIERARCHIES]
    lower_output = min(x[2] for x in hierarchy_lower)
    return {"certified": lower_output > null_threshold,
            "inference_lower_bound_on_power_event": lower_output,
            "power_lower_if_certified": env["power_lower_if_uniform_rejection"],
            "hierarchy_lower_bounds": tuple(hierarchy_lower), "envelope": env,
            "oracle_entropy_width_assumption": entropy_width,
            "oracle_slope_width_assumption": slope_width}


class PositiveCoefficientPolynomialLoss:
    """Small exact verification oracle; raw g has nonnegative coefficients.

    High-degree needle losses should supply their own certified compact-form
    oracle.  This class is intentionally not a conversion of candidate truth.
    """
    def __init__(self, raw_coefficients):
        coefficients = tuple(map(Q, raw_coefficients))
        if not coefficients or min(coefficients) < 0 or not any(coefficients):
            raise ValueError("Need nonzero polynomial with nonnegative coefficients")
        a0 = sum(v/Q(j+2) for j, v in enumerate(coefficients))
        a1 = sum(v/Q((j+1)*(j+2)) for j, v in enumerate(coefficients))
        self.amplitude = max(a0, a1)
        self.coefficients = tuple(v/self.amplitude for v in coefficients)
        self.variation = (a0+a1)/self.amplitude

    def _outcomes(self, x):
        x = Q(x)
        l0 = sum(v*x**(j+2)/(j+2) for j, v in enumerate(self.coefficients))
        l1 = sum(v*(Q(1, (j+1)*(j+2))-x**(j+1)/(j+1)
                    +x**(j+2)/(j+2)) for j, v in enumerate(self.coefficients))
        return l0, l1

    def entropy(self, x):
        l0, l1 = self._outcomes(x)
        return Interval.point((1-x)*l0+x*l1)

    def slope(self, x):
        l0, l1 = self._outcomes(x)
        return Interval.point(l1-l0)


if __name__ == "__main__":
    print("Pure functions only; run verify_exact.py for deterministic checks.")

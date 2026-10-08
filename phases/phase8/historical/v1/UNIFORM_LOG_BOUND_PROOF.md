# Uniform logarithmic improvement: theorem and proof

7 October 2026. Purely analytic research proof. The note records the analytic argument and explicit constants. No numerical experiment is used. This file does not alter the delivered v0 paper. Historical novelty and external peer review are not established.

## Statement

There is an absolute constant C such that for every r >= 1, every nonzero polynomial curvature g >= 0 of degree at most r on an interval, and every four distinct equally weighted posterior locations, the optimal deterministic two/three-output hierarchy price R satisfies

    R <= C (r+1)/log(r+2).

The same upper bound holds for stochastic hierarchies. Combined with the already-audited fixed-quartet Chebyshev lower construction, this establishes sharp order Theta(r/log r) for both degree envelopes. This is a mathematical-order statement, not a historical novelty claim.

A completely explicit intermediate inequality established below is: if R >= 4096, then, with n = r+4 and epsilon = 64/R,

    4 n epsilon >= (1/(100 pi)) log(1/epsilon) - log(512).        (1)

Constants are intentionally loose.

## 1. Two compatible candidate hierarchies give both needed estimates

The existing v0 contiguity/tie analysis is retained. Unless compatible flat optima exist (and R=1), the only conflict is the sorted fine pair BC and coarse split AB|CD.

Write

    q = P_BC,
    Q = P_AB + P_CD,
    T = T_ABC.

A hierarchy using the flat-optimal coarse split AB|CD and its better outer fine pair has price min(P_AB,P_CD)/q. A hierarchy using the flat-optimal fine pair BC and its better adjacent triple coarsening has price min(T_ABC,T_BCD)/Q. These are admissible candidates, irrespective of whether other noncontiguous coarsenings improve the actual optimum. Consequently, actual optimal price R satisfies

    R <= min(P_AB,P_CD)/q,
    R <= min(T_ABC,T_BCD)/Q.

In particular,

    q <= T/R^2,                       (2)
    S := P_AB + P_BC <= 2T/R.         (3)

Indeed P_AB <= Q <= T/R, and q <= P_AB/R <= T/R^2. Thus S <= (1+1/R)T/R <= 2T/R.

Affine-normalize the TRIPLE hull [A,C] to [-1,1], keeping B=b in (-1,1). The fourth point D may lie beyond 1; it is not used in subsequent polynomial estimates. Degree, nonnegativity on the triple hull, and all the displayed cost relations are preserved by this affine change of variables. Put

    w(t) = 1-|t|,
    w_b = w(b),
    s = sqrt(1-b^2),
    theta_b = arccos b,
    mu = integral_{-1}^1 w(t)g(t) dt.

The triple curvature kernel K_ABC obeys, pointwise,

    w(t)/4 <= K_ABC(t) <= w(t)/2.     (4)

The lower bound follows because a triple Jensen gap dominates the constituent endpoint-pair gap K_AC=w/4; apply the centroid/Jensen decomposition to hinge generators. The upper bound follows because the kernel vanishes at both endpoints and each piecewise slope has magnitude <=1/2. Thus

    2T <= mu <= 4T.                  (5)

The sum of the two adjacent-pair kernels on this hull is exactly

    (1/4) min{t+1, |t-b|, 1-t}
      = (1/4) min{w(t), |t-b|}.      (6)

## 2. A high peak close to the inner knot

Assume R>=4096, and set

    epsilon = 64/R <= 1/64,
    I = {t: |t-b| <= epsilon w_b}.

This interval is contained in [-1,1]. Outside I, the 1-Lipschitz property of w gives

    |t-b| >= epsilon w(t)/(1+epsilon).

Equations (3) and (6) therefore imply

    integral_{[-1,1]\I} w g
      <= 4(1+epsilon)S/epsilon
      <= 8(1+epsilon)T/(R epsilon)
      = (1+epsilon)T/8
      <= T/4.

By (5), integral_I wg >= T. Since I has length 2 epsilon w_b, some t_0 in I satisfies

    w(t_0)g(t_0) >= T/(2 epsilon w_b).

Also w(t_0)>=(1-epsilon)w_b. Define the nonnegative trigonometric polynomial

    F(theta)=g(cos theta) sin^4 theta,
    n=r+4,
    theta_0=arccos t_0.

It has trigonometric degree at most n. Since (1-t^2)^2 >= w(t)^2,

    F(theta_0) >= (1-epsilon)T/(2epsilon)
                 >= T/(4epsilon).                         (7)

Set

    h = epsilon s,
    a = 1-h.

Here h<=1/64 and a>=63/64. For every t between b and t_0,

    |(1-t^2)-s^2| <= 2|t-b| <= 2epsilon w_b <= 2epsilon s^2.

Thus sin(arccos t)>=s/2, and

    |theta_0-theta_b| <= 2epsilon w_b/s <= 2h,
    sin theta_0 <= 2s.                                     (8)

The proof will use the Poisson measure centered at a exp(i theta_0):

    dnu(theta) = P_a(theta-theta_0) dtheta/(2pi),
    P_a(u) = (1-a^2)/(1-2a cos u+a^2).

It is a probability measure on the circle.

## 3. Uniform bound for the full Poisson average

For circular distance d=dist(u,2pi Z) in [0,pi],

    P_a(u) <= 8h/(h^2+d^2).                                (9)

Indeed 1-a^2<=2h and 1-2a cos u+a^2=h^2+2a(1-cos u), where 1-cos u>=2d^2/pi^2 and 4a/pi^2>1/4.

For theta in [0,pi], either of the circular distances d between theta and +theta_0 or between theta and -theta_0 satisfies

    sin theta <= sin theta_0+d <= 2s+d.

Using (9),

    P_a(theta +/- theta_0) sin theta
      <= 16hs/(h^2+d^2)+8hd/(h^2+d^2)
      <= 16s/h+4
      <= 20/epsilon.                                      (10)

Because F is even and dt=-sin(theta)dtheta,

    integral F dnu
      = (1/(2pi)) integral_{-1}^1
          [P_a(theta-theta_0)+P_a(theta+theta_0)]
          g(t)(1-t^2)^(3/2) dt,
      theta=arccos t.

Since (1-t^2)^(3/2)/w(t)=(1+|t|)sin theta<=2sin theta, (10) and (5) show

    integral F dnu <= (80/(2pi epsilon)) mu
                    <= (160/pi)T/epsilon
                    <= 64T/epsilon.                       (11)

This is the spacing-uniform step: the full unweighted integral of F need NOT be comparable to the mass near B. Instead the Poisson kernel supplies the correct local angular scale even as B approaches either endpoint.

## 4. The inward arc has fixed harmonic mass and small curvature average

Take the arc pointing from B toward C,

    J = [theta_b-5h, theta_b-4h].

It lies in (0,pi): theta_b>=s and 5h<=5epsilon theta_b<theta_b. On J, the sine function is 1-Lipschitz, so

    (1-5epsilon)s <= sin theta <= (1+5epsilon)s <= 2s.

Writing t=cos theta, it follows that

    t-b >= 4h(1-5epsilon)s >= 3epsilon s^2 >= 3epsilon w_b,
    t-b <= 5h(1+5epsilon)s <= 6epsilon s^2 <= 12epsilon w_b.

Consequently

    1-t >= (1-b)-12epsilon w_b >= (1-12epsilon)w_b
          >= 3epsilon w_b.

The BC pair kernel on the image of J is therefore at least 3epsilon w_b/4. By (2),

    integral_{cos J} g(t) dt <= 4q/(3epsilon w_b).

Hence

    integral_J F(theta)dtheta
      = integral_{cos J} g(t)(1-t^2)^(3/2)dt
      <= 8s^3 * 4q/(3epsilon w_b)
      <= (64/(3epsilon))s q,                               (12)

where s^2/w_b=1+|b|<=2. Also P_a<= (1+a)/(1-a)<=2/h, so

    integral_J F dnu
      <= (64/(3pi epsilon^2))q
      <= (64/(3pi R^2 epsilon^2))T
      = T/(192pi)
      <= T.                                               (13)

Let alpha=nu(J). From (8), |theta-theta_0|<=7h throughout J. The denominator of P_a is at most h^2+(7h)^2=50h^2, while its numerator is h(1+a)>=h. Because |J|=h,

    alpha >= 1/(100pi)=:alpha_0.                           (14)

## 5. Outer factor and Poisson/Jensen force the logarithm

By the Fejer-Riesz factorization, there is an algebraic polynomial p of degree m<=n, with no zeros in the open unit disk, such that

    F(theta)=|p(exp(i theta))|^2.

Zeros on the unit circle are allowed. One may choose the zero-free factor by first removing any monomial factor z^k, whose boundary modulus is one, and then reflecting each nonzero interior zero in a spectral factor across the unit circle, which changes its modulus only by a constant absorbed into its normalization. For each root zeta with |zeta|>=1 and 0<a<1,

    |a z-zeta| >= a|z-zeta|,  |z|=1.                       (15)

Indeed the difference of the squares is

    (1-a)[(1+a)|zeta|^2-2a Re(z conjugate(zeta))]>=0.

Multiplying over the roots gives

    |p(a exp(i theta_0))| >= a^m |p(exp(i theta_0))|
                           >= a^n |p(exp(i theta_0))|.

Since log|p| is harmonic inside the disk, with integrable logarithmic boundary singularities at any unit-circle roots, its Poisson formula gives

    log F(theta_0)+2n log a <= integral log F dnu.           (16)

Apply Jensen's inequality separately on J and its complement. Equations (11), (13), and the entropy bound -alpha log alpha-(1-alpha)log(1-alpha)<=log 2 yield

    integral log F dnu
      <= alpha log(T/alpha)
        +(1-alpha)log(64T/((1-alpha)epsilon))
      <= log(T/epsilon)+alpha log epsilon+log128
      <= log(T/epsilon)+alpha_0 log epsilon+log128.          (17)

The zeros of F have zero arc measure and cause no issue; log F is integrable. From (7), (16), and (17),

    2n[-log(1-h)] >= alpha_0 log(1/epsilon)-log512.

As -log(1-h)<=2h<=2epsilon, this proves (1).

## 6. Conversion to the degree envelope

Let

    K = 64 * 512^(100pi),
    A = 25600pi.

Equation (1) is equivalently

    R log(R/K) <= A(r+4),       whenever R>=4096.            (18)

For R>=max(4096,K^2), log(R/K)>=0.5 log R, giving R log R<=2A(r+4). This implies R=O(r/log r). For an elementary split: if R<=sqrt(r+4), the desired bound already holds for large r; otherwise log R>=0.5 log(r+4), so R<=4A(r+4)/log(r+4). All fixed finite thresholds are absorbed into a universal constant. The stochastic price is bounded by the deterministic one because every deterministic hierarchy is admissible stochastically.

Together with the already-audited lower construction, the conclusion is

    C_r^det = Theta(r/log r),
    C_r^stoch = Theta(r/log r),

uniformly over all four distinct equal-mass posterior configurations in the upper bound, with the lower bound already witnessed by one fixed equally spaced quartet.

## Technical verification checklist

1. Check triple-hull normalization and the two candidate hierarchy inequalities without assuming an exact deterministic Pareto formula.
2. Check the uniform Poisson average (11), including the reflected angular contribution near both endpoints.
3. Check inward-arc inclusion and leakage estimate (13).
4. Check outer-factor direction, boundary zeros, and Poisson/Jensen signs.
5. Check that no hidden posterior-gap lower bound entered any step.

## Verified primary source for the classical factorization

Michael A. Dritschel and Hugo J. Woerdeman, “Outer factorizations in one and several variables,” arXiv:math/0403099 (2004), introduction, first paragraph, PDF page 1: https://arxiv.org/pdf/math/0403099 . The exact scalar theorem states that a nonnegative trigonometric polynomial of degree at most n has a modulus-square factor of algebraic degree at most n, which can be chosen with no roots in the open unit disk. This includes possible unit-circle roots. The source was directly inspected on 7 October 2026. The radial comparison (15) and Poisson/Jensen application are proved above rather than attributed to that article.

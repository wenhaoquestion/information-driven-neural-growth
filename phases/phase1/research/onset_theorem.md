# Cardinality-sensitive information-bottleneck onset

Proposed mathematical research draft, 2026-09-13. This auxiliary result was not folded into the main manuscript; correctness and novelty review are less complete.

## 1. Statement and notation

Let the source alphabet have size n >= 2. Let p be strictly positive and W be a stochastic matrix. Delete output symbols of zero pW probability, so r = pW is also strictly positive. Every logarithm is natural. On the probability simplex define

\[
A(q)=D(q\Vert p),\qquad B(q)=D(qW\Vert r),\qquad
\eta=\sup_{q\ne p}\frac{B(q)}{A(q)}.
\]

On the tangent space V = {v: sum_x v_x = 0} define

\[
G(v)=\sum_x v_x^2/p_x,\quad H(v)=\sum_y (vW)_y^2/r_y,
\quad \rho^2=\max_{v\in V\setminus\{0\}}H(v)/G(v).
\]

Assume throughout that eta > rho^2. Set

\[
Q(v)=\eta G(v)-H(v),\qquad L(q)=\eta A(q)-B(q),
\quad \mathcal M=\{q\ne p:L(q)=0\}.
\]

Then Q is positive definite on V and M is a nonempty compact set separated from p. Define

\[
\mathcal D=\left\{\frac{q-p}{A(q)}:q\in\mathcal M\right\},\qquad
K_k=\left\{\sum_{i=1}^k\alpha_i d_i:
 \alpha_i\ge0,\ \sum_i\alpha_i=1,\ d_i\in\mathcal D\right\},
\quad \kappa_k=\min_{d\in K_k}Q(d).
\]

Zero coefficients allow fewer than k points. Write K = conv(D), kappa = min_K Q. The sets K_k and K are compact. Since dim(V)=n-1, Caratheodory's theorem gives K_n=K.

For integer k >= 1 let

\[
F_k(R)=\sup\{I(Y;T):Y-X-T,\ I(X;T)\le R,\ |T|\le k+1\}.
\]

Let F(R) have no cardinality restriction (arbitrary finite output alphabet suffices).

**Theorem (second-order onset).** Under the strict-gap assumption,

\[
F_k(R)=\eta R-\frac{\kappa_k}{2}R^2+o(R^2),\qquad
F(R)=\eta R-\frac{\kappa}{2}R^2+o(R^2)
\quad (R\downarrow0).
\]

The statement gives a value expansion, without claiming differentiability of an optimizer, a smooth encoder branch, or uniqueness of a maximizing posterior. It permits an arbitrary compact maximizer set M.

## 2. Complete proof

### 2.1 Local expansions and compactness

For v in V sufficiently small, strict positivity of p and r permits Taylor expansion with uniform remainders:

\[
A(p+v)=\tfrac12G(v)+O(\|v\|^3),\quad
B(p+v)=\tfrac12H(v)+O(\|v\|^3),\quad
L(p+v)=\tfrac12Q(v)+O(\|v\|^3).
\tag{1}
\]

Because Q >= (eta-rho^2)G, it is positive definite on V. In particular there are delta_0,c,C>0 such that, for ||q-p|| <= delta_0,

\[
L(q)\ge c A(q),\quad A(q)\le C L(q),\quad
L(q)\ge \bigl(\tfrac12-\epsilon(\delta)\bigr)Q(q-p)
\quad\text{when }\|q-p\|\le\delta\le\delta_0,
\tag{2}
\]

where epsilon(delta) tends to zero. Also limsup_{q->p} B(q)/A(q) <= rho^2. The ratio is continuous off p, so its strictly larger supremum eta is attained on a compact subset bounded away from p. Thus M has all the asserted properties. Continuity of q -> (q-p)/A(q) on M proves compactness of D and K_k; finite-dimensional convex hull compactness proves compactness of K.

### 2.2 Exact posterior formulation

Every encoder is represented by posterior atoms q_t=P(X|T=t) with weights w_t=P(T=t), satisfying

\[
\sum_t w_tq_t=p,\quad
I(X;T)=\sum_t w_tA(q_t),\quad
I(Y;T)=\sum_t w_tB(q_t).
\tag{3}
\]

Conversely every probability measure on the simplex with barycenter p defines an encoder by P(T=t|X=x)=w_tq_t(x)/p_x. Its rows sum to one, and nonnegative entries with row sum one are automatically <=1. Thus (3) is an exact equivalence, including boundary posterior atoms. The source/output alphabets are finite, so all quantities are finite.

### 2.3 Achievability with k rare atoms and one background atom

Take any d=sum_i alpha_i(q_i-p)/A(q_i) in K_k. Put

\[
s=\sum_i\frac{\alpha_i}{A(q_i)},\qquad
w_i(t)=t\frac{\alpha_i}{A(q_i)},\quad
w_0(t)=1-ts,\qquad
q_0(t)=p-\frac{t}{1-ts}d.
\tag{4}
\]

For all sufficiently small t>=0, q_0(t) is a valid strictly positive posterior, w_0(t)>0, and these weights/posteriors have barycenter p. Their source information J(t) and deficit E(t) are

\[
J(t)=t+(1-ts)A(q_0(t))
 =t+\tfrac12G(d)t^2+O(t^3),
\]
\[
E(t)=\sum_{i=0}^kw_i(t)L(q_i(t))
 =(1-ts)L(q_0(t))
 =\tfrac12Q(d)t^2+O(t^3).
\tag{5}
\]

Here every rare atom has L(q_i)=0. J is smooth near zero with J'(0)=1. By the inverse function theorem, every sufficiently small R>0 equals J(t(R)) for t(R)=R+O(R^2). Consequently the output information is

\[
\eta R-E(t(R))=\eta R-\tfrac12Q(d)R^2+O(R^3).
\tag{6}
\]

Choose a minimizer of Q on K_k. This gives the lower bound on F_k and supplies a uniform O(R^2) upper bound on its loss relative to eta R. Choosing a minimizer on K and representing it by at most n points yields the unrestricted lower bound with at most n+1 labels.

### 2.4 Converse for arbitrary encoders

Fix a sequence R_j down to zero and posterior measures mu_j of barycenter p with source information a_j<=R_j. Define their total loss from the linear bound by

\[
D_j=\eta R_j-\int B\,d\mu_j
 =\eta(R_j-a_j)+\int L\,d\mu_j.
\tag{7}
\]

It is enough to consider sequences with D_j=O(R_j^2): otherwise the desired lower bound on liminf D_j/R_j^2 is automatic after a subsequence. For near optimizers such boundedness follows from (6). Nonnegativity of both terms in (7) gives

\[
a_j/R_j\to1,\qquad \int L\,d\mu_j=O(R_j^2).
\tag{8}
\]

Fix delta in (0,delta_0) so M lies outside U_delta={q:||q-p||<delta}; set C_delta=Delta_n\U_delta, a compact set. Since A is bounded below by a positive constant on C_delta,

\[
\mu_j(C_\delta)=O(R_j),\qquad \mu_j(U_\delta)\to1.
\tag{9}
\]

By the second inequality in (2),

\[
\int_{U_\delta} A\,d\mu_j=O(R_j^2).
\tag{10}
\]

Define a finite measure on C_delta by

\[
d\nu_j(q)=\frac{A(q)}{R_j}\,d\mu_j(q).
\tag{11}
\]

Equations (8) and (10) imply nu_j(C_delta)->1. Along a subsequence it converges weakly to a probability measure nu on C_delta. Since L/A is continuous and nonnegative there,

\[
\int_{C_\delta}\frac{L(q)}{A(q)}\,d\nu_j(q)
 =\frac1{R_j}\int_{C_\delta}L\,d\mu_j=O(R_j)\to0.
\tag{12}
\]

Hence nu is supported on M. Continuity of (q-p)/A(q) on C_delta now gives

\[
\frac1{R_j}\int_{C_\delta}(q-p)\,d\mu_j
 \longrightarrow d:=\int_{\mathcal M}\frac{q-p}{A(q)}\,d\nu(q)\in K.
\tag{13}
\]

The barycenter constraint makes the corresponding integral over U_delta equal to the negative of the one over C_delta. Jensen's inequality for the positive quadratic form Q, applied to the conditional law on U_delta, gives

\[
\int L\,d\mu_j
\ge\left(\tfrac12-\epsilon(\delta)\right)
 \frac{Q\left(\int_{U_\delta}(q-p)\,d\mu_j\right)}{\mu_j(U_\delta)}.
\tag{14}
\]

Dividing by R_j^2 and using (9) and (13),

\[
\liminf_j\frac{D_j}{R_j^2}
\ge\left(\tfrac12-\epsilon(\delta)\right)Q(d)
\ge\left(\tfrac12-\epsilon(\delta)\right)\kappa.
\tag{15}
\]

To be entirely explicit about subsequences: start with a subsequence realizing the liminf of D_j/R_j^2, and then extract the weakly convergent subsequence above; the realized liminf is unchanged. The resulting bound holds for each fixed sufficiently small delta. Let delta decrease to zero. This proves liminf >=kappa/2. Together with (6), it proves the unrestricted expansion.

### 2.5 Cardinality converse

If each mu_j has at most k+1 atoms, (9) ensures at least one lies in U_delta for all large j. Thus nu_j has at most k atoms. Weak limits on a compact space of measures supported on at most k points have at most k support points: enumerate and pad the lists of locations and masses, take subsequences for all k locations and masses, and discard zero limiting masses. Hence the measure nu in (13) is supported on at most k points of M, and d lies in K_k. Replace kappa by kappa_k in (15). This proves the cardinality-constrained expansion. No fixed choice of rare labels across j is assumed.

## 3. Exact linearity and support threshold

For fixed k>=1, the following are equivalent:

1. 0 belongs to K_k.
2. kappa_k=0.
3. F_k(R)=eta R for all sufficiently small positive R.
4. There is a sequence R_j down to zero with F_k(R_j)=eta R_j.

Equivalence of 1 and 2 uses positive definiteness and compactness. If d=0 in construction (4), then q_0(t)=p, J(t)=t, and E(t)=0 exactly; this proves 1 implies 3. Implication 3 to 4 is immediate. Implication 4 to 2 follows from the expansion. The unrestricted analogue replaces K_k by K.

Exact linearity is not asserted as a new phenomenon. The new candidate result is the full curvature and its dependence on restricted cardinality, for arbitrary maximizing sets and without smooth optimizer assumptions.

## 4. Symmetric-channel explicit cardinality penalty

Let n>=3, p=u=(1/n,...,1/n), and

\[
W=aI+(1-a)\mathbf1u,\qquad 0<a<1.
\]

Thus qW=aq+(1-a)u. On V, rho^2=a^2 and Q(v)=n(eta-a^2)||v||_2^2.

### 4.1 Strict separation a^2<eta<a

For any nonzero v with sum v=0, Taylor expansion gives

\[
\frac{B(u+tv)}{A(u+tv)}
=a^2\left[1+\frac{n(1-a)}3
 \frac{\sum_i v_i^3}{\sum_i v_i^2}t+O(t^2)\right].
\tag{16}
\]

Take v=e_1-u. Then sum v_i^3=(n-1)(n-2)/n^2>0, so eta>a^2. Strict convexity of KL in its first argument gives B(q)<a A(q) for every q!=u. A maximizing q exists by the strict gap; applying that inequality at a maximizer yields eta<a.

### 4.2 All maximizing directions are positive simplex rays

Set b=(1-a)/n and

\[
f(t)=\eta t\log(nt)-(at+b)\log(n(at+b)),\quad t\in[0,1].
\]

Then L(q)=sum_i f(q_i). Each q in M is a global minimum of L over the simplex. It is interior: if q_i=0 and q_j>0, transferring epsilon from j to i changes L by eta epsilon log epsilon+O(epsilon)<0, a contradiction. For positive t,

\[
f''(t)=\frac\eta t-\frac{a^2}{at+b}
=\frac{a(\eta-a)t+\eta b}{t(at+b)}.
\tag{17}
\]

Since eta<a, the numerator decreases strictly with t and changes sign at most once, from positive to negative. Thus f' first increases strictly and then decreases strictly. Lagrange stationarity requires all f'(q_i) to be the same, so there are at most two distinct coordinate values. If all equal, q=u, excluded from M. Therefore there are exactly two, a low value on the strictly convex branch and a high value on the strictly concave branch. There cannot be two high coordinates: varying those two in opposite directions gives negative second variation 2 f''(high), contradicting minimality. Consequently

\[
q=q^{(i)}(z):=u+z(e_i-u),\qquad 0<z<1.
\tag{18}
\]

The equality case at the inflection cannot invalidate this argument: if a stationary level equals the maximum of f', there is only one root; if the level is below that maximum, the two roots lie strictly on opposite sides.

Let Z be the compact nonempty set of z values for which q^(1)(z) lies in M. Permutation invariance implies

\[
\mathcal D=\{c(z)(e_i-u):z\in Z,\ 1\le i\le n\},
\quad c(z)=z/A(q^{(1)}(z)),\quad c_* =\min_{z\in Z}c(z)>0.
\tag{19}
\]

This proof does not need uniqueness of z. Existing Hamming-channel results in Benger--Asoodeh--Chen 2023 can alternatively be used to identify a unique maximizing z through their scalar tangency characterization.

### 4.3 Explicit minimization over at most k rays

For 1<=k<n,

\[
\min_{d\in K_k}\|d\|_2^2=c_*^2\left(\frac1k-\frac1n\right),
\qquad
\kappa_k=n(\eta-a^2)c_*^2\left(\frac1k-\frac1n\right).
\tag{20}
\]

Proof: combine repeated ray types into a single term. Their effective coefficient is a convex average of their original c values and remains >=c_*. Suppose j<=k distinct ray types occur, with coefficients c_i>=c_* and convex weights alpha_i. Write w_i=alpha_i c_i and h_i=1/c_i, so h^T w=1 and w>=0. Then

\[
\|d\|_2^2=w^T M_jw,\qquad M_j=I_j-\mathbf1\mathbf1^T/n.
\]

M_j is positive definite and M_j^{-1}=I_j+11^T/(n-j), which has nonnegative entries. Cauchy--Schwarz in the M_j metric yields

\[
w^TM_jw\ge\frac1{h^TM_j^{-1}h}
\ge\frac1{c_*^{-2}\,jn/(n-j)}
=c_*^2\left(\frac1j-\frac1n\right)
\ge c_*^2\left(\frac1k-\frac1n\right).
\]

The second inequality uses 0<h_i<=1/c_* and entrywise nonnegativity of M_j^{-1}. Equality is attained by taking k distinct rays, all c_i=c_*, and alpha_i=1/k. This proves (20). For k>=n, equal weights on all n directions give kappa_k=0.

Therefore the exact low-rate penalty for using at most k+1 labels, 1<=k<n, is

\[
F(R)-F_k(R)
=\frac{n(\eta-a^2)c_*^2}{2}
\left(\frac1k-\frac1n\right)R^2+o(R^2).
\tag{21}
\]

Here F is exactly eta R on some initial interval; that qualitative fact and necessity of n+1 labels are known from Benger--Asoodeh--Chen. Formula (21) is the proposed quantitative refinement. It shows all alphabets >=2 share the optimal first derivative, while missing simultaneous rare posterior types imposes a strict second-order cost.

## 5. Adversarial checks and limitations

- Strict eta>rho^2 is decisive. Without it Q can be singular and M can approach p, invalidating the fixed near/far decomposition; nonquadratic powers or other behavior are possible.
- The theorem concerns stochastic finite representations with unrestricted encoder access to X. It does not by itself establish realizability or learning dynamics of neural units.
- Cardinality counts posterior symbols. Calling every symbol a physical neuron would overstate the model.
- Budgets are <=R; proof (7)-(8) explicitly handles unsaturated budgets.
- No curvature contribution from drift of rare maximizing posteriors is missing in the converse: all off-M deficits are nonnegative; near-optimal cost-weighted laws concentrate on M. Achievability already attains the lower bound with fixed rare atoms.
- The R^2 term uses the Q norm of the averaged direction, not the average Q norm. Replacing one by the other would incorrectly discard cooperation among births.
- Requiring at most k nontrivial atoms before adding a background atom is essential in the cardinality argument. At small rates a background atom is forced, not freely assumed.
- The symmetric example is deliberately restricted to 0<a<1 (positive correlation). Benger et al. cover a broader Hamming parameter regime; no such broader extension is claimed here.

## 6. Closest primary references inspected

1. Ngampruetikorn and Schwab, *Perturbation Theory for the Information Bottleneck*, NeurIPS 2021, https://papers.nips.cc/paper/2021/file/af8d9c4e238c63fb074b44eb6aed80ae-Paper.pdf. The historical source review inspected equations concerning rare-state perturbations and a single emergent posterior. In the paragraph between (13) and (14), the paper takes the normalized rare-state likelihood ratio to be independent of the state index. Equations (23)-(27) then give a scalar second-order coefficient under a positive-curvature assumption. Equation (13) by itself is a nonlinear fixed point condition and does not establish uniqueness across state labels. Our symmetric construction has multiple optimizing posteriors and their averaged direction is zero; replacing them by a single posterior would give a strictly positive coefficient. This draft must still be compared against the complete paper/supplement before making a broad novelty claim. The present candidate differs through a rigorous global value expansion, multiple maximizing directions, and fixed-cardinality penalties.
2. Benger, Asoodeh, Chen, *The Cardinality Bound on the Information Bottleneck Representations is Tight*, ISIT 2023, pp.1478-1483, DOI 10.1109/ISIT54713.2023.10206791; arXiv:2305.07000v2. The historical source review inspected the full parsed first three pages and relevant theorem statements. Theorem 3 proves n+1 necessary at low positive rate for uniform Hamming channels; Theorem 2 identifies their IB function with a concave envelope; Lemma 1 (credited to their reference 9, Lemma 7) identifies one-high-coordinate entropy optimizers; Section IV/Lemma 2 establishes scalar convex/concave shape. These qualitative results must not be claimed as new. Their Introduction explicitly separates exact cardinality from approximate cardinality results, citing their reference 7.
3. Agmon, *The Information Bottleneck's Ordinary Differential Equation: First-Order Root Tracking for the Information Bottleneck*, Entropy 25(10):1370 (2023), https://www.mdpi.com/1099-4300/25/10/1370. Inspected Section 5.3, Proposition 1 and Theorem 6: enough clusters are necessary to detect certain bifurcations; discontinuous bifurcations yield linear segments. These precede broad claims about structural births and linearity.
4. Hirche and Winter, *An Alphabet-Size Bound for the Information Bottleneck Function*, ISIT 2020, pp.2383-2388, DOI 10.1109/ISIT44484.2020.9174416. Institutional metadata and abstract inspected at https://portalrecerca.uab.cat/en/publications/an-alphabet-size-bound-for-the-information-bottleneck-function/ . The result concerns uniform additive approximation with an alphabet bound depending on target-label alphabet size and approximation tolerance. This differs from a sharp small-R loss at each fixed small alphabet. The institutional PDF https://ddd.uab.cat/pub/poncom/2020/265369/classical-approximate-sufficienc_v2.pdf was found but full opening returned a fetch error; do not describe its complete proof as inspected.

Further novelty search needed for second-order cardinality-constrained IB and general convex-analytic expansions of moment-constrained measure optimization. The proof has not exposed a mathematical counterexample, but prior equivalence remains an active question.

## 7. Reproducible computation

Command: `.venv/bin/python research/onset_experiment.py` (executed twice successfully after setting local plotting caches). The script requires NumPy and Matplotlib only. It saves all posterior vectors, masses, rates, mutual informations, and parameters in `research/onset_results.json`; figures are `research/onset_curvature.pdf` and `research/onset_curvature.png`. The PNG was visually inspected and is readable, with no missing labels.

This computation checks explicit encoders (4) for symmetric channels n=3,4 and a=1/2 at 31 logarithmically spaced rates from 1e-5 to 1e-2 nats. A scalar tangency root identifies z; a grid checks the number of sign changes, and bisection sets the exact input-information budget. Stable KL series near uniform distributions avoid cancellation. Every encoder passes barycenter and rate checks. This is not a numerical global optimization; global optimality of the limiting coefficient is supplied by the theorem.

| n | symbols k+1 | predicted gap / R^2 | observed at R=1e-5 |
|---|---:|---:|---:|
| 3 | 2 | 0.067710850 | 0.067718071 |
| 3 | 3 | 0.016927713 | 0.016926631 |
| 3 | 4 | 0 | floating-point zero |
| 4 | 2 | 0.075958148 | 0.075962056 |
| 4 | 3 | 0.025319383 | 0.025319598 |
| 4 | 4 | 0.008439794 | 0.008439745 |
| 4 | 5 | 0 | floating-point zero |

These runs support the predicted limiting coefficients and expose the finite-R correction, which can have either sign relative to the limiting scaled gap. They do not test finite-sample learning, neural realizability, or generalization.

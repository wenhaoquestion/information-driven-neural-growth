# C02 first-pass lower-family audit

Scope: Phase 8 v1/v2 lower construction, including the fixed quartet, exact curvature degree, all deterministic partitions and at-most budgets, stochastic leading term, and bounded proper losses. Only the neutral definitions packet and final manuscript sources were read; no earlier review or author response was used.

**Finding: no error found in this scope.** The entire lower-family section is byte-identical between v1 `imports/v1/curvature_hierarchy_v1_source/curvature_hierarchy.tex:439–691` and v2 `manuscript/curvature_hierarchy.tex:471–723`. All line references below use the v2 source; subtract 32 for v1.

## Kernel and polynomial construction

- Lines 478–497: the Chebyshev argument is in `[epsilon^2/16,1]` when `epsilon <= |s| <= 4`, giving the stated tail bound. On `|s| <= epsilon/2`, `u >= 3 epsilon^2/64`; squaring `cosh(n arcosh(1+u)) >= exp(n sqrt(u))/2` gives exponent `(sqrt(3)/4)n epsilon`, exactly the constant used.
- Lines 500–522: evenness and zero mean give `H(t)-t_+ >= 0`; for positive `t >= epsilon`, the error is `integral_(t)^4 (s-t) kappa(s) ds <= 8/Z`. Reflection handles negative arguments, and arguments outside the support have zero error. With `epsilon_n=16 log(n)/n`, `delta_n=2 n^(1-4sqrt(3))/log(n)` satisfies `delta_n/epsilon_n^3 -> 0`.
- Lines 529–539: every shifted argument over the full eventual report domain lies strictly inside the cutoff interval. Thus the lack of smoothness at the cutoff does not enter the generator on this domain. Each curvature summand has degree exactly `4n` with positive leading coefficient; reflection cannot cancel that coefficient. The added `8 epsilon_n^2` makes curvature strictly positive. Integrating gives degree exactly `4n+2` for the generator.

## Exhaustive deterministic check

Lines 551–608 correctly include the means of all nonempty subsets, not only contiguous cells. An independent exact-rational Python check generated all 15 partitions of four states and all 39 nested pairs with coarse count at most two and fine count at most three. It derived every cost as a quadratic polynomial in epsilon directly from the reference generator and original masses `1/4`.

The six cost types exactly match lines 568–572:

| Type | Constant | epsilon | epsilon squared |
|---|---:|---:|---:|
| a | 0 | 1/4 | 1/2 |
| b | 1/4 | -1/4 | 2 |
| s | 0 | 0 | 1/2 |
| q | 1/2 | -1/2 | 9/2 |
| t | 1/4 | -1/4 | 2 |
| u | 1/2 | -1/2 | 14/3 |

For every one of the 39 pairs, the check verified domination by cost corner `(2a,a)` or `(t,s)` over the **whole interval** `0 <= epsilon <= 1/16`, using exact endpoint and possible interior-vertex minima of the quadratic differences. Thus this check is not merely a finite epsilon grid. It also verified flat minima `OPT_2=2a` and `OPT_3=s` over that interval. For positive epsilon the flat denominators are positive. Supplementary exact examples gave deterministic prices `62/9`, `1010/33`, and `261890/513` at epsilon `1/16`, `1/64`, and `1/1024` respectively, all equal to `t/(2a)`.

The diagnostic was subsequently saved as `lower_partition_check.py` and executed once from that saved file. Its output, `lower_partition_results.json`, records all 39 hierarchy costs, each dominating corner, and exact whole-interval minimum certificates for the cost differences. The saved check additionally verifies attainment of both Pareto corners. All assertions passed. Reproduce from the workspace root with:

```sh
python3 cross_stage_audit_work/research/equal_mass/lower_partition_check.py
```

Lines 605–614 transfer the reference calculation legitimately: the relevant gaps persist under `O(delta_n)` perturbations, and every deterministic subset mean remains at least epsilon from a hinge. The claimed asymptotic needs neither preservation of `b=t` nor identification of which perturbed corner is smaller, since both corners are asymptotic to `1/(2 epsilon_n)`.

## Stochastic lower bound and explicit upper channel

- Lines 619–639: independent random encoder/coarsener tables can be sampled independently of the source. Conditional on both tables, the source retains its four equal probabilities and the representations are a deterministic nested pair. Conditional Jensen gives each coordinate's distortion lower bound. The exact-count refinement in line 624 is valid: split fine cells within coarse cells until there are three; retain an existing two-cell coarse partition or split its single cell by grouping fine cells. Both costs weakly decrease. Hence the deterministic enumeration covers all at-most-budget stochastic conditional tables.
- Lines 626–636: each conditional ratio vector dominates one of `(1,alpha_n)` and `(beta_n,1)`. Taking expectations then minimizing the maximum of their two affine mixture coordinates gives the displayed lower bound. For `alpha_n ~ beta_n ~ 1/(2 epsilon_n)`, it is asymptotic to `1/(4 epsilon_n)`. This only reveals tables for a lower-bound argument and does not give a public seed to the admissible predictor.
- Lines 643–676: the explicit channel uses exactly three fine outputs and a deterministic two-output coarsener. Direct exact-rational calculation gives fine means `(-1,1/3,5/3)` and coarse means `(-1/5,5/3)` and exactly recovers
  `D_3=epsilon/8+(2/3)epsilon^2` and
  `D_2=1/8+epsilon/2+(26/15)epsilon^2`.
  In particular the nearest unusual mean, `-1/5`, is at distance `1/5-epsilon >= epsilon` from the left hinge for the stated `epsilon <= 1/16`.
- Lines 676–679: numerator approximation errors are `O(delta_n)`. The fine ratio's denominator perturbation is `O(delta_n/epsilon_n^3)`; the coarse ratio has smaller worst-order amplification, `O(delta_n/epsilon_n^2)`. The common bound asserted in the source is therefore valid and tends to zero. Both ratios have leading term `1/(4 epsilon_n)`, matching the lower bound.

## Fixed quartet, loss domain, and constants

Lines 684–707 use `z=4x-3/2`, mapping the four locations to `(1/8,3/8,5/8,7/8)` and the **entire report domain** `[0,1]` to `[-3/2,5/2]`. The chain rule gives `phi_n''(x)=F_n''(4x-3/2)/M_n`, hence `0 < phi_n'' <= 1`; exact degree is preserved. Every deterministic or stochastic posterior mean transforms affinely, so every numerator and flat denominator is scaled by the same `1/(16M_n)`.

The loss construction is strictly proper because conditional regret is precisely `B_phi(x,q)`, positive unless `q=x`. Symmetry implies equal endpoint values and opposite endpoint derivatives. Both outcome losses vanish at their appropriate truthful endpoint, and their derivatives have the stated signs. Consequently they stay in `[0,-phi_n'(0)]=[0,phi_n'(1)]`, whose maximum is `integral_(1/2)^1 phi_n'' <= 1/2`. This verifies boundedness and smoothness on the full closed report interval, including endpoints.

Finally, `r=4n` gives
`1/(2 epsilon_n)=r/[128 log(r/4)] ~ r/(128 log r)` and
`1/(4 epsilon_n)=r/[256 log(r/4)] ~ r/(256 log r)`.
Taking `n=floor(r/4)` establishes the lower order for every sufficiently large degree budget. These are constants of the constructed sequence, not constants of the separate worst-case envelopes; the source makes that distinction.

No remaining blocker was identified for this lower-family subaudit. The upper envelope, fixed unequal-weight comparison, and historical attribution were outside its assigned scope.

# Technical audit of Phases 8–10

The result chain is a mathematical study of Jensen/Bregman distortion under compatible partitions. Its primary objects are population-optimal reports, finite posterior sources, polynomial curvature and ratios to independent flat optima. The formulas do not themselves provide an implementable neural-growth rule. The retained [claim matrix](CLAIM_MATRIX.csv) gives dependencies and precise source locations.

## Retained claims and quantifiers

For four equal masses at capacities two and three, the degree envelope has order $r/\log r$. Fixed positive weights admit weight-dependent constants; this cannot be promoted to weight-uniformity when the smallest mass may vanish. The separate weighted endpoint construction exploits that distinction. The equal-mass lower family and its complete partition checks remain in `research/equal_mass/`.

The general two-level bound for nonnegative polynomial curvature is at most $1+128r^2$ for $r\ge1$, with separate treatment at degree zero. All-capacity statements require one hierarchy working simultaneously at all budgets, not a different pairwise hierarchy for each budget. The retained polynomial-transfer and Hilbert-comparison argument gives the improved upper bound stated in the Phase 8 manuscript. Fixed-generator constants that depend on a normalized positive curvature ratio are uniform only when that ratio is itself fixed. They do not establish a uniform envelope across degenerate generator sequences.

Arbitrary positive weights on a quartet give upper bound $1936r^2/\log^2r$ and a matching-order endpoint lower family. The kernel comparison must preserve original masses and its directed orientation. A naive weighted-triple comparison fails, so the proof uses the detailed four-state argument. The lower family has a strictly positive curvature floor; the finite-degree small-parameter hypotheses and all pair/coarsening cases are part of the proof, not consequences of a plot.

For arbitrary $N\ge3$ at the last two capacities, the quartet reduction preserves the two flat denominators and proves domination. For each $N\ge4$, the far-anchor family supplies a matching order with a sufficient degree threshold uniform in $N$. Sources and losses may depend on $N,r$. The exact tiny-mass padding counterexample and strict-domination example are retained in Phase 10, preventing unjustified shortcuts in this argument.

Flat stochastic and deterministic optima agree by conditional Jensen and deterministic function-table arguments. A two-level rounding comparison costs a factor at most two. This does not automatically become a many-level factor-two theorem or exact equality of stochastic hierarchy prices. Zero-flat-cost cases need their explicit treatment rather than division by zero.

## Method obstructions and the network bridge

The same-source, same-generator coordinate comparison obstruction has order $r^2\log r$ in its comparison quantity but hierarchy price one. It limits that particular comparison route. It neither increases the hierarchy-price lower bound nor proves that all possible all-capacity methods fail.

The indicator-ReLU correspondence uses a restricted finite-state representation, indicator hidden weights, arbitrary subset splits and exact population report refitting. Under those restrictions, partition distortion has a precise network interpretation. Removing the restrictions changes the feasible model; there is no theorem identifying ordinary hidden width, an actual optimizer or a sample-trained model with this partition problem. Function-preserving expansion is not a proof of a strict risk improvement.

## Additive risk and scale

A multiplicative ratio can be large because its denominator is small. With bounded loss, a simple feasible fine-flat hierarchy and a one-cell coarse option give $\rho\le D_1/\operatorname{OPT}_2$, while refining a coarse flat optimum gives $\Delta\le\operatorname{OPT}_2$. For positive denominators,

$$\Delta\le\operatorname{OPT}_2\le D_1/\rho.$$

Consequently, simultaneous divergence of the ratio and a fixed positive additive cost is not the useful asymptotic target under a common loss bound. The meaningful remaining question is a finite-degree window after fixing the actual full-domain amplitude. The additive study addresses ten frozen witnesses, not all losses or all source configurations.

The proper full-domain normalization is

$$\ell_g(q,0)=\int_0^q t g(t)\,dt,\qquad \ell_g(q,1)=\int_q^1(1-t)g(t)\,dt,$$

$$A(g)=\max\{\int_0^1tg(t)\,dt,\int_0^1(1-t)g(t)\,dt\},\qquad \ell=\ell_g/A(g).$$

This sets the maximum outcome-wise oscillation exactly to one. The full report domain matters after an affine coordinate change; the supporting posterior convex hull is not enough. Candidate-specific losses remain custom losses, and the additive optimum need not coincide with the ratio-optimal hierarchy.

## Exact finite diagnostics

The audit's `research/equal_mass/` contains a direct lower-family partition calculation and separate adversarial diagnostics. `research/weighted_top/` retains exact kernel/source calculations and saved configurations. `experiments/risk_boundary_checks.py` checks scale/representation boundary examples. These are finite corroboration and counterexamples, not substitutes for asymptotic proofs or evidence of training.

Earlier numerical failures and corrections are preserved in the stage technical notes: Phase 8 initially encountered endpoint quadrature and JSON serialization problems; a later multiprecision comparison concerns the corrected endpoint evaluation. The additive study separately retains its $n=8192$ floating-point discrepancy relative to a very narrow rigorous interval. None of these values is silently rounded into agreement.

The audit's prior-work reading supported attribution to standard Jensen/Bregman, polynomial localization and hierarchy-approximation tools. It did not certify novelty priority or exhaustive literature coverage. Third-party full texts and excerpts are not redistributed here; paper bibliographies provide the references. The preserved theorem chain and transparent limitations, rather than the number of technical reviews, are the basis on which the work can be assessed.

## Remaining limits

General prescribed capacities and simultaneous all-capacity sharp degree orders are unresolved. Unknown state masses, standard practical losses, learned representations, sample-driven splitting, optimizer behavior, generalization, training acceleration and resource gains are outside the completed evidence. The later additive result establishes one known-mass synthetic finite benchmark with analytic size and pointwise-power guarantees. It does not retroactively turn early training plans or bounded-loss ratio theorems into practical success evidence.

# Technical findings

This note restates project proof and implementation checks as technical findings. It records neither human approval nor external peer review.

The equal-mass proof needs the compatible-optimizer reduction, a Poisson estimate uniform under degenerating gaps, outer Fejér–Riesz factorization, the explicit polynomial construction, and random-table lower-bound reasoning. The retained standard-library audit records 95 rational polynomial instances and 156 Poisson arc-point checks; those finite checks support implementation consistency only. The original width condition first holds at n=1938, and the upper proof has log K≈1963.99, so the asymptotic constants are not practical low-degree guarantees.

For the weighted endpoint construction, original source masses must remain inside every Jensen kernel. With d=(4 log n/n)², b=d², points (0,2d,1−2d,1) and masses (1/2−b,b,b,1/2−b), all 18 nested partition pairs are covered, including noncontiguous ones. The deterministic lower bound is 1/(32d); random-table rounding gives a stochastic lower bound of 1/(64d). The conservative sufficient threshold n≥65536 concerns intermediate analytic estimates, not the validity of directly computed smaller finite examples.

The all-capacity argument uses h(x)=∫₀ˣ√g, an established coordinate, with a degree-uniform comparison for each divergence orientation. Centers must be minimized over before transferring to squared-distance clustering; h of a posterior mean is not generally the mean of h. Cost inequalities handle zero optima, and constant curvature has its own case. The argument does not settle whether the all-capacity logarithm is necessary.

## Implementation failures

1. An initial interior scan wrote CSV rows but failed to serialize a NumPy boolean in JSON metadata. The serializer was repaired and the scan rerun. This was an output-record failure, not a mathematical counterexample.
2. Initial whole-interval high-order floating-point Gauss integration at n=8192 had a normalization crosscheck discrepancy of approximately 3.65×10⁻⁶. The run was rejected. Splitting the integration domain at d/4 and d, together with the cell-kernel breakpoints, reduced the final maximum discrepancy to approximately 2.03×10⁻¹¹. This is still a floating-point crosscheck, not rigorous interval error control.

The raw private logs are omitted; their scientific content is retained here without machine paths or orchestration context.

## Negative and limiting numerical results

With fixed interior width 1/16, increasing degree saturates near price 6.88889. At n=8192, a width of 0.125 times the v1 width gives price about 2.12227 despite a hinge-reference prediction near 225.795; too-narrow finite-degree needles leak curvature. Many spacing/weight-screen cases have price one. Tiny masses alone do not ensure the endpoint separation. The endpoint construction's old explicit stochastic channel was worse than the deterministic feasible upper bound; the reported bracket uses the better feasible bound.

The original endpoint normalized absolute costs use a curvature bound on [0,1]. They must not be reused for an affine interior lift, whose global curvature normalization has to cover a larger interval. Relative prices can diverge while all absolute excess costs vanish.

# Technical findings and failed approaches

This is a neutral account of project technical checks and limitations, not a human approval record or external peer review.

For hierarchy price R>1, the weighted four-state reduction isolates the conflict between fine pair BC and coarse split AB|CD. The proof selects the triple according to the middle masses, reflects if necessary, and obtains original weights (u,v,w) with u≤v. For its triple kernel K, outer-pair kernel H and adjacent kernels, the relation 0≤K−H≤k_q+k_p retains the relevant curvature mass without dividing by a potentially tiny K(c). Chebyshev extrapolation and Markov mass control give R²≤16r² exp(20r/√R) for R≥128, and R(log R)²≤1936r² for R≥256. This is a four-state argument; general source counts remain open.

## Failed proof routes

An arbitrary-triple transplant of the equal-weight Poisson argument is false. With constant curvature, points (0,1/2,1) and weights (1,ε²,ε), the relations q≤T/R² and S≤2T/R can hold for formal R=1/(4√ε)→∞. This disproves that isolated lemma; it is not a large four-state hierarchy-price example. The correct triple choice removes the pathology.

Normalizing a weighted Poisson estimate solely by K(c) also fails uniformly: K(c) can be arbitrarily small relative to nearby kernel values. The outer-pair comparison avoids the required relative-variation bound.

Seeking an Ω(r²) four-state example through sharper endpoint tails or Jacobi/Christoffel needles is ruled out by the new uniform four-state upper theorem. It does not rule out different behavior for larger source counts.

## Comparison method and neural bridge

The coordinate obstruction uses one generator and one three-state source, not just unrelated worst cases for opposite inequalities. Its global partition-comparison product can grow as Ω(r² log r), although all-capacity hierarchy price is exactly one. Improving only these global comparison constants cannot establish a better arbitrary-source hierarchy order; a more specific hierarchy construction might still do so.

The indicator-network risk identity separates representation distortion from report-estimation or report-optimization error. Function-preserving birth alone leaves risk unchanged: in the two-atom squared-loss example, excess risk stays 1/16 while the next flat optimum is zero. In the inherited scalar quadratic example, a simultaneous gradient step of size 3/4 sends parameters (0,1) to (3,1) and excess risk from 2 to 8; exact scalar refitting instead reaches zero. This is not an Adam or benchmark-training analysis.

Unrestricted hidden weights on one-hot inputs permit one ReLU unit to encode all N posteriors. That counterexample uses an N-dimensional encoding and known posterior parameters; it is not a cheap sample-learning procedure. The restricted hierarchy theorem gives no general width lower bound.

## Numerical scope and implementation warning

The saved record contains 378 exact weighted source instances, 12,993 kernel checks, 27 finite indicator populations, 108 oracle states, 81 split/refit transitions and 27 unrestricted-unit fits. A separate classical appendix checks 34 extremizers for degrees 0–33. Coordinate quadrature uses two orders; its independent 100-digit calculation checks only four base integrals at m=4,16. It does not separately certify tiny weighted costs, OPT₂ or comparison products.

The initial figure build emitted an unwritable default font-cache warning but produced a valid figure. Redirecting the cache locally and redrawing changed no scientific results. Raw machine-specific logs are omitted. The degree-2044 coordinate example has comparison-product lower bound about 149433.85 and normalized OPT₂ about 1.67×10⁻³⁸, while its hierarchy price remains one.

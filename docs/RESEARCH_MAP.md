# Research map and current conclusions

Author: Wenyu Huang, University of California San Diego.

This archive follows a sequence of increasingly explicit questions. The stages are preserved because later theorems, learning experiments and statistical checks address different objects. A later phase number does not mean that all preceding limitations have disappeared.

## Finite representation and hierarchy costs

Phase 1 studies compatible two- and three-state representations. An explicit full-support four-state family has unbounded hierarchy price under total Bayes logarithmic loss, with different sharp asymptotic coefficients for deterministic and stochastic representations. The absolute losses vanish in that family. This is a finite-representation theorem, not a trained-network benchmark.

Phase 4 strengthens the source-mass result to sharp order `Theta(sqrt(log(1/gamma)))` for four states. Its finite-representation manuscript is the consolidated early entry. It also provides a separate evidence-interface manuscript and a retained-evidence research report; these should not be merged into a single universal learning theorem.

Phase 7 gives a fixed smooth strictly convex loss with bounded endpoint slopes and unbounded **excess-risk** hierarchy price for four equal masses. The corresponding total-risk prices tend to one. Phase 8 then studies how polynomial curvature degree controls the worst hierarchy price: equal or fixed positive masses have order `Theta(r/log r)`, while allowing masses and locations to degenerate leads to a different regime. It also supplies an all-capacity upper bound `O(r^2 log(r+2))` for arbitrary finite sources.

Phase 9 closes the arbitrary-positive-weight four-state degree envelope at `Theta(r^2/log^2 r)`. Its coordinate-comparison obstruction limits a proof method; the exhibited source has actual hierarchy price one, so the obstruction is not a matching all-capacity hierarchy lower bound. The restricted indicator-ReLU bridge uses a specific representation class and population refitting, not arbitrary neural width or SGD.

Phase 10 extends the sharp degree order to the **last two capacities** `(N-2,N-1)`, uniformly over `N>=4`. General sources are dominated by a quartet; exact deterministic price preservation is proved for a specially designed far-anchor family. There is no claim of general per-source equality or exact stochastic embedding. Arbitrary prescribed capacities and a sharp simultaneous all-capacity order remain open.

## Realizable and repeated neural growth

Phase 2 puts explicit neural parameters and operation constraints into the question. Passive activation information can fail to identify useful immediate splits. Designed stationary-start direction benefits weaken or vanish with longer fitting, ordinary pretraining or random initialization. The operation-aligned binned information statistic does not establish a general advantage over random selection.

Phase 3 studies actual repeated additions/splits, fresh audit labels and fitted rejected branches. A conditional gate guarantee and a restricted fresh paired-loss audit-interface lower bound are proved. The lower bound is not universal for learners retaining their training data. Residual-score superiority remains unresolved; capacity, additional training and alternative data use explain part of the observed gains. Exploratory, post-hoc and confirmatory records remain distinguished.

Phase 4's additional retained-evidence diagnostic shows benefits from releasing completed audit labels to subsequent fitting in its synthetic setup. It does not establish residual-guided growth superiority.

Phase 5 examines exact failure mechanisms in centered quadratic learning. A stored population-moment approximation and an empirical residual correlation are different objects. Generic retained-moment, adaptive-coreset and residual-spectral formulations do not establish substantial novelty by themselves. Exact failure diagnostics, restricted bounds and ordinary-start comparisons are retained, including unfavorable controls.

Phase 6 tests complete algorithms under declared scalar-work budgets. The prespecified 256-seed mean-AUC conjunction passes (`p=0.00311735`), with historical-moment differences of approximately `-0.00058145` against R128 and `-0.00043108` against R138. The fixed-width comparison is tail-sensitive: H wins on only 118 of 256 paired seeds. Random growth is better at the smallest evaluated budget, and the threshold endpoint does not establish minimum-cost savings. Work units are not measured hardware FLOPs; memory and data-access accounting have explicit boundaries.

## Memory and statistical interpretation

Phase 7's exact memory result concerns continuous summaries that answer every centered-quadratic empirical residual query for every dataset in the stated fixed-size domain. At `d=12` the total dimension is 1,521, or 1,443 extra coordinates if the label moment is supplied. This is neither a Gaussian average-case learning lower bound nor a fixed-tolerance finite-word memory requirement.

The additive-risk study fixes the **actual loss amplitude**, posterior coordinates, ten existing witnesses and a one-million-label budget for one preselected candidate. Three interior witnesses exceed the population additive threshold 0.001; seven endpoint witnesses are excluded by certified upper bounds. Only interior `n=2048` receives a recorded confirmation. Its 95% interval is approximately `[0.0024967632816669787, 0.004042417685311304]`, above the prespecified null threshold 0.0005.

The confidence statement holds over all four-state posteriors with the same known masses and public special loss. The analytic power bound of at least 80% is pointwise at the preselected reference posterior. It is not inferred from the single simulated success or guaranteed uniformly over all alternatives. The target is the population optimal-refit hierarchy penalty, not a random trained model's test loss.

## Current boundary

The archive contains internally examined working papers, controlled evidence and a finite statistical bridge. It does not establish ordinary real-data neural-growth superiority, unknown-mass statistical guarantees, standard-loss transfer, autonomous architecture choice, or general savings in hardware time and memory. No all-stage unified submission or external acceptance is claimed.

Further unknown-mass, learning or general-capacity work is a proposal, not an already executed continuation. See the [paper catalogue](PAPERS.md), stage-specific scientific status notes and [known corrections](ERRATA.md).

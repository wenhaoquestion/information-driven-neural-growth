# Phase 10: the last two capacities

The [English working paper](manuscript/phase10_last_two_capacities.pdf) and [LaTeX source](manuscript/phase10_last_two_capacities.tex) study arbitrary finite sources at capacities $(N-2,N-1)$. Read [scientific status](SCIENTIFIC_STATUS.md), [reproduction](REPRODUCE.md), and [technical checks](TECHNICAL_REVIEW.md) alongside the paper.

For $N\ge4$, the worst degree envelopes are $\Theta(r^2/\log^2 r)$ with absolute constants and a sufficiently large degree threshold uniform in $N$. The quartet reduction is domination for a general source. Exact deterministic equality belongs to the specially constructed anchor family; no exact stochastic equality is asserted. $N=3$ has price one.

The separate derivations in `research/upper/` and `research/adversarial/` supplement the assembled paper. `research/experiments/results/` contains the complete saved exact-arithmetic results. The historical fixed-location anchor proof has an $N$-dependent degree threshold; the final far-anchor proof supplies the uniform threshold.

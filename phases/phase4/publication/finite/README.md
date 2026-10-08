# Finite representation hierarchies

[The paper](manuscript/main.pdf) and [LaTeX source](manuscript/main.tex) study finite encoders under hard cardinality constraints and Bayes logarithmic loss. The sharp constants apply to the explicit rare-state family; the universal four-state result proves a matching source-mass order while leaving its best leading constant open. Finite encoding states are not identified with neurons.

`results/` contains all saved scale/sensitivity and partition calculations. `research/` contains source-mass diagnostics, their original random seeds, saved raw test instances, high-precision checks, and plotting code. These are mathematical challenges to proofs, not learned-network experiments. The squared-loss control and near-zero/degenerate cases are preserved.

See [phase status](../../SCIENTIFIC_STATUS.md) and [reproduction instructions](../../REPRODUCE.md). The scientific manuscript is historical; its public PDF was rebuilt with the supplied author block. No experiment or random diagnostic was rerun for repository preparation. Use a disposable copy before running `code/reproduce.py`, which overwrites derived outputs and runs randomized checks as well as TeX.

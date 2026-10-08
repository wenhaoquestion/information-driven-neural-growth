# Frozen diagnostic protocol

Written before execution on 2026-09-20. This checks exact Gaussian theory and estimator mechanisms; it is NOT confirmation of a novel algorithm.

Dimension8, teacher random rotation of diag(1,.3,0,...,0), Gaussian designs, noise SD0/.1, eight independent batches of512,64 seeds830000–830063. Compare pooled raw signal moment, fresh-residual full-matrix correction, known-rank2 correction (privileged), and full lifted online least squares with much larger sufficient-statistic storage. Same chronological data per arm. No labels are selected using evaluator values. Raw X,Y and estimates saved.

Primary theoretical checks: exact 5-point tensor Gauss–Hermite integration of mean/risk/variance identities in dimensions1–4; hard-spectral trajectory identity; ensemble mean raw risk versus exact finite-n formula; positive stale-surrogate birth at the exact teacher; reduction of fresh-residual estimator error; competent full lifted least squares as an explicit negative control against claiming unique statistical superiority. Bands describe independent replicates; no global multiple-testing discovery claim. No seeds removed. Finite Gaussian quadrature verifies polynomial calculations, not proofs.

All estimators are known primitives or specializations. This experiment cannot establish useful autonomous width adaptation; the separate trained-neural experiment examines feature discovery and refitting.

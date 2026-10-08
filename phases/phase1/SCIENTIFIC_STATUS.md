# Phase I scientific status

This is a publication copy of historical September 2026 research, not a new experimental run or external peer review. Supplied authorship and portable paths were applied for release; no theorem or numerical conclusion was strengthened.

The main paper proves a full-support four-state / binary-target family with total Bayes log-loss hierarchy prices asymptotic to `sqrt(L)/(2 sqrt(log 2))` for deterministic growth and half that value for stochastic Markov refinements. It enumerates all 15 partitions and 18 compatible chains, supplies a source-mass upper bound `10(2+log(1/gamma))`, an exact mass-`w` reassignment result, and a sufficient divergent-endpoint-slope theorem for excess loss. The qualitative four-point conflict is prior work, especially Kartowsky--Tal (2019).

The finite computations preserve all 15 scaling cases, 10 sensitivity cases and four higher-precision comparisons. They bracket the finite-L stochastic optimum; the feasible encoder is not claimed to be a global numerical optimizer. The squared-loss control has hierarchy factor one on the tested family. Absolute losses vanish, and the most extreme source masses are impractical to learn by ordinary sampling.

The worst four-state source-mass order remains between the proved square-root logarithmic lower construction and logarithmic upper bound. Neural learning, finite-sample utility and a characterization of all proper losses remain unproved. `research/onset_theorem.md` is a separate proposed information-bottleneck onset result with a less complete correctness/novelty review; it is not a premise of the main paper.

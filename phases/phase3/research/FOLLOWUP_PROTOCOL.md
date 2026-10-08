# Fresh-seed follow-up: a less conservative valid acceptance gate

Frozen before any follow-up outcome is observed. Motivation: the original empirical-Bernstein (EB) confirmation revealed that its range-dependent linear term can account for a large fraction of the rejection threshold. Positive but unresolved proposals do not establish an unavoidable information-theoretic cost. We therefore compare a standard betting test against the original EB gate, with matched fresh seeds and exactly the same candidate-generation and training procedures.

## Fixed execution

* Seeds 63000 through 63015 inclusive; independent of development and original confirmation seeds.
* Tasks: null, linear, additive, interaction, as implemented by the unchanged original runner.
* Training labels 2,048; fresh audit labels 4,096 per opportunity; evaluator input count 16,384; eight opportunities.
* Methods: normalized residual candidate and random candidate. Four candidate draws, 60 probe steps, 200 initial steps, 100 branch steps, 600 diagnostic tail steps, all unchanged.
* Lifetime alpha .05, split across 24 comparisons. Growth comparisons test gain above tau=.0005; continuation comparison tests gain above zero.
* Two independently run complete trajectories using the SAME seeds: `--gate betting` and `--gate eb`. The EB arm is a paired gate comparator, not an unpaired historical comparison.
* Original `phase3/code/sequential_growth.py` and its completed evidence are preserved. `betting_growth.py` is an explicit frozen copy, with a small logged change adding the gate and its event diagnostics.

## Fixed betting rule (established method)

For a tested paired loss difference D in [-1,1] and threshold tau, let Z=D-tau. Fix five lambdas `[.05,.1,.2,.5,1/(1+tau)]`. For each lambda compute the product of factors `1 + lambda * Z_i` over the fresh audit, and average these five products. In log space use log-sum-exp and subtract log5. Accept only when log e >= log(1/alpha_pair). The lambdas are fixed before looking at the audit; choosing the best one without the mixture penalty would be invalid.

Under the null E[D|past]<=tau and fresh iid examples conditional on the fitted candidates, every factor is nonnegative and has conditional expectation at most one. Each product, and their fixed average, has expectation at most one. Markov's inequality gives false acceptance probability at most alpha_pair. Conditional bounds and allocation across the 24 possible comparisons give lifetime probability at most .05 of an unsupported accepted inequality. This is a known betting/e-value construction, not a proposed new statistical test. Primary source: Waudby-Smith and Ramdas, *Estimating means of bounded random variables by betting*, JRSSB 86(1):1–27, 2024 (online 2023), Section 4; https://doi.org/10.1093/jrsssb/qkad009.

## Primary outcomes and adverse evidence

Compare paired final Brier risk, accepted-growth count, retained width, and total training/probe parameter-example updates at the same audit-label budget. Report 16-run paired effects by task and method; do not treat eight events as independent replications. Secondary outcomes: continued versus held actions, post-tail risk, fixed-capacity and full-label-pool controls inherited from the runner, and event-level log e versus threshold. The code computes ordinary EB diagnostics in both arms; the betting arm additionally pays for five product streams per pair. This overhead must remain visible and no wall-time efficiency claim will be inferred from parameter-update proxies.

Falsifying result: the new valid test may fail to accept useful actions, increase capacity without better risk, favor random proposals, or lose to fixed-capacity controls. Retain all of these. A successful gate comparison would establish an avoidable sufficient-bound cost in this setting, not a new information score or a universal autonomous-growth algorithm.


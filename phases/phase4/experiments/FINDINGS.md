# Completed-audit-label recycling: measured findings

## Outcome and scope

Releasing completed audits into later fitting pools lowered final mean Brier risk for both proposal mechanisms on all four tested tasks. The residual proposal did not establish an advantage over matched random additions. A fixed width-26 online learner receiving the same labels at the same times achieved lower mean risk on the interaction and radial tasks, including after additional training. These results identify avoidable label withholding within this implementation and preserve a serious negative control for an information-guided-growth claim.

This is new empirical evidence, not a new statistical theorem. The delayed-release guarantee is an ordinary conditional-validity corollary of the fresh-audit argument. No earlier e-product is transferred to a changed prediction pair. The synthetic experiments do not establish universal superiority, equivalence between proposals, or an impossibility result for structural adaptation.

## What actually ran

- Read-only recovery of P3 interaction seed 63000: the full two-method history, candidates, gates, controls and tails matched saved results at rtol 1e-9, atol 1e-11 (timing excluded).
- Main protocol frozen at **2026-09-15 03:45:05 UTC**, before any seed 750000–750011 was started.
- **48 independent task–seed pairs**, seven methods each, **336 complete trajectories**, **8,064 decision events**. No missing or discarded seed, method, task or event.
- Four stationary synthetic tasks: null, additive, interaction and a new radial boundary, each with six Gaussian inputs and sampled Bernoulli labels.
- **24 events**, **2,048 initial labels**, **4,096 fresh labels/event**, **100,352 unique observed labels per arm**. Evaluator uses 32,768 independent input points per pair and integrates conditional Bernoulli noise.
- **198 million parameter–example updates per arm**, apart from recorded integer remainders (6 million warmup + 24 × 8 million), then a separately reported **32 million-update ungated tail**. All seven arms use persistent Adam state and the same minibatch/rate conventions; discarded branches do not change the committed state.
- Main execution: **330.7 seconds** wall time with two local CPU workers. No GPU, paid service, external upload or remote data is involved.

The original P3 runner uses a different full-batch/reset schedule. Its representative recovery establishes provenance. The new release factorial comparisons are within the new shared persistent-minibatch runner; they are not a data-only comparison against an unmodified historical P3 run.

## Primary paired effects

Negative values favor release. Intervals are descriptive 95% Student-t intervals across 12 paired seeds per task, without multiplicity adjustment.

| Task | Released − withheld, residual | Released − withheld, random | Residual − random after release |
|---|---:|---:|---:|
| Null | −0.001994 [−0.002813, −0.001176] | −0.002016 [−0.002846, −0.001186] | +0.000022 [−0.000054, +0.000097] |
| Additive | −0.004576 [−0.006648, −0.002505] | −0.003943 [−0.005666, −0.002221] | +0.000063 [−0.000237, +0.000363] |
| Interaction | −0.003721 [−0.004557, −0.002885] | −0.004657 [−0.006277, −0.003038] | −0.000110 [−0.000744, +0.000524] |
| Radial | −0.001142 [−0.002131, −0.000152] | −0.001199 [−0.002081, −0.000318] | +0.000554 [−0.000171, +0.001280] |

All four release×proposal difference-of-differences intervals also include zero. This does not establish equivalence; it provides no resolved evidence that residual proposals benefit more from release under this protocol.

## Strong online controls and capacity

| Task | Withheld residual risk | Released residual risk | Released random risk | Fixed 26 online risk | Fixed 26 gated risk | Released residual width |
|---|---:|---:|---:|---:|---:|---:|
| Null | 0.252710 | 0.250716 | 0.250694 | 0.250857 | 0.251677 | 2 |
| Additive | 0.136484 | 0.131908 | 0.131845 | 0.131491 | 0.132636 | 4 |
| Interaction | 0.148025 | 0.144304 | 0.144414 | 0.137553 | 0.140503 | 6 |
| Radial | 0.122568 | 0.121427 | 0.120872 | 0.117145 | 0.118665 | 3 |

Released residual minus fixed26 online is +0.006752 [0.006203, 0.007300] on interaction and +0.004282 [0.003656, 0.004907] on radial. The same comparison with fixed26 gated is +0.003801 [0.003225, 0.004378] and +0.002762 [0.002014, 0.003509]. Thus the fixed architecture's favorable performance on these tasks is not explained solely by removing its gate. The fixed gated arm spends its .05 trajectory error budget across one test per event; the adaptive arm spends it across three.

The adaptive models are much smaller. The experiment therefore reveals a risk–capacity tradeoff rather than dominance of one method on every resource. A fixed width-two learner is competent on the null task (0.250107 mean risk) but underfits the nonlinear interaction and radial tasks (0.209650 and 0.192170). Choosing only that small comparator would give a misleading impression of the adaptive method's standing.

## Longer training and adverse outcomes

After the preset 32-million-update ungated tail, released residual mean risks are 0.250161, 0.131254, 0.142542 and 0.118292, respectively. The release-minus-withheld effects remain negative: −0.003938, −0.005746, −0.004571 and −0.001248; all four descriptive intervals exclude zero. The tail is not a protected deployment update.

The fixed26 online advantage remains on interaction: released residual minus fixed26 is +0.004743 [0.004021, 0.005464], and on radial +0.001445 [0.001061, 0.001829]. On additive, the tail comparison instead modestly favors the smaller released residual network: −0.000329 [−0.000491, −0.000167]. All outcomes are retained. One cannot turn the negative interaction/radial control into a claim that fixed width-26 always wins.

No gated retained checkpoint increased the evaluator's Monte Carlo risk estimate in the 240 gated trajectories. This is a finite observation, not the validity proof. Ungated fixed2 and fixed26 trajectories had 520 and 427 estimated increases across their 1,152 events each; ordinary ungated training does not promise stepwise risk monotonicity.

## Accounting and attribution limits

All arms observe the same 100,352 labels, but the last audit is released only after the final online decision. Thus at most 96,256 labels can have been fitted on before that horizon ends. Withheld arms actually fit on exactly 2,048 distinct labels. Released residual/random arms fit on essentially all previously available labels (roughly 96,061–96,256 depending on task/arm); fixed26 fits about 90,625–90,644 distinct labels within its same parameter–example budget. The tail can fit on the last released block.

Repeated fit accesses differ substantially despite matched parameter–example work: released residual averages approximately 9.84m (null), 6.36m (additive), 4.71m (interaction), and 7.31m (radial), whereas fixed26 makes 0.947m accesses and fixed2 11.647m. This is not a match of data passes, optimizer steps, FLOPs or hardware time. Random proposals spend their saved probe work on branch fitting, rather than burning it.

Residual candidate optimization costs 6.881m parameter–example updates over the horizon; prediction/selection work is separately recorded. For released residual, rejected branch fits alone average 168.727m–182.226m parameter–example updates across tasks, a large fraction of the 198m total. Gate prediction work adds 5.800m–14.090m parameter–example evaluations; fixed26 gated adds 41.091m. Adaptive gates perform 1,474,560 betting-factor evaluations per trajectory; fixed26 gated performs 491,520. None is silently counted as matched optimization work.

The comparison answers the implemented attribution question. It does not optimize label allocation, learn a growth schedule, demonstrate retained certification evidence, benchmark deep networks, or establish that this optimizer horizon is enough for every alternative. Optimizer persistence, warmup matching and the changed P3 schedule are explicit. Evaluator probabilities are excluded from the learner API; future samples are stored by the simulator but unavailable to fitting indices until the stated release time.

## Verification and artifacts

`results/audit.json` reconstructs **all 8,064 events and 14,976 tests** from saved raw data and branch parameters, including risk calculations; maximum log-e discrepancy is **9.95e−13**. `results/integrity.json` verifies the frozen protocol/config/runner/check hashes, all 48 run-file hashes, complete inventory, and conservation of all 336 access ledgers. The final audit is never fitted before the horizon ends. Gradient error is at most 7.7e−12, addition is function-preserving, and label-flip checks preserve pre-audit candidates and earlier decisions as required.

A separate full interaction seed 750000 replay reproduces all seven methods and 168 events at rtol 1e-10/atol 1e-12, excluding elapsed times. All three PNG figures were visually inspected; the paired-effect tick labels were revised for readability without changing the analysis. Figure trajectory bands are ±1 standard error, while paired-effect and capacity error bars are 95% intervals. Precise commands and artifact paths are in `README.md`.

Remaining review is scientific: inspect the synthetic scope, proxy-compute fairness, error allocation, interpretation of the fixed controls, and whether this diagnostic belongs as an empirical section of the information-access manuscript. It is not proposed as an independent fourth preprint.

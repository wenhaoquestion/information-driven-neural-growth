# Confirmation protocol frozen before new-seed execution

Date: 2026-09-13 local. This is confirmation of the selected mechanism study, not preregistration before the entire research project. Exploration: pilot seeds 51000--51002 and 52000--52002 are excluded.

## Main repeated-learning runs
- Seeds 61000--61015 (16 independent repetitions for each of four stationary tasks: null, linear, additive, interaction).
- n_train=2048, n_eval=16384, ordinary width-two random initialization, 200 warmup Adam steps.
- Eight structural opportunities. Four candidate additions optimized 60 steps; one selected using fitting labels only. Both selected growth and no-growth continuation receive 100 all-parameter Adam steps. No-growth includes holding the previous parameters.
- Primary method: normalized residual candidate score. Proposal controls: shuffled residual with the same candidate-optimization workload, random addition, squared centered covariance (Cascade-inspired), and established SSD split.
- Fresh independent audit n=4096 versus n=16384, with alpha=.05 per method trajectory allocated over 24 comparisons. Brier risk improvement threshold tau=.0005 for growth. Fixed audit sizes, no within-round sequential peeking.
- Same training/evaluation streams, initialization, and event-specific candidate starts across methods and audit budgets. Audit blocks across budgets are not nested draws; both are independent of training/evaluation, and per-trajectory blocks are disjoint.
- All audit labels also available to fixed-width small and maximum-width persistent-minibatch controls under the normalized-residual controller's branch-and-probe parameter-example budget. Actual unique labels and optimizer steps reported. No hardware-efficiency equivalence is claimed.
- Fixed final-capacity comparator initialized from scratch at retrospectively observed adaptive width and matched to training/probe parameter-example budget; 600 additional full-batch steps test durability. Ordinary no-growth width-two and fixed width-ten receive the retained-path step budget, and the same 600-step tail.

## Primary questions and summaries
1. Does the residual proposal improve final Brier risk over shuffled residual and random under the SAME gate? Report paired seed means with two-sided 95% Student-t intervals, per task/audit budget. No multiple-comparison discovery claim; report every contrast.
2. Does the process actually perform multiple accepted growth decisions on nonlinear tasks and refrain from growth on null/linear tasks? Report full event traces, width distributions, accepted/declined counts, and retention risk.
3. Does increased fresh evidence change rejection of evaluator-positive proposals at fixed fitting data? Report positive unresolved proposals, gate margins, paired-loss variance, prediction movement, and final risk/capacity.
4. Are any benefits preserved against fixed final capacity, longer optimization, and full-label-pool baselines? Report unfavorable results and actual cost mismatch.

## Acceptance ablation
Use the same seeds/data/proposal definitions with n_audit=16384 but accept by positive empirical mean, with tau=.0005 for growth. Methods residual, shuffled, random. This is an explicitly heuristic policy, with NO simultaneous population-risk guarantee. It distinguishes strict evidence requirements from candidate/training failure. It will not replace the primary method based on evaluator risk. All trajectories retained.

## Scope and deviations
No test-driven hyperparameter changes are permitted within this confirmation batch. Implementation correctness fixes must be documented, invalid affected results retained and rerun under a new filename. The 600-step tail is an ungated diagnostic and is outside the deployment monotonicity theorem. Conditional-probability evaluation is Monte Carlo in X, not exact population risk. Intervals quantify independent run variability, not proof of equivalence or adjusted discoveries. Audit-label cost is separate from fitting labels, training work, evaluator-only compute, and final network size.

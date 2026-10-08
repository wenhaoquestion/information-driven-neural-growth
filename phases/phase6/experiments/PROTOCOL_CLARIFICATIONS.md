# Wording clarifications after execution

Date: 2026-10-07 UTC. The original prospectively hashed protocol is retained unchanged. These clarifications do not alter any seed, algorithm, hyperparameter, budget, endpoint, test or inclusion rule.

1. The phrase that learners never "receive" future observations is too strong as an API description: the local harness passes the complete synthetic x/y arrays to the learner, whose implementation accesses only arrived blocks. The enforceable scientific rule implemented and audited is **no access to future observations**. It is not an operating-system or capability isolation boundary. Independent perturbation of future x/y left earlier trajectory states bitwise unchanged, and complete saved access ledgers were verified. Teacher matrices and evaluator scores are not arguments to the learner.

2. Learner timing is the sum of named algorithm blocks, excluding data creation, evaluation, serialization and major access-counter work. It can still contain minor instrumentation/control/allocation and timer overhead. Separate component-level timings for every excluded item were not recorded. Whole-invocation elapsed time is also recorded. No claim of zero measurement overhead or controlled wall-clock equivalence is made.

3. The frozen primary analysis source remains unchanged. `analysis/plot_diagnostics.py` is an explicitly post-analysis descriptive visualization of all primary seeds and already calculated pointwise contrasts; it does not change the endpoint, tests or outcome selection.

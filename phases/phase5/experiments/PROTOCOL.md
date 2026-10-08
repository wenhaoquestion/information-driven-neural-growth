# Controlled discovery and refitting study

Status: development implementation checks. Confirmation has not started when this protocol was written. The final `frozen.json` records the exact timestamp and hashes before confirmation.

## Scientific question and nonclaims

Does an all-history raw target moment `S_n - A_current` identify useful new quadratic directions more effectively than an empirical residual moment, learned residual probes, random additions, or fixed online capacity, after charging for proposal work and allowing every hidden vector to train?

The main test is an ablation of discovered directions and retention, **not autonomous width selection**. Birth opportunities and final width are fixed before the data. No claim of safety, always-beneficial growth, novel spectral estimation, novel replay, or guaranteed optimal nonlinear optimization is made. The null task deliberately tests the cost of adding useless capacity. A zero-added-coefficient birth preserves the current function; subsequent unvalidated fitting can increase population risk.

## Populations and learner

All inputs are independent `N(0,I_12)` vectors. A task-specific hidden orthogonal rotation is fixed by RNG seed `[571902, task_id]`; learners never receive it. There are four task laws:

- `null`: conditional mean zero, independent Gaussian noise SD .5.
- `lowrank_noiseless`: centered quadratic `x'Ax-tr(A)` with eigenvalues .8,.45 and ten zeros, zero observation noise.
- `indefinite_noisy`: centered quadratic with eigenvalues .65,-.5,.3,-.2 and eight zeros, Gaussian noise SD .3.
- `nonlinear`: rotated mean `.8 tanh(1.5 z1)tanh(1.5 z2)+.4 sin(z3)`, Gaussian noise SD .2. This is outside the quadratic family and tests model mismatch.

The fitted prediction is `c + sum_j a_j[(w_j'x)^2 - ||w_j||²]`. All hidden vectors `w_j`, all output coefficients `a_j`, and `c` are trainable. Initial hidden vectors are ordinary normalized independent Gaussians and initial output coefficients have SD .01. All growing arms start at width one and add one discovered/random direction at each of the six later arrivals, ending at width seven. The fixed comparator starts and remains at width seven.

There are seven sequential arrivals of 1,024 labels, 7,168 unique labels in total. An event's fitting pool is the current fresh block plus the arm's retained examples from earlier blocks. The current block is incorporated into the reservoir only after event fitting. No future block enters a proposal, fitting update or decision. Reservoir sampling uses Algorithm R independent of labels.

## Seven arms

1. `historical_moment`: retain all-history `S_n = sum y(xx'-I)/(2n)` and a 128-row replay reservoir. Propose the eigenvector of largest absolute eigenvalue of `S_n - A_current`, where `A_current=sum a_j w_jw_j'`. This uses the known Gaussian population inverse and is **not** the empirical residual moment. Finite-sample signal-induced noise remains even when current predictions fit the teacher.
2. `reservoir_residual`: a 138-row reservoir; propose from the current residual moment computed on that old reservoir.
3. `fresh_residual`: a 138-row reservoir; propose from the residual moment computed on the full currently arriving block.
4. `learned_probe`: four random incoming vectors; twenty normalized gradient-ascent steps on squared residual covariance divided by candidate second moment plus .001, using the 138-row old reservoir. Select by that same objective. This is a conventional residual-feature search baseline.
5. `random_replay128`: random birth, 128-row replay reservoir.
6. `random_matched138`: random birth, 138-row replay reservoir.
7. `fixed7_replay138`: online fixed width seven, current block plus 138-row replay reservoir, all saved candidate-search compute allocated to fitting.

No method receives target directions, target rank, target matrix or population conditional means. All arms have the same arrival schedule. The fixed architecture knows the experiment-wide cap seven, not the teacher's rank.

## Resource accounting

Each arm receives an arithmetic-proxy budget of 2,000,000 per arrival. A fitting example costs `3*d*k+4*k+3`; moment accumulation costs `d²+d` per example; residual eigencandidates charge residual prediction and matrix accumulation; eigendecomposition charges `10*d³`; probe gradient/search passes are charged explicitly. Any saved proposal work becomes additional fitting updates. No exact FLOP or wall-clock matching is claimed. Random draws, reservoir replacement and scalar optimizer constants are not included in this proxy and are recorded or disclosed separately.

Persistent retained-array storage for the historical arm is `128*(12+2)+144=1936` eight-byte-word equivalents, including replay input, label and ID plus its full matrix. The enlarged random reservoir uses `138*(12+2)=1932`, four words fewer. Growing architectures have the same scheduled width and model/Adam sizes, making this a near match in persistent algorithmic storage. This is **not a match of peak process memory**. Model plus first/second Adam moments, current block, duplicated fitting pool, gradient workspace and candidate-array workspace bounds are logged separately. Six controller/RNG state words are additionally disclosed. BLAS/eigensolver internal workspaces, Python overhead and the evaluator/archive process are excluded from the array ledger; no physical byte-memory claim follows.

All fit/proposal/moment label accesses are saved by source ID in compressed arrays. Every event stores available pool IDs, retained IDs, model and optimizer snapshots, candidate vectors/scores, cumulative charges and risk-independent actions. The raw archive/evaluator is external instrumentation; its whole-data storage is not available as learner retention.

## Optimization and controls

Persistent Adam uses learning rate .015, beta values .9/.999, epsilon 1e-8, minibatches at most 128, and global gradient clipping at norm 20. New-unit Adam moments are zero; existing state and step counter carry forward. No normalization transformation is applied during full model training. Probe steps use normalized vectors and learning rate .05.

Every growth adds a direction with output coefficient exactly zero, followed by fully trainable refitting. Probe/eigendecomposition work is included in the event budget. There is no rejection branch, audit gate, holdout reuse or post-hoc best checkpoint in the primary study. Population risk may increase.

## Paired same-checkpoint diagnostic

After all primary method trajectories have finished, take each of the six pre-growth checkpoints and optimizer states from `random_matched138`. At that *same* checkpoint and old reservoir, construct all five direction types: historical moment, old-reservoir residual, fresh residual, learned probes, and random. All see the same permitted prefix, current block and retained rows. For historical moments, the analysis reconstructs the all-history matrix from the archive; this grants the diagnostic its stated moment interface, and is not retroactively available to the random primary learner.

Each direction receives identical post-birth fitting budget (2,000,000) and identical minibatch randomness. Its candidate-search cost is additional diagnostic work and recorded separately. This deliberately isolates initialization from unequal training work. For quadratic tasks, the evaluator computes exact best possible gain along the direction, `2(w'(A*-A)w)^2`, and the unrestricted rank-one oracle gain. It also evaluates the actually trained branch. All these quantities are evaluator-only. This distinguishes raw direction quality, capacity existence and realized finite-time optimization, and can reveal erasure or reversal of an initialization advantage.

## Development, confirmation and endpoints

Development seeds are 810000–810003 on `lowrank_noiseless` and `nonlinear`. Development is limited to implementation correctness, finite/stable execution, resource/access assertions and report generation. Evaluator outcomes are not used for tuning. Initial implementation issues were a missing parenthesis, a NumPy-bool JSON cast in the test harness, and relative-path metadata resolution; these were corrected before any confirmation and are recorded in `DEVELOPMENT_LOG.md`.

Confirmation uses seeds 820000–820011 on all four tasks: 48 paired configurations, seven primary methods, 336 trajectories and 2,352 primary events. Every task-method mean is reported. Primary paired comparisons are historical moment minus fresh residual, reservoir residual, learned probe, memory-matched random, and fixed7. Random R128 versus R138 is a persistent-storage control. Descriptive 95% Student-t intervals use independent seeds as the replication unit and have no multiple-testing correction. They are not discovery-adjusted significance claims.

Quadratic-task population risk is exact `2||A_current-A*||_F²+c²+noise_variance`. Nonlinear-task risk uses 32,768 independent Gaussian evaluator inputs per seed and integrates label noise; it remains Monte Carlo in X. Evaluator calls occur only after the complete learner trajectory is fixed. No evaluation outcome enters a proposal, update, retention choice or stopping rule.

Save all results, including divergence/failure if any. Confirmation code/config changes require a new version and must not silently replace the frozen study. Figures/tables must read the saved JSON files, never manually entered summary points.

Pre-confirmation accounting correction after independent review: the historical candidate additionally charges `d²k+dk` to form the current effective matrix. The post-trajectory diagnostic's reconstruction of its historical matrix explicitly logs `n_prefix` old-label rereads and `n_prefix*(d²+d)` work; that diagnostic acquisition is additional to candidate and identical-refit costs. It is neither free historical access nor charged as new label purchases.

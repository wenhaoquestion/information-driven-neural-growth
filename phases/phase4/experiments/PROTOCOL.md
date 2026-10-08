# Audit-label release: frozen diagnostic protocol

This protocol is fixed before seeds 750000–750011 are run. It is a new-seed diagnostic motivated by P3, not a claim to invent conditional validity or a new statistical test. Its timestamp and exact runner/config hashes are in `protocol_frozen.json`. All results, including failures and development runs, are retained.

## Prior recovery and development

The original P3 betting trajectory on interaction data, seed 63000, was reconstructed using the unchanged original runner. Both proposal methods and all branches, tails, and fixed controls match the saved trajectory at relative tolerance 1e-9 and absolute tolerance 1e-11; timings are excluded. This is one representative recovery, not a replay of every P3 run.

Before new seeds, a three-event timing run uses seed 740001, and tiny implementation checks use seeds 741001–741002. The first timing execution reached output generation but failed on a relative-path serialization error; `logs/timing.log` retains that failure. The path fix changes only output handling. The retry takes approximately 2.2 seconds for all seven methods at a three-event horizon and a shorter evaluator/tail. No optimization settings were selected from its risks. Main execution should take roughly 5–15 minutes with two CPU workers, depending on contention. No GPU or paid compute is used.

## Question and arms

Does excluding every completed audit from subsequent fitting explain part of the observed limitation, under the same fresh audit policy and visible optimization budgets? Does any benefit belong to residual proposals, ordinary fitting, or the commitment gate?

Seven online arms are run on exactly the same paired arrival data, in the following fixed order:

1. `withheld_residual`: all fitting remains on the initial block; learned residual proposals.
2. `released_residual`: completed audit labels enter the next fitting pool; learned residual proposals.
3. `withheld_random`: initial fitting block only; random additions.
4. `released_random`: completed audits enter future fitting; random additions.
5. `fixed2_online`: a persistent-Adam width-two learner; completed audits enter future fitting; no gate.
6. `fixed26_online`: a persistent-Adam width-26 learner; same release times and no gate.
7. `fixed26_gated`: width-26 continuation candidates with the same fresh betting mechanism; release after each decision.

Width 26 is set before outcomes and equals the maximum possible final adaptive width (2 + 24). The ungated learners pursue ordinary online risk reduction; they have no certified non-increasing-checkpoint claim. The gated fixed learner separates commitment restrictions from architecture selection. The released/withheld and residual/random 2×2 comparison avoids confounding release with proposal type.

## Data and fixed horizon

- Tasks: null, additive, interaction (P3 definitions) and radial. All have X ~ N(0,I6) and Y ~ Bernoulli(0.05 + 0.90 sigmoid(g(X))).
- Null: g=0, checking inappropriate adaptation when the constant predictor is optimal.
- Additive: g=2.5 tanh(2X1+0.6) − 2 tanh(2X2−0.5) + 1.7 tanh(2X3+0.3) − 1.5 tanh(2X4).
- Interaction: g=4 tanh(2X1)tanh(2X2)+2 tanh(2X3)tanh(2X4), challenging a small additive hidden representation.
- Radial: g=3(X1²+X2²−1.4), a new curved boundary with four nuisance coordinates. It is not generated from the fitted tanh architecture or the growth theorem.
- Initial fitting labels: 2,048. Audit labels: 4,096 per event. Events: 24, chosen before confirmation and three times P3's horizon. Total unique observed labels: 100,352 per method/run, counting each once.
- Seeds 750000–750011 inclusive for each task, 48 independent task–seed pairs and 336 method trajectories, 8,064 events.
- Test inputs: 32,768 independent points per task–seed. Conditional Bernoulli label noise is integrated exactly, but risk remains Monte Carlo in X. Evaluator values never enter proposals, fitting, gating, or model selection.

Raw data and fitted-label access counts are saved in compressed NPZ files. Data are generated with separate training/audit and evaluator streams. The learner has X, Y, index availability, parameters and past decisions. Although the simulation stores future data in memory, all fit access is through the permitted index range; runtime assertions and perturbation checks detect current/future-audit fitting leakage. The evaluator is called only after the entire action history is frozen.

## Optimization and resource matching

Every arm receives 6,000,000 parameter–example updates for warmup and 8,000,000 per event. A final 32,000,000-update ungated tail is a separately reported optimization diagnostic. Unspent integer remainders below one batch's parameter count are recorded. This is a matching of an explicit operation proxy, not FLOPs, wall time, memory or number of optimizer steps.

All neural fitting uses minibatches of at most 256, Adam with learning rate .01, coefficients (.9,.999), epsilon 1e-8. A random permutation cycles through the currently allowed pool. The committed Adam state persists. Every continuation/growth branch clones its state, and only the chosen state is committed. A newly added unit has outgoing weight zero and new optimizer moments zero; the existing Adam step counter is retained. Thus no rejected fit modifies the incumbent, and random and residual arms use identical branch-optimizer conventions.

Residual proposals optimize four candidate incoming vectors/biases for 40 minibatches at learning rate .04, using the P3 normalized residual tangent score. Final candidate selection uses at most 4,096 historical fitting points. Random additions draw from the same initial candidate distribution and choose a random index. All candidate optimization is charged. Each arm's remaining event budget is split equally between growth and continuation; therefore random proposals receive extra branch fitting corresponding to their saved search work. This is deliberate useful-compute matching, rather than burning their savings. The proposal fitting costs 286,720 parameter–example updates per event at the full batch size; additional proposal prediction/selection work is recorded separately.

Fixed learners use the entire event budget for continuation, reflecting the resource cost of two-branch search. No offline learner receives future labels. Fitting access counts, unique fitted labels, candidate work, rejected branch work, gate prediction work, betting factors, elapsed time, retained width and parameter count are all reported separately. Larger available pools do not imply every label has been fitted on; actual counts determine that.

## Decision and release order

For every gated event: clone the incumbent; construct and fit candidates on H_(t−1); freeze their hashes; access the new audit A_t; run the gate; commit exactly one saved branch/state; only then expand the fitting pool to include A_t. Release itself cannot refit the current predictor. All current and later audit IDs must have zero fit accesses during candidate construction.

The established P3 fixed-mixture bounded-mean test averages products with lambdas [.05,.1,.2,.5,1/(1+tau)] and accepts at log(e) ≥ log(1/alpha_pair). Growth needs improvement over both incumbent and continuation by tau=.0005; continuation needs improvement over incumbent by zero. Adaptive arms allocate alpha_pair=.05/(3×24). The one-comparison fixed gated arm allocates .05/24; both spend the same per-trajectory total error budget. Each prediction pair starts a new test. Historical labels are reused only for fitting, never as old certification factors for a new adaptive pair.

Conditional on past history, newly fitted candidates are fixed and the current audit is independent. Thus post-decision release preserves the existing conditional validity argument. This is a standard corollary; the empirical diagnostic is distinct from any substantive new theorem. Warmup, ungated fixed baselines and diagnostic tails are outside the checkpoint guarantee. Experimental method/seed multiplicity is not covered by a single trajectory's .05 bound.

## Preset analysis

Primary outcome: final committed Brier risk at event 24. Primary paired contrasts by task: released minus withheld for residual and random; released residual minus released random; and the release×proposal difference of differences. Report descriptive two-sided 95% Student-t intervals across the 12 independent seeds. These are not multiplicity-adjusted discoveries; no general superiority or equivalence claim is based on crossing/containing zero.

Secondary outcomes: event-12 and last-six-event mean risks; all risk trajectories; final width; growth/continue/hold counts; final post-tail risk; same-arrival fixed comparisons; training/audit/candidate/rejected work; unique observed/fitted labels and repeated accesses. Compare released residual with both fixed26 arms and fixed2. Show risk–capacity and risk–resource plots. No task, seed, branch or horizon is removed because its outcome is unfavorable.

Implementation checks cover finite-difference gradients, function-preserving addition, immutable committed parameters/state, release timing, disjoint audit blocks, current/future-label perturbations, independent reconstruction of every saved gate and evaluator risk, exact candidate selection/retention hashes, work ledgers and raw-data hashes. Computational checks are evidence about the implementation; they do not replace the conditional proof.

## Limits fixed before results

This is a modest synthetic diagnostic of a specified tanh learner and conservative risk gate. It neither establishes a generally effective structural-learning principle nor rules one out. Different optimization, growth actions or evidence policies may change the results. The persistent minibatch optimizer and matched event budget differ from P3's full-batch/reset runner; the original recovery establishes provenance, while the new factorial arms isolate release within the new controlled runner. End-of-run tails are clearly outside the online certification claim.

# Phase 6: prospectively locked risk--work frontier study

Status: protocol specified before smoke or development outcomes; code freeze follows correctness review. This is a local prospective lock, not externally certified registration. Date: 2026-10-07 UTC.

## Question and incremental value

Does a retained historical moment actually improve the final population risk attainable across a prespecified online training-work grid, after charging search, moment maintenance, optimization, and retention operations? Phase 5 already found that better initial directions did not establish final-risk superiority. This study does not rename the method, erase those results, or assert a new general learning principle. Its increment is independent teacher rotations, a prospectively fixed work frontier, balanced development selection, broader paired replication, and an optimizer-age sensitivity check.

## Mathematical and algorithmic scope

- Independent Gaussian inputs in dimension 12. Seven arrivals of 1,024 observations. Centered quadratic network with trainable hidden vectors, signed output coefficients and intercept.
- Primary teacher eigenvalues [0.65, -0.5, 0.3, -0.2, 0, ...], independently rotated for each seed; additive Gaussian noise standard deviation 0.3.
- Four methods: historical_moment (128 replay rows plus a full 12x12 historical moment); random_growth128; random_growth138; fixed7 (138 replay rows). The two random baselines distinguish direction information from extra replay. Ten rows are 140 float64/int64-equivalent words versus 144 moment words; memory is close but NOT exactly matched. Model/optimizer, working memory, RNG/control and logging are reported separately.
- Growth is scheduled from width 1 to 7. This experiment does not test autonomous timing or adaptive width. Fixed7 trains all seven units from the beginning. All methods receive data at the same arrivals and use the same fitting-data permissions.
- Actual optimization is clipped Adam, batch size 128, gradient clip 20, full-batch carry across epoch boundaries. A final budget-limited partial batch is allowed and reported. It cannot be confused with the old replay-capacity-dependent epoch-tail problem.
- Global Adam step with zero moments for newborn coordinates is primary. Local coordinate age is a separately locked sensitivity analysis; old coordinates are not reset. No best-age policy is selected after looking at confirmation.
- Learners never receive teacher matrices, exact risks, test labels or future observations. All trajectories in a seed's job are completed before evaluator-only population risks are added.

## Work and wall-clock accounting

Per-arrival allocated work caps are [3.2, 5, 6.4, 8, 10] million declared scalar-work units. All methods run all five budgets independently; a high-budget run is not a continuation of a lower-budget run. The grid was fixed before any new outcomes; its scale accounts for an expanded cost definition relative to Phase 5 and is not claimed directly comparable to the old units.

The implementation must charge initialization, births, proposal construction, historical-moment updates, eigendecomposition, replay maintenance/copies, fit-pool construction, minibatch gathering/permutation, gradient computation, clipping and Adam. Exact deterministic formulas and category totals are emitted by the runner. The eigensolver uses a declared 10*d^3 surrogate. This is a transparent algorithmic work proxy, NOT measured hardware FLOPs. Auditors must check that no per-event total exceeds the cap and account for integer remainders.

Full learner wall-clock includes these algorithm operations; data generation, post-trajectory evaluation, archive serialization and independent audit are separately reported. Python, BLAS, allocation, cache and OS scheduling prevent a strict wall-clock-equivalence claim. Named-array memory is not peak resident memory. No candidate branch is rejected or silently omitted: these are four complete fixed algorithms, not a validation-selection controller.

## Development selection, frozen before development

To avoid handicapping the fixed-width control, all four methods receive the same development selection rule: 16 independent teacher/data seeds 96010000--96010015; learning-rate candidates {0.0075, 0.015, 0.03}; all five work caps. Each method selects the rate with the lowest mean development AUC, with the smaller rate breaking exact ties. There is no seed exclusion, extra candidate rate, task selection, or optimizer-specific extra search. Development uses separate synthetic teachers and is not confirmation evidence. Its sample/work/wall cost is retained and reported; it is not hidden inside a free oracle. Selected rates are frozen before starting primary confirmation and reused without retuning on secondary tasks and local-age controls.

Smoke seeds 96000000 and 96000001 are only for numerical correctness, record structure and resource timing. Smoke outcomes cannot change the scientific grid, primary statistic, learning-rate candidate set or confirmation sample size. Necessary bug fixes require preserved failure logs and a new source freeze before confirmation.

## Primary sample and statistic

Exactly 256 independent teacher/data/optimization seeds 96100000--96100255 on indefinite_noisy. Each is the paired unit; budgets, methods, arrivals and checkpoint rows are not independent replications. A seed is retained if it performs badly; nonfinite results or incomplete jobs trigger a disclosed failure investigation, not silent removal.

For seed i and method m, let e_imj be final exact excess population risk after the j-th independently trained budget. Let x_j=log(7 B_j). The single primary summary is normalized trapezoidal area

    A_im = sum_j (x_(j+1)-x_j)(e_imj+e_im(j+1))/2 / (x_5-x_1).

Only the budget coordinate is logged; risk is NOT logged. Population risk is sigma^2 + 2||B-A_*||_F^2 + c^2. The primary contrasts are the three paired means A_H-A_R128, A_H-A_R138 and A_H-A_F7. The intersection-union success criterion requires each one-sided 95% t upper bound to be below zero (equivalently max of the three one-sided p-values <0.05). No multiplicity correction is required for this conjunction; it does not license separate family-wide claims. The paired-t approximation is stated, and fixed-seed 10,000-resample paired bootstrap intervals are a sensitivity analysis, not a second opportunity to choose a favorable test.

At N=256, a conservative planning calculation needs roughly a 0.20 standardized paired AUC effect in the weakest contrast for 80% joint power. Twenty-percent work savings alone cannot determine power. Report observed paired variance and precision. Do not interpret an interval crossing zero as equivalence. Do not increase N after observing p-values.

## Secondary analyses and limits

- Pointwise full-grid risk curves and 95% descriptive intervals; paired differences at every allocated budget; actual charged work, steps, batches, retained/model/workspace bytes and wall-clock.
- Threshold epsilon=0.01 in excess risk is fixed for the primary noisy task. Report every success and failure. The first successful *evaluated final-budget grid point* is a discrete grid quantity; no interpolation, monotonic-risk assumption, or dropping nonattainers. It is not an oracle continuous first-passage training cost.
- Predefined 20% allocated-work compression contrasts H(6.4M) vs each comparator(8M), and H(8M) vs comparator(10M). Report full paired risk differences and target attainment. These are exploratory secondary contrasts. Passing them would not establish that a comparator could not also meet the target at a lower budget.
- Null and lowrank_noiseless: 32 independent seeds 96200000--96200031 for each task, all four methods and five budgets, selected primary development rates. Null diagnoses unnecessary scheduled growth; noiseless rank-two checks whether conclusions depend on noise. These are scoped descriptive controls, not additional independent evidence for the primary significance test.
- Local-age sensitivity: primary seeds 96100000--96100031, all four methods and five budgets under local age, paired with their global-age primary runs. No retuning and no treating these as 32 new primary samples.

## Resources and stop rules

Existing local Python environment only, NumPy CPU, BLAS thread counts pinned to one; no GPU, cloud compute, purchases, new credentials or installations. Host: Apple M1 Pro, 8 logical CPUs, 16 GiB memory. Expected training compute: approximately 5--15 minutes, to be calibrated by smoke. Each invocation has an explicit wall cap, primary at 1,200 seconds; do not launch indefinite background work. Progress is written per seed. If smoke predicts an impractical bound, report and stop before confirmation rather than silently changing the sample size.

Scientific continuation requires the primary conjunction plus an interpretable advantage that survives actual-cost and optimizer-age inspection. A statistically detectable but negligible difference is not a new general principle. Failure against a competent random or fixed-width control ends this candidate's broad efficiency claim for the tested regime. Report inconclusive intervals as inconclusive. No additional search for a favorable task, endpoint, seed subset or budget after confirmation.

## Archive and publication note

The original scientific configurations, source identities and completion records are retained. Public reproduction commands and omissions are documented in `../REPRODUCE.md`. This curated protocol removes only the original local installation/publication workflow paragraph; its original SHA remains in the freeze record.

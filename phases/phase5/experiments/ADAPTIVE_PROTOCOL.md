# Separate adaptive-width extension

Frozen after the main controlled-direction study and before new seeds 840000–840011. This is a new exploratory extension with a frozen execution recipe, not an independent confirmation of a prespecified main-paper algorithmic claim. No main results are overwritten. The candidate family and heuristic selection rule were specified before this extension's outcomes.

Use the same four task laws, centered quadratic model, all-parameter Adam, seven arrivals of 1,024 labels, two-million arithmetic-proxy event budget, and maximum width seven. Every arm retains a 138-row reservoir. At each arrival the first 768 labels may enter current fitting/proposals; the final 256 are unseen validation labels until all current branch parameters are frozen. After committing, release the whole block into the reservoir. Initial warmup is ungated. Future data are inaccessible.

The five arms are fresh-residual adaptive growth, random adaptive growth, fixed width two with validation, fixed width seven with validation, and fixed width seven without validation. The growth arms start at width one. At each later opportunity, if below the cap, propose a zero-output new unit, and fit both continuation and growth branches from cloned persistent Adam states. Split the fitting budget in half after deducting candidate work and validation predictions. Fixed arms use their whole remaining budget for continuation. Saved candidate and rejected branch work is charged. The ungated fixed arm allocates saved validation work to fitting.

For each fitted candidate, compute old-minus-new squared losses on the same 256 validation labels. Qualify it if mean gain exceeds `2*sample_standard_error + .002 + .001*width_increase`. Among qualifying candidates install the one with highest mean gain minus `.002+.001*width_increase`; otherwise hold the exact incumbent. No post-decision optimizer update occurs. This is an explicitly heuristic, multiple-comparison-unadjusted rule for unbounded Gaussian losses; it has **no advertised finite-sample safety guarantee**. The familiar fresh-holdout pattern is not a novelty claim. Holding and rejecting do not certify adequacy of the architecture or candidate family.

Save all branch parameters, pre-branch optimizer states, validation statistics, actual grow/continue/hold actions, post-decision reservoir IDs, access arrays and arithmetic/resource ledgers. Record current validation exclusion with runtime assertions. Population risk is evaluated only after each complete trajectory. Quadratic task risk is exact; nonlinear risk uses the unchanged independent Monte Carlo evaluator. Twelve independent new seeds per task yield 240 complete trajectories and 1,680 events. Report every task and method, paired risk effects, achieved widths, number of accepted growth/continuation/hold events, and evaluator risk increases despite the heuristic gate.

Memory is not exactly matched across widths: fixed width-seven optimizers are larger from the outset, and adaptive branch workspaces can exceed fixed ones. Persistent retention and branch-array estimates are logged; BLAS, Python and evaluator/archive storage are excluded. This extension tests a deployable adaptive mechanism under visible restrictions, not theorem-level novelty or deep-network efficacy.

Reproduce with:

```
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python phase5/code/adaptive_quadratic.py --config phase5/experiments/adaptive_confirmation.json --output-dir phase5/experiments/adaptive_replay/results --workers 2
```

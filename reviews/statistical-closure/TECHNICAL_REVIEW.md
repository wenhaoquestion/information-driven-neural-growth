# Technical review of the additive-risk inference

The saved experiment supports a finite detectable additive-conflict window under known equal state masses and one specified custom loss. The standalone mathematical implementation and the separate code implementation reproduce the same exact interval, all 39 hierarchy intervals, all 15 report-error bounds, and the pointwise power certificate. Their historical results report 335 and 827 assertions, respectively. Finite checks supplement the analytic arguments below.

## Estimand and information

For a partition $P$, let $F_P(\theta)=\sum_{B\in P}p_B h(\mu_B)$, with $h$ the public loss's Bayes entropy and $\mu_B$ its population posterior mean. The common statewise Bayes risk cancels from regret:

$$D(P)-\operatorname{OPT}_k=\max_{Q:|Q|\le k}(F_P-F_Q).$$

The target is $\Delta=\min_{H_3\text{ refines }H_2}\max_{k=2,3}[D(H_k)-\operatorname{OPT}_k]$, with all at-most-budget partitions allowed, including noncontiguous ones. There are 15 partitions, 8 at most two-block partitions, 14 at most three-block partitions and 39 compatible pairs. The parameter is fixed for fixed $(\theta,p,\ell)$; the confidence interval is random. It does not estimate a trained network's random test risk or identify its best partition.

Inference reads only `candidate/counts/public_p/m`, requires the prescribed equal masses, and rejects extra reference-posterior fields. The public loss parameters may identify the construction, but validity still covers every $\theta\in[0,1]^4$ for that same loss. The full-domain oracle does not assume the unknown posterior follows the reference formula. Simulation and inference share a process; the reviewed input dependency is not a security boundary or external blinding.

## Uniform coverage and size

Conditioning on any state counts, invert the Bernstein inequality

$$n(t-q)^2\le c[2q(1-q)+(2/3)|t-q|],\qquad c=1269/250.$$

The quadratic is convex on either side of the sample proportion. Rational bisection keeps the outward bracket, with error at most $2^{-56}$; a zero state count gives $[0,1]$. A positive-series rational certificate proves $8e^{-c}<0.05$. Thus the four-dimensional rectangle simultaneously covers the posterior with probability at least 95%, conditional on all counts and hence unconditionally.

For $J_{PQ}=F_P-F_Q$,

$$\partial_i J_{PQ}=p_i[h'(\mu_{P(i)})-h'(\mu_{Q(i)})].$$

The entropy slope is decreasing. The implementation encloses its differences throughout the rectangle, caps them at the valid total variation bound two, integrates along line segments, and applies correct refinement-sign bounds. Monotonicity of finite min/max operations then encloses Delta. The 39 optimizations share one coverage event and require no separate factor of 39. Rejection only when the interval's lower endpoint exceeds 0.0005 has size at most 5% throughout the same fixed-loss null class.

The separate report-error intervals bound $h(q)+(\mu-q)h'(q)-h(\mu)$ for sample-fitted reports. They use the same coverage event but are not an interchangeable definition of Delta. Estimated state masses cannot be substituted for the known $p$ without a new argument.

## Full-domain oracle and pointwise power

The report domain is all of $[0,1]$, corresponding to the complete auxiliary interval $[-3/2,5/2]$. The oracle uses rational logarithm bounds, outward integer-square-root bounds, a positive-series exponential bound, and all four endpoint quotients for uncertain positive normalizers. Its narrow tail bounds apply only where the entire relevant interval is separated from both needles. Elsewhere, it returns conservative globally valid hinge and slope intervals. Therefore uniform coverage does not rely on the narrow local region.

For the reference posterior $(1/8,3/8,5/8,7/8)$ and one million observations, count and label events have failure budgets at most 0.1 each. Their intersection has probability at least 0.8. The count lower cutoff is 248642 for each state. The certificate controls the actual procedure's moving center, actual confidence radii, and entropy/slope interval errors, rather than only showing that a reference box has a large true Delta. Every one of the 15 full cell-mean intervals is inside the certified narrow region. The uniform widths are about $1.79434\times10^{-8}$ for entropy and $2.87094\times10^{-6}$ for slope.

On this event, the actual procedure's lower endpoint is at least 0.002025397175466932 (conservative decimal display), which exceeds 0.0005. Thus power is at least 80% at this one alternative. No further subtraction of 5% is needed because the event directly implies rejection. This is not uniform power over all $\Delta\ge0.001$, and it is not power estimated from a single observed success.

## Counts and interpretation

The retained sufficient statistics are $(N_i,K_i)=(250588,31304),(250072,93995),(249277,156057),(250063,218532)$, totaling one million synthetic observations. A multinomial state draw followed by conditional binomial label draws has the same count law as IID $(I,Y)$ samples under the model. A fixed PCG64 seed is one implementation of that model, not proof of ideal randomness or a million training records.

The exact saved-count interval has decimal display [0.0024967632816669787, 0.004042417685311304]. The frozen null is rejected. A frequentist 95% interval is not a post-data 95% probability statement about a fixed parameter. Only one preselected candidate received labels; the ten-candidate population screen did not produce ten confirmation p-values. The other two positive interior candidates have no labeled confirmation here.

## Corrections and evidence limits

The protocol was fixed before confirmation labels, with analytic work already underway. It is not an externally timestamped preregistration before all analysis. Local hashes, timestamps and an exclusive lock are consistent with one recorded attempt but cannot prove absence of unrecorded external trials. The current runner is single-process; an earlier description of an inference subprocess was corrected before the recorded draw. The confirmation receipt does not itself preserve numerical thread settings, so those settings are not independently authenticated by that receipt.

The historical code review initially compared a decoded tuple with a list; after correcting the test adapter, all assertions passed. A separate inspection initially treated a cell dictionary as a list. Neither error changed scientific source or data. These are preserved here as review implementation corrections rather than raw diagnostic logs.

The $n=8192$ floating-point population value falls about $3.9\times10^{-18}$ below a displayed rigorous lower endpoint. The original value remains unchanged: quadrature is diagnostic, while the population classifications use the independent analytic rational bounds. Earlier scale wording is corrected in the cross-stage errata. The Phase 9 auxiliary asymmetry check binds a research-source TeX copy, not the distinct final-manuscript TeX bytes, and has no located dedicated stdout log.

These records do not establish unknown-mass inference, standard-loss performance, neural representation learning, optimization success, generalization gains, resource savings, novelty priority, human-author approval, external peer review, or formal verification. The mathematical validity statements come from the derivations and model assumptions; assertion counts and local hashes have narrower roles.

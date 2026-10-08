# Phase 7: exact query memory and smooth-loss hierarchy limits

Wenyu Huang · University of California San Diego · Research working paper, 2026-10-07.
No institutional endorsement or external publication status is implied.

This phase contains two mathematical results: the exact continuous summary dimension for universal empirical quadratic residual queries, and a fixed smooth proper-loss construction with unbounded excess-risk hierarchy price for four equally weighted source states. It also includes a classical approximate-query corollary and an information-theory appendix. It contains no new neural training experiment.

Start with [the scientific status](SCIENTIFIC_STATUS.md), [the English working paper](output/pdf/phase7_working_paper.pdf), or [the Chinese decision note](PHASE7_DECISION_ZH.md). The [Chinese PDF](output/pdf/phase7_decision_zh.pdf) is a publication rendering with private operational material removed. [REPRODUCE.md](REPRODUCE.md) distinguishes reading saved evidence from running optional arithmetic or document builds.

For fixed unweighted sample size `n >= p+q`, where `p=d(d+1)/2` and `q=binomial(d+3,4)`, the minimum continuous real summary dimension is `2p+q`, or `p+q` additional coordinates when the label moment is supplied separately. These are 1,521 and 1,443 at `d=12`. The theorem requires exact answers on every dataset and every query. It is not a physical-memory, fixed-tolerance, Gaussian average-case, or learning-benefit lower bound.

The final finite construction uses one fixed symmetric smooth strictly convex generator and four source masses equal to `1/4`. For integer `j >= 16`, the deterministic excess price is `(j*j-j+8)/(2*j+4)` and the stochastic price is asymptotic to `j/4`. On the same examples the total-risk prices tend to one, and absolute excess costs vanish.

The [file map](FILE_INDEX.md) identifies final proofs and the preserved earlier Shannon-copy route. [TECHNICAL_REVIEW.md](TECHNICAL_REVIEW.md) summarizes historical internal checks and the resolved local-domain correction. Internal checking is not external peer review. [SOURCE_INDEX.md](SOURCE_INDEX.md) records historical project citations and their original content identities; third-party full texts are not redistributed.

Publication preparation copied the research and saved results, removed private operational records, and rebuilt only the sanitized Chinese document. It did not rerun arithmetic checks, random draws, training, data generation, or literature searches. The original research files were not modified.

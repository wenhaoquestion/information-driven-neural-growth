# Scientific status — Phase 7

Research evidence date: 2026-10-07. Public-copy preparation: 2026-10-08.

## Supported scope

The complete integrated proof is in `manuscript/main.tex` and its included sections; `output/pdf/phase7_working_paper.pdf` is the archived 21-page rendering. This is a working research artifact, without external peer-review or novelty certification.

1. **Exact residual-query information.** For centered quadratic empirical residual queries over every symmetric matrix and intercept, the canonical information is the label quadratic moment and raw input moments of degrees two and four. With `p=binomial(d+1,2)`, `q=binomial(d+3,4)`, and fixed unweighted `n >= p+q`, every continuous universally exact summary needs `2p+q` ideal real coordinates. That count is attainable. When the `p`-coordinate label moment is supplied separately, the additional count is `p+q`. At `d=12` the counts are 1,521 and 1,443. Every decoder-accessible data-dependent state must be counted.
2. **Local precision statement.** On a specified realizable compact family, simultaneous absolute error on the finite query family yields a bit-packing lower bound and a quantization upper bound with the same leading `D log2(1/epsilon)` term for fixed dimension, sample count and local radius. The local radius has no proved dimension-uniform lower bound.
3. **Smooth-loss hierarchy boundary.** One fixed symmetric `C-infinity([0,1])` generator has strictly positive interior curvature, finite endpoint slopes and bounded smooth strictly proper losses, yet four equal source masses admit deterministic excess price `(j*j-j+8)/(2*j+4)` for each integer `j >= 16`. The stochastic price is asymptotic to `j/4`. Curvature degenerates at the endpoints. Absolute distortions vanish and the same instances' total-risk hierarchy prices tend to one.
4. **Classical supporting results.** A Frequent Directions covariance guarantee transfers to uniform bounded-norm empirical residual queries. The information appendix supplies classical Bayes identities, a forgetting term for nonnested representations, and explicit counterexamples. A zero-output ReLU birth may add hidden label information while leaving current prediction risk unchanged. No novelty or learning guarantee is claimed for these principles.

## Preserved negative results and limits

- Label and second input moments alone do not determine every residual query; the explicit one-dimensional collision is preserved.
- Additivity alone does not imply a continuous-dimensional lower bound: discontinuous ideal-real encodings are a separate model. Small sample sizes, fixed teacher manifolds, a single optimizer trajectory and almost-sure decoding require separate analysis.
- Neither the continuous theorem nor local high-accuracy bound proves practical `Omega(d^4)` memory at a fixed useful tolerance, physical-byte costs, Gaussian average-case optimality or out-of-sample learning benefit.
- The smooth-loss theorem is an excess-risk result. It supplies no unbounded total-risk penalty, large absolute-loss guarantee, training-hardness theorem or full generator classification.
- Information gain alone does not guarantee fitted-risk improvement, a better classification decision or monotone gains under retraining. Activation entropy and input mutual information are not an operational memory/work ledger.
- The earlier Shannon-copy construction is retained as research history; the final equal-mass theorem is stronger and self-contained. Global four-state Shannon leading constants and general multi-level classifications remain open in this record.
- The dated literature audit retains access gaps for Lin (2010), Plaxton (2006) and Großwendt. Absence of a located precedent is not a priority proof.
- The Chinese decision note cites historical Phase 6 confirmation results, including failure to establish the requested 20% cost saving. Those observations are inherited context, not new Phase 7 training evidence.

## Existing evidence versus this preparation

`research/memory/check_results.json` records historical PASS results for polynomial feature ranks at `d=1..5`, Jacobian witnesses at `d=1..4`, exact query reconstruction and a fourth-moment collision. Its generator uses a seeded pseudorandom search for the Jacobian witnesses; it was preserved but not run here.

`research/finite/equal_mass_check_results.json` records historical exact rational assertions for `j=16,17,32,64,128,256,1024`, with all 18 compatible hierarchies enumerated at each value. The earlier `route_report.md` retains the saved Shannon-copy arithmetic table. These finite checks support algebra and enumeration; they do not numerically prove infinite smoothness, universal topology or an asymptotic limit.

Historical internal mathematical checks and the resolved compact-domain clarification are summarized in `TECHNICAL_REVIEW.md`. They are not external peer review. Their reported runs were not repeated in this publication task.

This preparation reviewed source text, parsed saved JSON and Python syntax, checked local manuscript dependencies and file hashes, and screened retained documents for private paths and operational records. Only the sanitized Chinese PDF was recompiled with shell escape disabled. No scientific check script, random-number generation, new experiment, training, package installation, web research or third-party communication was performed. The other three retained PDFs and both saved result JSON files preserve the original bytes. Future research suggestions remain unexecuted.

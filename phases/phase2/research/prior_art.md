# Phase II primary-source and novelty audit

Date: 2026-09-13. This is a targeted primary-source review, not a proof that no equivalent theorem exists. Search queries, sections actually inspected, and limitations are recorded below. The Phase I novelty review remains in `../phase1/research/novelty_audit.md`; it is not silently reclassified as Phase II work.

## Direction and permitted novelty claim

The useful direction that emerged is a separation between task information, the observations available to a structural controller, and the operations a parameterized learner can realize. The per-neuron splitting matrix and its negative-eigenvalue method are established. The proposed contribution must rest on precise counterexamples and operation restrictions, not on renaming this existing method as information-guided splitting.

A defensible narrow claim, subject to complete proofs and finished experiments, is: an explicit ordinary-probability tanh model exhibits task-pair indistinguishability under activation summaries despite task-dependent beneficial splits; a noisy-parity model separates fixed positive-weight local splitting from a positive-weight path approaching the boundary while keeping two children and preserving the incoming weighted mean. The latter has an explicit weight/radius scaling and logistic-loss rate. The first tests decision information; the second tests nonuniform operation limits. Finite-sample geometry-based training is empirical evidence of an accessible bridge, not a novel general splitting algorithm or a guarantee on realistic networks.

## Closest sources inspected beyond abstracts

### Splitting Steepest Descent (2019)

Primary: [NeurIPS proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2019/file/3a01fc0853ebeba94fde4d1cc6fb842a-Paper.pdf). Official record lists Lemeng Wu, Dilin Wang, Qiang Liu; the downloaded PDF prints Qiang Liu, Lemeng Wu, Dilin Wang. Use the official proceedings order if citing that record. Title: *Splitting Steepest Descent for Growing Neural Architectures*. NeurIPS 32 (2019).

Inspected Sections 2.1–2.3, Theorems 2.2–2.4, and Algorithm 1. They define the residual-weighted neuron-output Hessian, decompose parameter motion and mean-preserving splitting, and select equal-weight binary splitting along a negative eigenvector. The second-order formula is prior. Their positive-definite and negative-eigenvalue conclusions leave the zero-eigenvalue case outside that dichotomy; the text notes higher-order dependence. The proposed Phase II summary-observation pair and normalized positive boundary path are not stated in these inspected sections. Claiming discovery that loss geometry can guide neural growth would duplicate the prior framework.

### Signed splitting (2020 preprint; inspected v5, 2021)

Primary: [arXiv PDF 2003.10392](https://arxiv.org/pdf/2003.10392). Authors: Lemeng Wu, Mao Ye, Qi Lei, Jason D. Lee, Qiang Liu. Inspected PDF title: *Steepest Neural Architecture Descent: Escaping Local Optimum with Signed Neuron Splittings*. The arXiv abstract record retains an earlier title; a citation should identify the version rather than invent a conference publication.

Inspected Section 2, Theorems 2.1–2.3, and the immediate discussion. The optimization is of a second-order quadratic form under normalized signed weights, bounded outgoing absolute sum, bounded incoming offsets, and zero weighted mean. Signed binary splitting obtains descent whenever the splitting matrix is nonzero. When that matrix is zero, the quadratic criterion is zero. A high-order positive-weight construction would not refute these quadratic theorems. Any claim contrasting fixed positive weights with signed operations must acknowledge that signed splitting is already an established remedy in nonzero-curvature cases.

### Firefly Neural Architecture Descent (2020)

Primary: [NeurIPS proceedings PDF](https://proceedings.neurips.cc/paper/2020/file/fdbe012e2e11314b96402b32c0df26b7-Paper.pdf). Authors: Lemeng Wu, Bo Liu, Peter Stone, Qiang Liu. Title: *Firefly Neural Architecture Descent: a General Approach for Growing Neural Networks*. NeurIPS 33 (2020).

Inspected Sections 2.1–2.3 and Algorithm 1. A function neighborhood and complexity increment define allowed growth, including new neurons and depth. Candidate weights are optimized before an approximate integrated-gradient ranking selects retained additions. Thus neither learned candidate additions nor geometry-dependent structural selection is new. The computational advantage of any Phase II controller must include its candidate search and training cost. Firefly allows a broader operation family than fixed positive mean-preserving splits, so failure of the latter cannot be claimed as failure of Firefly.

### Stationary-plateau characterization (2026): closest to the high-order boundary

Primary: [arXiv PDF 2606.04327](https://arxiv.org/pdf/2606.04327). Tian Ding, Dawei Li, Ruoyu Sun, *A Geometric Characterization of the Stationary Plateau for Two-Layer Neural Networks*, preprint, 2026. Inspected Section 4.1, Theorems 1–3, and Section 6.3.1, PDF pages 9–11 and 22–23.

For square loss, the paper defines an inner Hessian and analyzes split stationary points. Its Theorem 3 covers locally effective neurons even when that Hessian vanishes. Its proof sets a child output coefficient to zero, then chooses a small coefficient perturbation and nearby incoming vector using residual correlation. Boundary-weight escape is therefore explicit prior work. It does not impose the Phase II combination of strictly positive normalized child weights at nonzero radius and an exactly preserved incoming weighted mean. The candidate contribution is that constrained separation and its explicit noisy-parity rate, not the general escape mechanism. The source establishes existence on a plateau, not a finite-sample controller guarantee.

### Limits of local learning (2016)

Primary: [author preprint 1506.06472](https://arxiv.org/pdf/1506.06472); [publisher record](https://doi.org/10.1016/j.neunet.2016.07.006). Pierre Baldi and Peter Sadowski, *A theory of local learning, the learning channel, and the optimality of backpropagation*. Neural Networks 83, 51–74 (2016).

Inspected Sections 6.4 and 7.1 (preprint pp23–25). The paper stresses that the allowed local variables must be specified, and argues that target-dependent learning requires a feedback channel. Its “in most cases” claim is explicitly acknowledged as informal rather than completely tight. Phase II differs by supplying an exact complementary-task witness despite permitting label and residual summaries; the missing information is their pairing with input geometry. However, the general principle that a learner cannot reconstruct unavailable target-dependent directions is established. Do not claim an impossibility of all local synaptic learning, since its usual local variables include presynaptic activity.

### Gradient indistinguishability and parity (2017)

Primary: [ICML paper](https://proceedings.mlr.press/v70/shalev-shwartz17a/shalev-shwartz17a.pdf); [official metadata](https://proceedings.mlr.press/v70/shalev-shwartz17a.html). Shai Shalev-Shwartz, Ohad Shamir, Shaked Shammah, *Failures of Gradient-Based Deep Learning*. PMLR 70, 3067–3075 (2017).

Inspected Section 2 and Theorem 1, including its parity calculation. Orthogonality of candidate targets makes gradients nearly independent of the target and connects parity failures to statistical-query limitations. Thus parity's invisibility to low-order information and a distinction between representation and gradient learnability are established. The Phase II fixed-dimension local Taylor and weight/radius limit statement is narrower and does not imply computational hardness of learning all parity functions. In particular, a single known parity is trivial for an unrestricted learner; a local split controller's failure is an operation restriction, not an information-theoretic sample impossibility.

### Infomorphic networks (2025)

Primary: [updated author preprint](https://arxiv.org/pdf/2306.02149), marked published in PNAS; [DOI](https://doi.org/10.1073/pnas.2408125122). Abdullah Makkeh, Marcel Graetz, Andreas C. Schneider, David A. Ehrlich, Viola Priesemann, Michael Wibral. *A general framework for interpretable neural learning based on local information-theoretic goal functions*. PNAS 122(10), e2408125122 (2025).

Inspected the neuron model, binned input distributions, Equations 8–9, and supervised experiment in Sections 2–3. The model trains stochastic neurons against local PID-based goals, with presynaptic vectors appearing in weight updates. Continuous inputs are binned and the binning gradient is ignored explicitly. This is concrete prior information-guided neural learning, not a neuron-growth proof. It prevents broad novelty claims about a first bridge from information to trainable local neurons. Phase II must not relabel ordinary conditional mutual information as a unique-information atom or apply the activation-summary impossibility to richer presynaptic observations used here.

## Newly derived exact obstruction versus standard reasoning

The complementary-task average-risk identity used by the auditor is elementary: averaging logistic risks for pointwise complementary posteriors yields `log 2 + E log cosh(f/2)`. With the same observation law, a randomized controller selects the same distribution over logits and cannot have positive expected immediate improvement on both tasks. This is a new use in the concrete witness; indistinguishability and convexity are standard proof methods. No separate claim of a new general no-free-lunch principle is justified.

## Limits

This historical search does not establish worldwide novelty. The exact positive-normalized, conserved-mean split theorem requires comparison with Ding--Li--Sun Section 6.3.1 and the SSD/signed-splitting feasible sets. The full proofs and all empirical limitations are in the manuscript.

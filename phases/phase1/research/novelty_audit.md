# Final primary-source novelty audit

Historical targeted literature review, 2026-09-13. This is not an exhaustive database review or proof of originality. The bibliography and source locations describe the historical search scope; they were not re-searched for this publication copy.

## Supported contribution statement

The precise claim supported by this audit is: **an explicit full-support four-source-state/binary-target family has an unbounded best simultaneous total-log-loss ratio at hard state budgets two and three, with matching family-specific deterministic and stochastic asymptotic constants; additional results establish a source-mass boundary and a divergent-endpoint-slope excess-loss extension.** The manuscript correctly distinguishes this from a worst-case matching rate over all four-state laws. None of the inspected primary sources establishes an equivalent theorem. The finite random-table argument and basic entropy identities are standard techniques; their application and matching feasible stochastic construction form part of this theorem package, not claimed new general representation lemmas.

The following must remain excluded from novelty claims: the information clustering objective; KL/Bregman centroid costs; qualitative nonnested optima; the four-point middle-pair versus adjacent-pairs pattern; greedy information clustering failure; information-bottleneck successive refinement and its neural interpretation; and the general idea of approximating KL cluster costs through Hellinger geometry. The manuscript's discussion of vanishing absolute losses and extremely rare source states is necessary for interpretation and is present.

## Closest predecessor: Kartowsky–Tal

Primary PDF: https://webee.technion.ac.il/people/idotal/papers/journal/DegradingUpgrading.pdf . Institution metadata: https://cris.technion.ac.il/en/publications/greedy-merge-degrading-has-optimal-power-law/ . Verified publication: Assaf Kartowsky and Ido Tal, IEEE Transactions on Information Theory 65(2), 917–934, February 2019, DOI 10.1109/TIT.2018.2879802.

Section III-A, Example 1, PDF page 3, gives a binary-input channel with four equally likely outputs. Its posterior points are 0, 1/3, 2/3, 1 after choosing label orientation. The greedy middle merge prevents the optimal two-pair result. Section III-A Theorem 1, same page, instead bounds additive degradation loss as a function of output alphabet budget; its power-law optimality is worst-channel scaling. The manuscript's description and explicit attribution are accurate. Its all-hierarchy, full-support, total-risk and stochastic asymptotic claims go beyond this inspected example and theorem. A bound for greedy degradation on the worst channel must not be described as an instancewise ratio to the optimal degradation of that same channel.

## AIB proceedings and separate extended manuscript

Official record: https://proceedings.neurips.cc/paper/1999/hash/be3e9d3f7d70537357c67bb3f4086846-Abstract.html . Official page metadata: https://proceedings.neurips.cc/paper_files/paper/1999/file/be3e9d3f7d70537357c67bb3f4086846-Metadata.json . Extended manuscript: https://www.cise.ufl.edu/~anand/sp06/ib.pdf .

The proceedings Section 1.2 (printed p619) defines the hard-cardinality information optimum; Section 2 gives the information-loss merge rule. The distinct extended manuscript Section 3.1, PDF page 6, explicitly observes that optimal partitions at consecutive cardinalities need not be connected by a merge. That section does not appear in the proceedings version. The manuscript's split attribution is correct. Bibliography now includes officially verified pages 617–623. The proceedings retains the official archive's NIPS 1999 convention; some references use the bound volume's 2000 publication year. The separate extended file has no explicit date, so its formerly asserted 1999 date was removed and its University of Florida course-archive provenance stated.

## Charvin–Catenacci Volpi–Polani

Primary PDF: https://uhra.herts.ac.uk/id/eprint/10690/1/entropy-25-01355.pdf . DOI: https://doi.org/10.3390/e25091355 . Verified first-page metadata: Hippolyte Charvin, **Nicola** Catenacci Volpi, Daniel Polani; Entropy 25(9), article 1355; published 19 September 2023. Corrected the erroneous given name Leandro in the bibliography.

Definition 5 is on PDF p8; Proposition 1's Markov characterization is on p10; Proposition 4's convex-hull characterization is on p12. Sections 2.3 and 3.2 contain numerical nonrefinability/soft-refinement investigations. Appendix D (p45 onward) concerns a uniqueness/injectivity conjecture, not the location of those examples. Their primal mutual-information constraint and hard alphabet cardinality are distinct resources. Their Section 4 already connects refinement to Blackwell comparisons and neural processing, so that interpretation is also prior. Technical citation corrections identified: replace the Appendix D locator by the relevant main sections; remove the contrast with “assuming an information-bottleneck Lagrangian,” since their paper explicitly defines the primal IB problem as well. The quantitative theorem package here is not established by their inspected results.

## Chaudhuri–McGregor: exact theorem location and stronger relevant precedents

Primary PDF: https://cseweb.ucsd.edu/~kamalika/pubs/cm08.pdf . Author publication record: https://people.cs.umass.edu/~mcgregor/research/research.html . Author CV confirming pages: https://people.cs.umass.edu/~mcgregor/goodies/mcgregor-cv.pdf . Official conference: https://www.learningtheory.org/colt2008/ . Verified citation: Kamalika Chaudhuri and Andrew McGregor, Finding Metric Structure in Information Theoretic Clustering, COLT 2008, 391–402.

**Theorem 8 is in Section 3, PDF page 5**, and gives a polynomial-time O(log n) approximation for flat minimum-total-KL clustering. Section 2 develops information geometry. More directly relevant to the manuscript's positive result: Section 3.2 Lemma 12 (PDF p6) compares total KL and Hellinger cluster costs logarithmically in the number of equally weighted points; Section 4 Lemma 18 (PDF p7) bounds KL by a logarithmic multiple of Jensen–Shannon under a bounded coordinate likelihood ratio. The manuscript proves its weighted source-mass and four-point hierarchy bound independently, but calling this only a “comparison philosophy” understates the concrete prior inequalities. Requested citation wording should acknowledge these comparisons explicitly. No matching sharp hierarchy-rate claim is warranted by the positive bound.

## Remaining bibliography verification

- Arutyunova and Röglin: official ESA 2022 record https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2022.10 verifies title, authors, volume 244, pages 10:1–10:14 and DOI. The primary PDF's Introduction and Section 2 define the simultaneous hierarchy criterion; its results concern metric radius/diameter. Existing citation is correct.
- Banerjee, Merugu, Dhillon and Ghosh: publisher record https://jmlr.org/papers/v6/banerjee05b.html verifies Clustering with Bregman Divergences, JMLR 6, 1705–1749 (2005). The main text properly treats this framework as prior.
- Gneiting and Raftery: primary https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf verifies title, authors, JASA 102(477), March 2007, pages 359–378 and DOI 10.1198/016214506000001437. Its scoring-rule/entropy connection is background.

## Additional adversarial search and inspection

Queries included `hierarchical Kullback unbounded clustering approximation`, `price of hierarchy entropy clustering`, `successive refinement cardinality logarithmic loss`, `hierarchy Bregman lower bound`, and exact titles/author names. Results were screened through primary papers rather than relying on search snippets for mathematical assertions.

Two additional possible matches were inspected: Aghagolzadeh, Soltanian-Zadeh and Araabi, *Information Theoretic Hierarchical Clustering*, Entropy 13, 450–465 (2011), DOI 10.3390/e13020450, primary publisher PDF mirrored at https://pdfs.semanticscholar.org/6069/05082bb33d319b723a2b65229ea61a6c13ed.pdf , uses quadratic mutual-information heuristics and experiments (Sections 2.4 and 3); no equivalent all-hierarchy lower theorem was found. Tan, Tegmark and Chuang, *Pareto-Optimal Clustering with the Primal Deterministic Information Bottleneck* (2022), primary https://arxiv.org/pdf/2204.02489 , treats the entropy/information Pareto frontier and algorithmic search; its central budget is representation entropy and its Pareto frontier differs from the pair of simultaneous hierarchy-risk coordinates here. These checks reinforce the need to avoid broad novelty claims about information-driven clustering or primal-versus-Lagrangian formulation.

## Disposition

The inspected sources did not supply a novelty-equivalent theorem. This is a bounded historical search finding, not a guarantee of worldwide originality. The final manuscript includes the Charvin section locator correction and explicit attribution to Chaudhuri--McGregor Lemmas 12 and 18.

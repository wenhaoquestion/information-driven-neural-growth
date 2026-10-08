# Fixed-amplitude additive-risk study

This bounded four-state study asks whether existing hierarchy-price witnesses have a statistically detectable *additive* conflict after fixing the full-report-domain loss oscillation to exactly one. The [public technical report](PILOT_RESULT_ZH.md), [population proof](theory/FIRST_PASS_THEORY.md), [statistical derivation](statistics/DERIVATION.md), and [final statistical review](../../reviews/statistical-closure/TECHNICAL_REVIEW.md) state the assumptions and limits.

Three interior candidates have certified population conflict above 0.001. All seven endpoint candidates have upper bounds below 0.001. Only the preselected interior $n=2048$ candidate received labels: one recorded fixed-seed aggregate draw with one million synthetic observations. The saved four-state counts yield the exact 95% interval whose decimal display is **[0.0024967632816669787, 0.004042417685311304]**, rejecting the frozen null $\Delta\le0.0005$.

The procedure has at most 5% size throughout the posterior cube for the same known masses and fixed public loss. Its at least 80% power is an analytic, pointwise statement at the specified construction. One observed rejection is not an empirical power estimate. No neural network was trained.

## Evidence

- `PROTOCOL.json` and `qa/protocol_freeze.json`: selection-before-labels constraints, parameters, disclosed timing. Analytic work had already begun before the protocol freeze; this is not external preregistration before all analysis.
- `theory/`: rigorous population and full-domain oracle derivations, exact rational bounds, independent ideal-model check.
- `computation/`: all 150 cell costs, 150 partition costs, 390 compatible hierarchy costs, the 64/128-point diagnostic comparison, and the preserved small floating-point discrepancy.
- `statistics/`: exact inference and pointwise power implementations, full certificates, reference model and all four saved state/label counts.
- `historical_snapshot/`: necessary earlier mathematical and numerical comparison inputs. Historical relative identifiers are retained as provenance names; executable path lookup is adapted to this directory.
- `../../reviews/statistical-closure/`: two independent deterministic implementations, their historical result records and instructions for a new public replay receipt.

Read [SCIENTIFIC_STATUS.md](SCIENTIFIC_STATUS.md), [REPRODUCE.md](REPRODUCE.md), and [PUBLICATION_PROVENANCE.md](PUBLICATION_PROVENANCE.md) before interpreting source hashes. Public metadata bindings identify these publication copies; they do not authenticate the historical execution anew. Raw counts and every scientific numeric value are preserved.

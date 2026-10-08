# Publication provenance and metadata bindings

The source experiment is the 2026-10-07 record. This public edition preserves its scientific values and exposes the transformations required for publication. `PUBLICATION_PROVENANCE.json` lists original and publication SHA-256 digests, byte counts and the transformation for each file. The private source was not edited.

## Preserved bytes and data

The original bytes of `honest_delta.py`, `infer_confirmation.py`, `certify_interior2048.py`, `local_loss_oracle.py`, `certify_bounds.py`, the complete population/oracle/power certificates, and `COUNTS_PUBLIC.json` are retained. The four count pairs, all exact rational endpoints, all 15 report-error bounds, all 39 hierarchy bounds, the one-million-label budget, seed/reference parameters, all ten-candidate numerical arrays and the negative results are unchanged. No labels, quadrature or training were regenerated to prepare this edition.

## Publication derivatives

Machine interpreter/platform fields were removed from affected JSON metadata. The population script's historical lookup root now points to the bundled `historical_snapshot/`; its arithmetic is unchanged. The simulator's coordination-oriented help/docstring was neutralized without changing line positions, sampling operations or its exclusive lock. It was not run. Technical prose replaces internal planning and delivery language; necessary errors and limits are retained.

The original raw-result freeze and execution receipt hashes referred to the original bytes. Where publication metadata or source paths changed, **the public copies' digest bindings were refreshed to the corresponding public files**, and they carry `publication_binding_notice`. These refreshed hashes establish current publication-copy consistency. They do not claim that the public derivative existed at the historical event time, re-authenticate an old execution, or prove that unrecorded trials never occurred. Historical event times remain historical; original and new file digests are distinct entries in the ledger. Textual historical hash references are interpreted through that ledger.

The final independent review implementations were adapted only for public input paths and distinct output filenames. Their original 335/827-assertion results remain in `../../reviews/statistical-closure/historical_results/`. A current deterministic replay produces new `PUBLICATION_*` receipts; it is not a second statistical confirmation. Publication verification should compare exact scientific fields and current source hashes, not timestamps or resource timings.

The provenance ledger and checksum inventory exclude themselves to avoid circular hashes. A repository commit provides the outer version boundary. Public machine logs, private correspondence and third-party full texts are not included.

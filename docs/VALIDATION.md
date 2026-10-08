# Release validation — 8 October 2026

The checks below were performed on fresh private Git checkouts of the sanitized publication tree. The source research archive was not used as the test working directory. No new neural training, random generation, bootstrap, dependency installation or scientific experiment was performed. Machine-readable outcomes are in [validation_checks.json](validation_checks.json).

## Completed checks

| Check | Result and limit |
|---|---|
| Main paper builds | All twelve catalogued English papers built successfully with shell escape disabled. Final passes had no undefined citations or references. Supporting PDFs were separately inspected during stage preparation; they are not all included in the twelve-build claim. |
| Phase 1 saved records | Passed: 15 scaling rows, 10 sensitivity rows, 225 partition records, 270 chain records and four historical precision records. No new population calculation. |
| Phase 2 saved records | Passed: 1,280 runs, 3,840 events, 192 probes and 20,000 scores; recomputed risk discrepancy at most approximately 1.11e-16. |
| Phase 3 saved records | Passed: 1,088 trajectories and 8,704 events, including parameter chains, gate arithmetic, cost accounting and summary means. |
| Phase 4 public evidence | Passed: 336 run summaries, 8,064 events, 128 contrasts, 153 base and 96 post-hoc groups, plus included raw/frozen-file hashes. Student-t quantiles and omitted raw runs were not recomputed. |
| Phase 5 public evidence | Passed: 768 endpoints covering 5,376 historical events, 1,440 checkpoint branches and 1,024 estimator-stage rows. An independent publication review additionally compared endpoint and branch rows with all 144 original raw JSON files. No training replay. |
| Phase 6 public evidence | Passed: 7,040 trajectory rows, all 1,408 reconstructed AUCs, mean contrasts, cost sums, 118/256 fixed-width wins and eight nonattainers. The historical bootstrap decision was read, not recomputed. |
| Phase 7 finite arithmetic | Both equal-mass cost and four-state transfer scripts completed successfully without random generation. This finite check does not establish the universal memory or smoothness theorems. |
| Additive-risk closure | Independent JSON arithmetic check: 335 checks passed. Independent source/count/inference check: 827 checks passed. The original confirmation counts were reused; this was not a second confirmation or an empirical power study. |
| Parse checks | All 146 Python sources parsed; all 230 JSON files and eight compressed JSON payloads in the tested snapshot parsed. Later additions were release metadata only. |
| Document navigation and release bytes | The release checker passed relative Markdown file links and file hashes. The final manifest includes nested stage manifests and is checked against the Git index, so a tracked file cannot silently fall outside its coverage. External URLs were not exhaustively retested. |
| Privacy and redistribution review | All 71 supplied PDFs were text-extracted and their metadata inspected; all eight gzip payloads, three NPZ headers and image text metadata were inspected. Static patterns and an independent content review found no unresolved private conversations, credentials or machine paths. Public bibliography URLs containing `/home/` were reviewed as valid source URLs. No source third-party literature PDF was redistributed. This is an internal publication review, not external scientific peer review. |

The successful clean-checkout environment used Python 3.14.6 and pdfTeX 1.40.27 from TeX Live 2025. It used standard-library checks only; no scientific Python dependency was installed or exercised by the saved-data checks. The numerical package versions recorded elsewhere describe the historical research environment, not a newly tested compatibility matrix.

The first version of the unified builder passed absolute output paths to BibTeX, which its write policy rejected. The builder was corrected to run BibTeX inside the isolated output directory with explicit bibliography search paths. A new clean checkout then passed all twelve builds. The source manuscripts were not changed to accommodate this fix.

## What was not validated by this release

Full neural training, stochastic exploration, bootstrap, all large raw-array audits, empirical power estimation and the entire optional historical reproduction pipeline were not rerun. Read the [reproduction guide](REPRODUCING.md) and [exclusion boundaries](EXCLUDED_ARTIFACTS.md) before invoking those entry points. PDF compilation, finite calculations, hashes and independent internal checking do not constitute proofs of all theorems, external peer review, priority certification or author sign-off.

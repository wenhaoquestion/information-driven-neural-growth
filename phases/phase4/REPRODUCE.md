# Reproduction and historical evidence

Commands below run from the repository root. `python3` must refer to the intended environment. `requirements.txt` and `ENVIRONMENT.json` record the historical dependency versions; no dependency installation or scientific rerun occurred during repository preparation.

## Minimum smoke check: no random draws or training

```sh
python3 phases/phase4/code/verify_publication.py
```

This standard-library-only command parses Python source, checks all 336 run and 8,064 event summaries, recomputes means/standard deviations/standard errors for 128 paired contrasts, checks 153 base and 96 post-hoc groups, verifies six included raw-file hashes, and verifies the four original frozen protocol/code hashes. It does not recompute Student-t quantiles, train networks, create random samples, or certify theorem correctness.

## Build the manuscripts

The sources use relative paths and include generated bibliography text where needed. With an existing TeX installation, work in a copy or set an output directory to preserve the supplied PDFs. From each of `publication/finite/manuscript`, `publication/evidence/manuscript`, and `new_research/evidence/manuscript` under this phase directory, run:

```sh
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
```

All three PDFs were rebuilt locally. The finite source only removes internal author-block comments; the other two have narrowly scoped archival wording and data-availability edits. Theorems and numerical conclusions were preserved. TeX and bibliographic versions can affect pagination and byte hashes.

## Regenerate plots from the distributed results

In a disposable copy, the following commands read saved results only; they perform no training or random draws. They overwrite derived figures/tables:

```sh
python3 phases/phase4/experiments/code/plot_saved_summaries.py
python3 phases/phase4/code/plot_retention.py
python3 phases/phase4/publication/finite/code/plot_results.py
python3 phases/phase4/publication/finite/research/plot_sharp_rate.py
```

The summary-only neural plotter uses all 336 run summaries and 8,064 events. The original raw-driven `summarize.py` now refuses incomplete historical run sets rather than silently replacing complete summaries with the single bundled run.

## New scientific computations: explicitly separate from the smoke check

These commands generate new outputs and are not claimed to have run during preparation. Use a disposable copy and a new destination. Do not point experiment runners at historical output directories.

A complete retained-evidence replay, including the original seeded R1/R2 draws and the separately labelled post-hoc supplement:

```sh
python3 phases/phase4/code/reproduce_retention.py --output-dir phases/phase4/replays/retention_new
```

The wrapper refuses an existing destination. It compares expanded raw JSONL digests against the original hashes in `RAW_ARTIFACTS.json` when the large original JSONL file is not bundled, compares summaries byte-for-byte, and compares all 62 saved count arrays. Seed formulas and sampling conditions are in `research/RETENTION_EXPERIMENT_PROTOCOL.md`; the post-hoc chronology is retained in its addendum.

A new seven-arm neural confirmation:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 phases/phase4/experiments/code/recycling.py --config phases/phase4/experiments/config.json --output-dir phases/phase4/experiments/reproduction_new/results --workers 2
```

The runner expects the output to be under its experiments directory because stored data paths are relative to that directory. It does not refuse an existing directory; choose a new one. Tasks, all twelve seeds, horizons, optimizer budgets, and methods are in `experiments/config.json` and the frozen protocol.

A full representative seven-arm historical comparison is possible with the bundled interaction seed 750000:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 phases/phase4/experiments/code/replay_confirmation.py --task interaction --seed 750000 --output-dir phases/phase4/experiments/replay_new/results
```

This retrains and compares history at the saved tolerance, excluding elapsed time. It is not a smoke check. `check_recycling.py` also generates samples and runs implementation training checks, so it must not be used as a no-RNG check. The full historical audit, `integrity.py`, the raw-driven summary plotter, and `research/independent_report_checks.py` need all original files named in `RAW_ARTIFACTS.json`, restored at their recorded relative paths. Only the representative original raw run is bundled. The historical `run_index.json` deliberately lists all 48 original files and their hashes, not only the distributed subset.

The finite `publication/finite/code/reproduce.py` reruns exhaustive calculations, randomized mathematical diagnostics, plots, and TeX; run it only in a disposable copy. Seeds 191733 and 20260914 are preserved in the mathematical check scripts. `research/independent_checks.py` uses seed 91527000; its optional duplicate simulation was unexecuted in the historical record and remains exploratory code. No scientific conclusion relies on outputs from that optional branch.

## Omitted raw files and historical metadata

[RAW_ARTIFACTS.json](RAW_ARTIFACTS.json) gives the relative path, exact byte size, original SHA-256, and included/archive-only status of each scientific raw file. It has no download URL: most raw files remain in the local archive. A user with those files can verify `sha256sum` (or `shasum -a 256`) against the ledger. The large retention JSONL tables can also be regenerated by the wrapper above from their original seeds.

`results/retention_audit.json`, the reports under `replays/` and `experiments/replay/`, and `publication/finite/results/reproduction.json` are historical outcomes, not claims of checks performed for this repository. Some original command-executable paths were normalized to `python` for portability. Their stored source or PDF hashes refer to the original research artifacts, which can differ from editorially cleaned public files. The original neural frozen files remain byte-identical.

The optional Phase 3 recovery script additionally requires `phases/phase3/code/betting_growth.py` and the original `phases/phase3/results/betting_followup4096.json`. The latter is distributed losslessly as `phases/phase3/evidence/betting_followup4096.json.gz`; the Phase 3 manifest records its hash. To restore the required result from the repository root, run:

```sh
python3 phases/phase3/code/restore_evidence.py
```

That restoration performs no training or random sampling and refuses conflicting files. It was not run during Phase 4 preparation. The archived Phase 3 recovery result is included here; executing `experiments/code/recover_phase3.py` performs neural retraining and is outside the minimum smoke check.

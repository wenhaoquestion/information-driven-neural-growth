# Reproducing and auditing the archive

Use a fresh working copy. Python dependencies are recorded in the root and stage-specific requirements files; they are not bundled. The paper builds require an existing TeX distribution with the packages named in the manuscripts. No command below installs software. Historical outputs, newly performed release checks and optional future experiments are distinct.

## Check the distributed release

From the repository root:

```sh
python3 -B scripts/check_release.py
```

This checks the public file hashes, all ten stage entry points and local Markdown file links. It does not validate external URLs, prove theorems or authenticate historical execution. `PUBLICATION_MANIFEST.json` describes the distributed bytes, including editorial derivatives; it is not the original experiment manifest.

## Check saved numerical evidence without training or random generation

The following checks read published evidence or recompute deterministic mathematical quantities. They require no new data collection or neural training. Use Python 3.10 or later; the listed checks use standard-library arithmetic and parsing only.

```sh
python3 -B phases/phase1/code/verify_saved.py
python3 -B phases/phase2/code/verify_saved.py
python3 -B phases/phase3/code/verify_saved.py
python3 -B phases/phase4/code/verify_publication.py
(cd phases/phase5 && python3 -B code/verify_public.py)
(cd phases/phase6 && python3 -B code/verify_public.py)
(cd phases/phase7 && python3 -B research/finite/check_equal_mass_costs.py)
(cd phases/phase7 && python3 -B research/finite/check_four_state_transfer.py)
python3 -B reviews/statistical-closure/theory/independent_verify.py
python3 -B reviews/statistical-closure/code/independent_code_check.py
```

The closure commands write new `PUBLICATION_THEORY_CHECK.json` and `PUBLICATION_CODE_CHECK.json` files beside their scripts. These are new checks, distinct from the preserved original outcomes. The other listed checks print their outcomes. The Phase 2–3 checkers read compressed audit tables directly; the stage restore helpers are only needed for optional legacy entry points. Phase 5–6 checkers validate the published tables but do not certify omitted raw arrays. Phase 6 reads the historical bootstrap decision and independently reconstructs AUCs and means; it does not rerun that stochastic bootstrap.

Phase 1’s checker verifies the published finite tables and structural record counts from saved evidence; it does not rerun the population evaluation or prove the mathematical claims. Phase 8–10 include their original computations, configs, figures and proof audits. Their full numerical entry points are described in their stage guides and are not silently included in a saved-data check. Exact arithmetic or typesetting success does not prove the universal mathematical statements.

## Build all twelve catalogued English papers

With `pdflatex` and `bibtex` already on `PATH`:

```sh
python3 -B scripts/build_papers.py
```

The builder uses a fresh `_build/papers/` directory, copies each stage into an isolated build tree, disables shell escape, and runs the required LaTeX/BibTeX passes. It refuses an existing output directory. It checks command success and unresolved references, and writes `BUILD_RESULTS.json`. The archival PDFs and LaTeX remain unchanged. To select papers or a different new destination:

```sh
python3 -B scripts/build_papers.py --paper phase4-finite --paper phase10 --output _build/selected
```

The [catalogue](papers.json) defines the supported main documents. Supporting notes and Chinese materials have additional instructions inside each phase. Platform-specific Chinese fonts are not redistributed. Different TeX releases or PDF metadata can change bytes without a scientific text change. A successful PDF build is a typesetting check, not proof review.

## Regenerate experiments only when intended

Each `phases/phaseN/REPRODUCE.md` separates minimum checks from full simulations, training, seeded diagnostics, bootstrap and plotting. Many historical entry points overwrite fixed filenames. Use a disposable copy and a new output directory as directed. Preserved seeds reproduce the intended study design, but floating-point libraries, runtime metadata and compressed-container timestamps can affect byte equality.

Some historical audits require the large local raw files listed in the [exclusions and recovery guide](EXCLUDED_ARTIFACTS.md). The release does not contain those arrays and supplies no public raw-data download. Included hashes identify original artifacts; they do not substitute for their bytes. Rerunning with changed hyperparameters, selection rules or endpoints constitutes a new study rather than confirmation of the old one.

See [VALIDATION.md](VALIDATION.md) for the exact checks actually performed for this publication.

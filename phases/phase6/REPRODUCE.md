# Reproduce Phase 6

From this phase directory, check the included public record with Python 3.10+:

```sh
python3 -B code/verify_public.py
```

This read-only standard-library check verifies public hashes and source syntax, reconstructs all saved AUCs and mean contrasts from the 7,040 included trajectory rows, checks cost sums and the 118/256 fixed-width win count, and reads the archived primary decision. It does not recompute bootstrap or certify the omitted raw parameters/data. No RNG is used.

Build the English paper from included inputs with an existing TeX installation:

```sh
python3 -B qa/build_working_paper.py --output-dir reproduce/paper
```

This uses `pdflatex -no-shell-escape` and BibTeX. Required packages are declared in `manuscript/main.tex`; all figures and generated tables are included. The clean Chinese Markdown is public; the historical Chinese PDF is omitted because it contained private execution metadata. No Chinese-font toolchain is required to build the English paper.

Historical numerical dependencies: Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and Matplotlib 3.11.2. The environment and dependencies are not bundled; see `requirements.txt`. Byte-level NPZ hashes and wall times are not expected to be portable across platforms.

The following optional commands perform scientific training and stochastic analysis. They are supplied for future reproduction and were **not run during publication**. They require an already provisioned numerical environment and new output directories:

```sh
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -B code/run_stage.py --config configs/confirmation_primary_selected.json --output-dir reproduce/confirmation_primary --workers 2
python3 -B code/run_stage.py --config configs/confirmation_secondary_selected.json --output-dir reproduce/confirmation_secondary --workers 2
python3 -B code/run_stage.py --config configs/age_sensitivity_selected.json --output-dir reproduce/age_sensitivity --workers 2
python3 -B analysis/analyze.py --primary reproduce/confirmation_primary --secondary reproduce/confirmation_secondary --age reproduce/age_sensitivity --development-selection analysis/development_selection/selected_lrs.json --output-dir reproduce/analysis
```

The last command uses the archived selected rates to reproduce the original question. It runs the fixed 10,000-resample bootstrap sensitivity. For development replication, run `configs/development_lr0075.json`, `development_lr015.json` and `development_lr03.json` into corresponding new directories, then use `analysis/analyze.py --select-development DIR0075 DIR015 DIR03 --output-dir reproduce/development_selection`. A newly selected rate differing from the archived choice defines a new study, not a rewrite of the original confirmation.

Full saved-state audits require the omitted raw NPZ/JSON, or regenerated counterparts. Use `reviews/audit_saved_results.py --run-dir reproduce/confirmation_primary --output reviews/reproduced_primary_audit.json`; the audit deliberately requires its output below `reviews/`. Historical snapshot hashes are original identities, and some earlier smoke versions differ from the final runner. Do not mix earlier-stage files into a later frozen run.

`code/test_frontier.py`, `analysis/analyze.py --self-test` and several audit scripts invoke RNG or training fixtures. They are not part of the nonstochastic publication check. The original protocols are historical records, accompanied by `PROTOCOL_CLARIFICATIONS.md`; public edits remove only private workflow instructions and path metadata. See `PUBLICATION_NOTES.md` and `EXCLUDED_ARTIFACTS.csv` for original-byte identity and omissions.

# Reproducing Phase 2

All reported measurements are historical. Publication preparation did not run training, RNG simulations or new experiments. Historical Python 3.12 dependencies are pinned in `requirements.txt`; no package installation is implied by these instructions. Run commands from `phases/phase2` unless otherwise stated.

```sh
python code/verify_saved.py
```

This read-only standard-library smoke check reads saved JSON and compressed evidence directly. It verifies archive hashes and saved arithmetic/history/summary consistency without generating random numbers, fitting parameters or writing results. It does not certify proofs or reconstruct unshipped population samples. Original research audit scripts may generate seeded samples and train models; they are not interchangeable with this safe entrypoint.

```sh
python code/restore_evidence.py
python code/reproduce.py --no-build
```

Restoration verifies compressed and uncompressed SHA-256 and refuses to overwrite conflicting files. It restores exact historical bytes, including all main fitted parameters, candidate and gate records. The default publication reproduction driver verifies saved evidence and regenerates plots/tables; it does not train. It may overwrite regenerated figures/tables, so use a working copy if preserving edits. Omit `--no-build` to compile the supplied paper with pdfLaTeX, BibTeX and `-no-shell-escape`. Add `--no-plots --no-build` for restoration and saved checks only.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python code/reproduce.py --full
```

`--full` explicitly runs all principal historical neural batches into `results/replayed/` and compares stored scientific fields at relative 1e-9 / absolute 1e-11 tolerance; timing is excluded. This is an optional future retraining action, not a result of publication preparation. Frozen configurations/seeds are embedded in the exact archived JSON, and public code keeps the original algorithms. Historical code hashes identify original source bytes; portable-path and documentation changes mean public source hashes may differ.

Large per-trial data and duplicate replays remain local and are enumerated in `RAW_MANIFEST.json`. Necessary main parameter/event records are the documented lossless-gzip exception in `evidence/manifest.json`; each archive is under 3 MB. SHA listings identify bytes and do not provide a download service or prove correctness. No third-party full PDFs, environment/cache directories, release ZIP duplicates, sensitive logs, session prompts or internal dialogues are included.

Optional non-neural regeneration: `python code/high_order.py` evaluates exact populations; `python code/score_sampling.py` intentionally regenerates 20,000 seeded score trials. The Phase I cross-check `python code/audit_quartet.py` resolves the adjacent `../phase1` study. Historical full neural replay passed at release according to `results/reproduction.json`; that record is historical and its machine-specific command prefix has been replaced by `python`.

# Reproduce Phase 5

Run the following from this phase directory. Python 3.10+ is sufficient for the read-only public checker:

```sh
python3 -B code/verify_public.py
```

It verifies release hashes, source syntax, the 768 published endpoints / 5,376 events / 1,440 branches, their agreement with saved summaries, and all 1,024 published estimator-stage rows. It does not certify omitted raw arrays or replay training. Original archive-wide `code/verify.py` is omitted because it requires private whole-project preservation manifests and the entire historical workspace.

Build the English paper using an already available TeX installation:

```sh
python3 -B code/build_paper.py --output-dir reproduce/paper
```

The build uses `pdflatex -no-shell-escape` and BibTeX. It requires the packages declared in `manuscript/main.tex`; all figure/table inputs are included. The saved English PDF is the historical compiled artifact.

The historical numerical environment was Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and Matplotlib 3.11.2. See `requirements.txt`. Dependencies and the virtual environment are not bundled. Floating-point trajectories and NPZ container bytes may differ across numerical libraries or platforms.

Full regeneration is optional and performs scientific simulation/training; these commands were **not executed during publication**. Use a disposable copy of this phase. All destinations below must be new, and must stay inside that phase directory because the runners serialize phase-relative paths. In the disposable copy, first move the included `experiments/{confirmation,adaptive_confirmation,fullbatch_confirmation}` metadata directories to a backup location, then run:

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
python3 -B code/moment_diagnostics.py --output results/moments_replay --reps 64
python3 -B code/quadratic_experiment.py --config experiments/confirmation.json --output-dir experiments/confirmation/results --workers 2
python3 -B code/adaptive_quadratic.py --config experiments/adaptive_confirmation.json --output-dir experiments/adaptive_confirmation/results --workers 2
python3 -B code/quadratic_fullbatch.py --config experiments/fullbatch_confirmation.json --output-dir experiments/fullbatch_confirmation/results --workers 2
```

The frozen neural seeds are 820000–820011, 840000–840011 and 850000–850011 respectively on each of four tasks; estimator seeds are 830000–830063 at each of two noise levels. Development seeds/configuration and the original moment-runner snapshot are also included. See `experiments/PROTOCOL.md`, `ADAPTIVE_PROTOCOL.md`, and `FULLBATCH_PROTOCOL.md` for scientific rules; their old workspace commands are historical and are superseded by this document.

After raw regeneration, the `summarize_quadratic.py`, `summarize_adaptive.py`, `summarize_fullbatch.py`, and `nonlinear_projection.py` scripts read those fixed raw locations. Run them only in the disposable copy: they overwrite derived tables/plots. `plot_moments.py` can already use the included `results/moments/raw.jsonl`; it also overwrites derived artifacts. For exact legacy moment auditing, use `results/moments/runner_snapshot.py` and its recorded hash; current `moment_diagnostics.py` changes only the documented accounting label.

Historical review scripts may invoke RNG, intervention simulations, or training even without a `--replay` option. They are archived reconstruction tools, not minimal publication checks. All full raw hashes are in `EXCLUDED_ARTIFACTS.csv`; these hashes identify retained local originals and are not a remote dataset download or a guarantee of byte-identical reruns.

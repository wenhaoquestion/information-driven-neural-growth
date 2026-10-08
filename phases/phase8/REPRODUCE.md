# Reproduction

The archived numerical results are the original recorded outputs. Publication preparation copied and checked files without running experiments, training, RNG, or installing dependencies. The main paper was rebuilt in three pdfLaTeX passes with shell escape disabled: its 37-page extracted text exactly matched the archived PDF. Source references and script syntax were checked; new scientific results were not generated.

To build the v2 paper with an existing TeX installation:

```sh
cd manuscript
sh build.sh
```

To rerun the archived finite calculations deliberately in a disposable copy of this phase directory:

```sh
PHASE8_PYTHON=python3 sh research/experiments/code/reproduce.sh
python3 research/audit/audit_checks.py
```

The first command overwrites its own result/figure files and uses the fixed recorded random seeds for finite Markov-channel checks. It is not a neural training run. It needs NumPy, SciPy, mpmath and Matplotlib already installed; the algebra/geometry audit uses only the Python standard library. The shell wrapper defaults to `python3`, accepts an existing interpreter through `PHASE8_PYTHON`, creates its local log directory, and fixes numerical thread limits at one.

The recorded original environment was Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, mpmath 1.3.0 and Matplotlib 3.11.2. This is environment provenance, not a newly tested compatibility matrix. Full settings and original seeds are in `research/experiments/configs/run_config.json`. Original result hashes refer to the source version used for those runs; publication edits to interpreter paths do not imply a rerun.

Raw execution and machine-environment logs are excluded from the public tree. Their actual serialization and quadrature failures are preserved in `TECHNICAL_NOTES.md` and the experiment report. Finite floating-point agreement is not interval certification or a proof of the asymptotic theorem.

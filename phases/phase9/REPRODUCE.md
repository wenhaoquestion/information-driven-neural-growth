# Reproduction

The saved outputs are original recorded results. Publication preparation did not run experiments, training, RNG, or dependency installation. The main paper was rebuilt in three pdfLaTeX passes with shell escape disabled: its 23-page extracted text exactly matched the archived PDF. Files, source references and script syntax were checked without generating new scientific results.

Build the paper with an existing TeX installation:

```sh
cd manuscript
sh build.sh
```

To deliberately rerun the finite checks in a disposable copy of this phase:

```sh
PHASE9_PYTHON=python3 sh research/experiments/code/reproduce.sh
python3 research/prior_art/check_exact_asymmetry.py
```

These commands overwrite their own result and figure files. They use deterministic exact-arithmetic and quadrature calculations, not neural training. Existing NumPy, SciPy, mpmath and Matplotlib are required. The portable wrapper defaults to `python3`, accepts `PHASE9_PYTHON`, creates its local log directory and sets numerical thread limits to one. Settings are encoded in the source scripts and saved JSON; no separate configuration file existed in the source archive.

The recorded original environment was Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, mpmath 1.3.0 and Matplotlib 3.11.2. This is historical environment information, not a newly tested compatibility claim. Original result hashes describe the original scientific source version; interpreter-path edits for publication are not evidence of a rerun.

Machine-environment paths and private execution logs are excluded. The original harmless font-cache warning and correction are described in the technical notes. Floating-point crosschecks do not provide interval-certified errors. In particular, the 100-digit independent coordinate check covers four base integrals only, not all tiny weighted costs or comparison products.

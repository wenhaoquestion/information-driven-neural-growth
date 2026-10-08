# Reproducing Phase I

Historical dependency pins are in `requirements.txt` (Python 3.12). No installation, training or RNG run was performed when preparing this copy. From `phases/phase1`:

```sh
python code/verify_saved.py
python -m json.tool results/reproduction.json
```

The checker verifies stored partition minima, chain maxima/minima, saved stochastic brackets and historical precision records using the standard library; it does not recompute the population or generate random numbers. To intentionally recompute the deterministic finite populations, run `python code/quartet_experiment.py`; this overwrites its named result files. `python code/plot_results.py` regenerates figures and the numerical table from saved CSVs. `python research/onset_experiment.py` evaluates the separate proposed onset construction, not a neural learner or a global numerical optimizer.

The legacy `code/reproduce.py` performs the deterministic grid, plots, seeded adversarial checks and a pdfLaTeX build; it is not a no-RNG smoke test. `research/main_review_checks.py` and `code/exploratory_search_b.py` intentionally use seeded random exploration and should be treated separately from saved-evidence inspection.

Build the supplied paper without computations from `manuscript/` using `pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex`, then `bibtex main`, then the same pdfLaTeX command twice. Generated TeX logs/caches are not source evidence.

All main numerical records are small and included in full. Third-party full papers, environments, caches, old ZIP packages, renderings and internal research dialogues are excluded. The bibliography and cleaned historical literature report retain source references.

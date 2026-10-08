# Reproduce — Phase 7

Run commands from this phase directory. The archived PDFs, complete proofs and saved JSON results can be inspected without running code. Publication preparation did not rerun any scientific calculation or random draw. Only the sanitized Chinese PDF was rebuilt.

## Small deterministic arithmetic checks (optional; not run in this preparation)

The following commands use Python 3 and its standard library only. They perform finite arithmetic/enumeration, not neural training. The scripts and algorithms are unchanged.

```sh
python3 research/finite/check_equal_mass_costs.py
python3 research/finite/check_four_state_transfer.py
```

The first prints exact-rational results for seven stated `j` values and should agree with `research/finite/equal_mass_check_results.json`. It writes only if `--output PATH` is explicitly provided. The second prints the earlier Shannon-family checks using `decimal.Decimal`; compare the saved table in `research/finite/route_report.md`. These computations do not replace the smoothness, asymptotic or universal proofs.

`research/memory/check_memory.py` is retained unchanged for completeness. Its `main()` uses seeded pseudorandom Jacobian witnesses and overwrites the sibling `check_results.json`; it is deliberately excluded from the minimal commands above. The saved output is historical. Do not run that entry point when random generation or overwriting archived evidence is disallowed. Deterministic subsets may be inspected directly in its source; no new driver or algorithm is introduced here.

## Build documents with an existing TeX installation

The English paper uses Python 3, `pdflatex` and `bibtex`, plus standard TeX packages named in `manuscript/main.tex` (`geometry`, `fontenc`, `inputenc`, `lmodern`, `microtype`, AMS packages, `booktabs`, `tabularx`, `array`, `enumitem`, `natbib`, `hyperref`, `xurl`). No third-party Python package is required.

```sh
python3 qa/build_working_paper.py --output-dir build/english --pdf build/phase7_working_paper.pdf
```

Choose an output directory that does not already exist. The script compiles the main manuscript through LaTeX/BibTeX passes with `-no-shell-escape`, checks unresolved references and records a build manifest. Supplying `--pdf` as shown avoids replacing the archival PDF. The script does not access earlier phases. Typesetting timestamps/tool versions can change PDF hashes without changing scientific text.

The sanitized Chinese TeX is already provided, so Pandoc is unnecessary for this direct build. With existing XeLaTeX, run:

```sh
mkdir -p build/chinese
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory build/chinese qa/phase7_decision_zh.tex
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory build/chinese qa/phase7_decision_zh.tex
```

The Chinese document uses `fontspec`, `xeCJK`, `unicode-math`, `xurl`, `graphicx`, `bookmark`, `titlesec`, `fancyhdr` and the other packages declared in its preamble. Its historical fonts are Songti SC, Heiti SC, TeX Gyre Pagella/Heros and Latin Modern Math. Songti SC and Heiti SC are platform-specific; if unavailable, choose installed CJK substitutes in a working copy and expect layout differences. No font files are redistributed.

`qa/build_decision_pdf.py` regenerates Chinese TeX from the sanitized Markdown using existing Pandoc and XeLaTeX. That historical helper writes the TeX, logs and PDF in this phase tree; use it only in a disposable working copy if the archived files must remain unchanged. The direct build above leaves the public sources and archived PDFs intact.

Standalone supporting documents are `research/memory/exact_query_memory.tex` and `research/information/main.tex` (the latter includes `theory.tex` and its local bibliography). The finite and sketch branch files are section fragments; the supported complete build is the integrated main manuscript.

## Evidence limits

The repository contains no Phase 7 training data or trained model because this increment is theoretical. Saved PASS outputs date to the original research and are not current rerun claims. A PDF build checks typesetting and references, not mathematical correctness. Literature links and access limitations are preserved from the dated audit and were not rechecked online during publication preparation.

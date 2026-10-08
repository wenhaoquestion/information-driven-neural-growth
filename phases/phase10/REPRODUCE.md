# Reproduce the paper and inspect exact results

Build with an existing pdfLaTeX installation:

```sh
cd phases/phase10/manuscript
sh build.sh
```

The canonical entry is `phase10_last_two_capacities.tex`; the bundled PDF is the original manuscript PDF. All included `.tex` inputs are local. The manuscript's bibliography cites the earlier phases, which remain separate working papers.

The three saved JSON files in `research/experiments/results/` contain exact rational witnesses, counts and checks. Their source scripts in `research/experiments/code/` write to that results directory. If independently rerunning them, use a disposable copy; retain the published evidence unchanged. `find_strict_quartet_relief.py` reads the saved top-capacity result. These are exact finite calculations, not training or estimates of worst-case asymptotics.

Publication preparation copied existing results and did not rerun the experiments. Public review results and metadata hashes are distinguished from the original scientific record in the repository provenance files.

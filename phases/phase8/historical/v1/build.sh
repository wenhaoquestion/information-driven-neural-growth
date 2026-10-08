#!/bin/sh
set -eu
cd "$(dirname "$0")"
engine="${PDFLATEX:-pdflatex}"
command -v "$engine" >/dev/null 2>&1 || { printf '%s\n' "pdfLaTeX is required. Set PDFLATEX to its executable path." >&2; exit 1; }
for source in curvature_hierarchy.tex fixed_quartet.tex uniform_log_upper.tex; do
  test -f "$source" || { printf 'Missing source: %s\n' "$source" >&2; exit 1; }
done
mkdir -p build
for pass in 1 2 3; do
  "$engine" -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build curvature_hierarchy.tex
done
printf '\nBuilt build/curvature_hierarchy.pdf\n'

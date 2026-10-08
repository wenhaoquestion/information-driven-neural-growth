#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p build
for pass in 1 2 3; do
  pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build curvature_hierarchy.tex
done
printf '\nBuilt build/curvature_hierarchy.pdf\n'

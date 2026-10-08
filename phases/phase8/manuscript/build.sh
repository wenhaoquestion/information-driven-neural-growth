#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p build
for paper_pass in 1 2 3; do
  pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build curvature_hierarchy.tex > "build/pass${paper_pass}.txt"
done
cp build/curvature_hierarchy.pdf curvature_hierarchy.pdf

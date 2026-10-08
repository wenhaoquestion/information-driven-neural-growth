#!/bin/sh
set -eu
cd "$(dirname "$0")"
output_dir=${1:-build}
mkdir -p "$output_dir"
for pass in 1 2 3; do
  pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error \
    -output-directory="$output_dir" phase9_weighted_quartet.tex \
    > "$output_dir/pass_${pass}.log"
done
cp "$output_dir/phase9_weighted_quartet.pdf" phase9_weighted_quartet.pdf

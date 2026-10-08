#!/bin/sh
set -eu
out_dir="${1:-../qa/paper_build}"
mkdir -p "$out_dir"
for pass in 1 2 3; do
  pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error \
    -file-line-error -recorder -output-directory="$out_dir" \
    phase10_last_two_capacities.tex > "$out_dir/pass${pass}.stdout" 2>&1
done
cp "$out_dir/phase10_last_two_capacities.pdf" phase10_last_two_capacities.pdf
pdfinfo phase10_last_two_capacities.pdf > "$out_dir/pdfinfo.txt"
pdftotext -layout phase10_last_two_capacities.pdf "$out_dir/manuscript_text.txt"

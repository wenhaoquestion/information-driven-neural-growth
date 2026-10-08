#!/bin/sh
set -eu
code_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exp_dir=$(dirname "$code_dir")
phase9_python=${PHASE9_PYTHON:-python3}
mkdir -p "$exp_dir/logs"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$phase9_python" "$code_dir/check_weighted_reduction.py" > "$exp_dir/logs/weighted_reduction.stdout.log"
"$phase9_python" "$code_dir/check_coordinate_obstruction.py" > "$exp_dir/logs/coordinate_obstruction.stdout.log"
"$phase9_python" "$code_dir/independent_coordinate_check.py" > "$exp_dir/logs/coordinate_independent_mp.stdout.log"
"$phase9_python" "$code_dir/check_indicator_bridge.py" > "$exp_dir/logs/indicator_bridge.stdout.log"
"$phase9_python" "$code_dir/render_coordinate_figure.py" > "$exp_dir/logs/figure.stdout.log"
printf 'Phase9 finite checks completed successfully.\n'

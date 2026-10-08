#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/logs/matplotlib_cache"
PYTHON=${PHASE8_PYTHON:-python3}
"$PYTHON" code/run_experiments.py correctness > logs/correctness.stdout.log 2>&1
"$PYTHON" code/run_experiments.py weighted > logs/weighted.stdout.log 2>&1
"$PYTHON" code/run_experiments.py polynomial > logs/polynomial.stdout.log 2>&1
"$PYTHON" code/run_experiments.py endpoint > logs/endpoint.stdout.log 2>&1
"$PYTHON" code/check_endpoint_mp.py > logs/endpoint_mp.stdout.log 2>&1
"$PYTHON" code/render_figures.py > logs/figures.stdout.log 2>&1
"$PYTHON" code/summarize.py

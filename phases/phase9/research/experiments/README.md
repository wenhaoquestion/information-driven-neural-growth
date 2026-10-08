# Actual Phase 9 computations

All checks ran in the existing environment. No installation or neural training. Analytic proofs establish the theorems; finite computations check reductions and implementations.

| Script | Actual result | Saved evidence |
|---|---|---|
| `code/check_weighted_reduction.py` | 378 exact rational instances (9×7×6); all 7 two-cell, 6 three-cell and 18 compatible pairs, including noncontiguous cases; 12,993 kernel checks; all passed. Maximum observed price ≈2.815301718. | `results/weighted_reduction_exact.json` |
| `code/check_coordinate_obstruction.py` | m=4,8,16,32,64,128,256,512; η=10⁻³/m⁴. Gauss orders96/192 on endpoint-aware panels; maximum discrepancy2.88658×10⁻¹⁵. At degree2044 product lower bound149433.85, normalized OPT₂≈1.67346×10⁻³⁸. | `results/coordinate_obstruction.json`, `.csv` |
| `code/independent_coordinate_check.py` | Independent expanded polynomial, m=4,16, 100-digit arithmetic; checks ONLY A_left,A_right,B_left,B_right. Maximum discrepancy≈2.24225×10⁻¹⁵. Does not separately check tiny weighted costs, OPT₂ or comparison products. | `results/coordinate_independent_mp.json` |
| `code/check_indicator_bridge.py` | Exact rational27 populations,108 oracle states,81 transitions,27 unrestricted single-unit fits; birth-only1/16→1/16 and gradient overshoot2→8 actually computed. All passed. | `results/indicator_bridge_exact.json` |
| `../prior_art/check_exact_asymmetry.py` | Separate classical appendix:34 extremizers, degrees0–33; higher-order quadrature discrepancy≤5.44×10⁻¹⁴. | `../prior_art/exact_asymmetry_check.json` |

The four core checks and figure were actually run to successful exit via `sh code/reproduce.sh`. Set `PHASE9_PYTHON` to an existing interpreter if necessary; its default is `python3`. Run the separate Jacobi script independently with the same interpreter. Do not confuse the four core steps with the four report evidence categories.

Recorded environment: Python3.12.14, NumPy2.3.5, SciPy1.18.1, mpmath1.3.0, Matplotlib3.11.2. Thread limits1; deterministic, no random seed. Core durations≈0.467,0.143,1.115,0.069 seconds, excluding startup/drawing; these are local observations, not performance benchmarks. JSON records preserve code hashes and exact settings.

The figure separates comparison-method distortion from actual hierarchy price (analytically1) and absolute normalized flat costs. First drawing emitted a default font-cache permission warning while producing a valid figure; The public technical notes retain its technical meaning; raw machine-specific logs are excluded. Cache was redirected into this stage and only the figure regenerated. Scientific results were unchanged.

Floating-point consistency is not interval certification. The tiny weighted costs have two-order consistency only. Explicit finite η choices are numerical examples; the proof uses an existence limit. No sample estimation, SGD/Adam, generalization or trained-model comparison is claimed. Publication preparation did not rerun scientific calculations.

# Saved population computation

`first_pass.json` contains all ten candidates with 15 nonempty-cell costs, 15 partitions and 39 compatible budget pairs each. CSV files provide the same inspection tables. `precision_recheck.json` preserves the single 64/128-point local quadrature comparison. `historical_comparison.json` compares raw generator costs with the selected prior inputs in `../historical_snapshot/`.

The finite population classifications rely on the separate analytic rational certificates. The interior $n=8192$ floating-point Delta lies about $3.9\times10^{-18}$ below the displayed rigorous lower endpoint; the original number is intentionally retained. Agreement between quadrature orders does not guarantee enclosure by a much narrower exact interval. `CERTIFICATE_COMPARISON.md` records this limitation. `failures_and_corrections.json` also records the timing-field name reuse: a per-candidate elapsed field must not be interpreted as total run time.

`compute_pilot.py` preserves the calculation and the guard against overwriting the first pass. Its historical lookup root has been relocated to the bundled snapshot; formulas are unchanged. Normal deterministic publication review reads the saved arrays and does not rerun quadrature. Public hashes are explained in the study provenance document.

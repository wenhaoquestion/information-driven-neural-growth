# Deterministic reproduction from saved counts

From the repository root, use an existing Python 3 environment. The inference and both final reviews use only the standard library. They do not call RNG or regenerate labels.

```sh
python3 -B studies/additive-risk/statistics/infer_confirmation.py \
  --counts studies/additive-risk/statistics/confirmation/COUNTS_PUBLIC.json \
  --output /tmp/additive-inference-public-replay.json
python3 -B reviews/statistical-closure/theory/independent_verify.py
python3 -B reviews/statistical-closure/code/independent_code_check.py
```

Choose a fresh output name for inference. The review commands write `PUBLICATION_THEORY_CHECK.json` and `PUBLICATION_CODE_CHECK.json` next to their scripts, leaving historical check results in `historical_results/` intact. Compare exact rational interval endpoints, all 39 hierarchy intervals, 15 report-error bounds, the decision and power bounds. Runtime and timestamps are not scientific equality targets. The theory review independently reconstructs the arithmetic without importing the original inference modules; the code review separately enumerates partitions and replays their pure functions.

The original core arithmetic implementations and power/oracle certificates remain available. A new replay receipt hashes the current publication source. The publication provenance ledger explains every adapted path, removed machine field and refreshed metadata digest; refreshed digests are not historical execution certification.

`confirm_once.py` is retained to document the original simulator and exclusive lock. **Keep the existing lock and fixed counts; do not rerun it for this study.** The population computation script refuses to overwrite its existing first pass. Its source requires NumPy/SciPy for an independently authorized future numerical replication; publication verification does not execute it or install dependencies. The 64/128-point arrays are diagnostics, not rigorous intervals. Mathematical bound scripts write local outputs; use a disposable copy if reviewing their original command-line entrypoints.

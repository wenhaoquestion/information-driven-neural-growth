# Reproduce the two deterministic reviews

From the repository root:

```sh
python3 -B reviews/statistical-closure/theory/independent_verify.py
python3 -B reviews/statistical-closure/code/independent_code_check.py
```

Use Python 3.10 or newer; only standard-library arithmetic is required. The first script reimplements the mathematical arithmetic from JSON, and the second independently enumerates partitions and invokes the original pure inference/power functions on the fixed public counts. Neither invokes simulator or old numerical `main` functions, imports NumPy, or creates random labels.

The scripts write new public receipts beside themselves. Historical results are retained separately and must not be relabeled as new executions. Input source hashes follow the publication copies, as documented by `../../studies/additive-risk/PUBLICATION_PROVENANCE.md`; this establishes public-copy consistency, not re-attestation of the 2026 experiment's environment.

# Accounting correction (no numerical result changed)

The original `execution.json` field `persistent_words` was poorly named and its selected arrays mix model state and batch accumulator. It is **not a comparable persistent-memory or measured peak-memory result**. It omits input streams, teacher, shared basis, saved paths and linear-algebra workspaces. The simulator stores all methods and the full generated stream concurrently for reproducibility.

Interpret 64 as the raw d×d moment array, 128 as a d×d model plus a d×d residual batch accumulator, and 1332 as the lifted 36×36 Gram plus 36-vector. These are named array examples, not total memory. No storage advantage follows. The trained-neural experiment supplies its separate accounting.

The originally executed source is saved byte-for-byte as `runner_snapshot.py` and matches `execution.json` source_sha256. The main source changes only the metadata field and its explanatory text for future replays. All numerical code, seeds, data and outcomes remain unchanged. Independent review identified this issue before manuscript finalization.

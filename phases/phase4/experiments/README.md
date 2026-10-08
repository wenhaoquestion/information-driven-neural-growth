# Audit-label release experiment

Historical September 2026 experiment: four tasks, twelve paired seeds, seven arms, 24 events. [PROTOCOL.md](PROTOCOL.md) fixes the design; [FINDINGS.md](FINDINGS.md) preserves measured effects, adverse comparisons, resource definitions, and descriptive uncertainty.

All seed-level summaries, event rows, paired contrasts, and method summaries are supplied under `results/`. Raw `interaction_750000.json` and its NPZ data/access counts provide a complete representative trajectory. The other 47 raw run JSON files and their NPZ arrays remain local, with hashes in [the phase raw ledger](../RAW_ARTIFACTS.json). `run_index.json` retains the full original inventory. Full-history reconstruction and plotting scripts require restoring the full raw set; a partial raw directory must not be treated as the full study.

See [REPRODUCE.md](../REPRODUCE.md) for safe output locations, commands, and the standard-library smoke check. Existing audit/replay JSON files describe historical checks. The development protocol records a path-serialization failure and retry; no failed scientific run has been relabelled as confirmatory evidence.

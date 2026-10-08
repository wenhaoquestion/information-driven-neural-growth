# Frozen studies and public evidence

The scientific recipes are `PROTOCOL.md`, `ADAPTIVE_PROTOCOL.md` and `FULLBATCH_PROTOCOL.md`. The first study uses scheduled widths; the second chooses grow/continue/hold using an explicitly heuristic validation gate; the third is a separately frozen response to the minibatch-tail confound. All configurations, seed lists, freeze records and original run indexes are retained.

Generated input/access NPZ and complete per-event JSON are not distributed in Git. Relative paths and original SHA-256 values are in `../EXCLUDED_ARTIFACTS.csv`. Public `../results/published_seed_endpoints.csv` includes all 768 neural endpoints and their source hashes; `published_checkpoint_branches.csv` includes all 1,440 diagnostic branch outcomes. The original result summaries retain every method, task and adverse comparison. Large raw records remain local and are reproducible with the supplied source/configurations; this release is not a hosted full raw-data archive.

Portable reconstruction and document-build commands are in `../REPRODUCE.md`. The publication pass did not execute the experiments. Historical run times are observations of the earlier local execution, not new benchmarks.

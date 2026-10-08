# Inspect and reproduce audit diagnostics

The self-contained Python scripts in `research/equal_mass/` and `experiments/risk_boundary_checks.py` use the standard library and write result files beside themselves. `research/weighted_top/adversarial_exact.py` additionally reads the public manuscript sources solely to record source hashes; its source locations are resolved from the repository root.

Inspect the retained JSON first. To recompute finite diagnostics, use a disposable repository copy so the original results remain intact. These scripts enumerate exact finite configurations; they do not train models or resample the additive confirmation. Publication preparation did not rerun them.

The historical audit's environment-log receipt crawler is omitted because it depended on private machine logs. Relevant scope limitations and the missing Phase 9 stdout evidence are preserved in `ERRATA.md`. Mathematical sources, fixed outputs and publication transformation hashes provide the public reproduction path.

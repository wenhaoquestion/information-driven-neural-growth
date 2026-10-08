# Historical implementation audit summary

This is a publication summary of the historical internal technical check. It removes private workflow/role descriptions and does not reproduce an internal conversation. It is not external peer review or a claim of personal author review. The publication pass did not rerun the scientific calculations. Original review hashes are retained in the private curation provenance; public numerical evidence and complete manuscript proofs remain available.

The main, adaptive and full-minibatch runners distinguish learner inputs from post-trajectory teacher/evaluator access. Frozen configurations, code identities, seed lists and original result indexes are retained. The empirical audits checked prefix access, model/optimizer state, fitted parameters, declared work and reported outcomes; saved numerical results describe the earlier execution.

Resource distinctions remain explicit: retained arrays, model/optimizer state, temporary arrays, scientific archives, example accesses, optimizer updates, scalar-work proxies and physical peak memory are not interchangeable. Moment diagnostic `persistent_words` was corrected to selected-array accounting without altering numerical results. See `../results/moments/ACCOUNTING_CORRECTION.md` and the exact historical `runner_snapshot.py`.

The original replay-capacity/batch-tail confound was diagnosed rather than hidden. The separate fullbatch study is retained with new seeds, original source/protocol hashes and all adverse contrasts. Public packaging alters metadata paths and the plotting cache location only; the frozen learning algorithms and configurations are unchanged.

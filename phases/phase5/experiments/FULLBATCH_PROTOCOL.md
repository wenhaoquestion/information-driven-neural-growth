# Targeted full-minibatch follow-up

This experiment is a post-hoc response to an independently identified implementation confound, not a replacement of the original study or a retroactive confirmation. The original frozen primary runner and all raw outcomes are retained.

In the original code, the fitting loop truncates each minibatch at the end of the current pool permutation. R128 makes the post-warmup pool exactly 1,152 rows (nine batches of128); R138 makes it1,162 and adds a ten-row optimizer step per epoch. With equal fitting-example accesses, random R128 and R138 execute975 and1034 Adam updates. Reservoir size therefore changes optimizer dynamics beyond access to stored examples.

Clone the frozen primary runner into `phase5/code/quadratic_fullbatch.py`. Change only the fitting batch assembly: consume rows across an epoch boundary, drawing the next independent permutation as needed, until128 rows form a batch. Only the final budget-limited batch can have fewer than128 rows. Preserve the model, populations, initializations, learning rate, clipping, optimizer state, candidate mechanisms, proxy charges, seven arrivals and all memory/data timing. Disable the expensive same-checkpoint diagnostic branches in this targeted study; their original results remain available.

Run four arms: historical moment R128, random R128, random R138, and fixed7 with R138. Use12 new seeds850000–850011 on each of the same four tasks, for192 trajectories and1344 events. Freeze source, config and this protocol before any new-seed output. No tuning is permitted on their evaluator results.

The targeted comparisons are random R128 minus R138 and historical moment minus each random arm. Report every task and the fixed-capacity comparator; use descriptive95% paired Student-t intervals across independent seeds with no multiplicity correction. This remains a finite-budget experiment, not a theorem that residual statistics, memory or larger models are universally beneficial. Verify identical fit-example counts AND identical optimizer-step counts for the two random arms after the fix.

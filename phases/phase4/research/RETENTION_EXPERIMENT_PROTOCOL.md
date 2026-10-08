# Frozen retention-geometry experiment protocol

Recorded before execution of the new simulations, 2026-09-15 UTC.
This experiment is separate from the trained-network diagnostic. It is an explicitly finite, known-input-distribution inference study; it does not train a neural proposal and does not establish practical neural superiority.

## R1: residual geometry and cost of finite historical evidence
Input X is uniform on the four sign pairs (f1,f2) in {-1,+1}². Let h=0.2(sqrt(1-z) f1 + sqrt(z) f2), p=(1-h)/2, q=(1+h)/2, and V=span(f1). Squared movement is 0.04; residual movement is 0.04 z. z is fixed to 1,1/4,1/16,1/64. Under the null eta=1/2; under the alternative eta=1/2 + 0.003 f2/(0.4 sqrt(z)), so every alternative has the same true gain 0.003. All probabilities are bounded away from zero and one.

1000 independent replicates per environment/z, seeds 941000 + 10*condition_index + z_index. Fresh audit sizes 128,512,2048,8192,32768,131072 are fixed in advance; counts represent nested iid draws. Report pointwise rejection rates (no claim of a simultaneous familywise test across displayed sizes). Compare: direct paired Bernstein bound, oracle retained population moment with residual Bernstein bound, and 32768 independent historical labels retaining the f1 moment with a Hoeffding interval plus residual Bernstein bound. All methods receive the same fresh counts. The last method charges its extra historical labels, and divides 0.05 failure budget equally between historical and fresh uncertainty; oracle exact moments are explicitly uncharged ideal information, not an equally resourced learner. No threshold/size/seed selection after outcomes.

## R2: adaptive-candidate historical reuse stress test
Under the global null eta=1/2 on m in {4,16,64} equiprobable states, draw n=4096 historical labels. Choose h(x)=0.4 sign(historical centered label moment for x), with p=(1-h)/2 and q=(1+h)/2. The comparison is chosen adaptively from those labels and has true gain zero. Across 1000 replicates (seeds 942000+m), compare naive fixed-pair Bernstein reuse, a simultaneous coordinate Hoeffding moment region, and a fresh independent n=4096 audit conditional on the chosen pair. Report all results including failures; R2 is designed to reveal invalid adaptive reuse, not estimate typical training behavior. The uniform moment method covers all bounded prediction differences on this fixed finite support; no per-candidate union bound is needed.

## Records and analysis
Save sufficient multinomial counts, per-replicate estimates/bounds/rejections, environment, hashes, and analytic parameter checks. Wilson intervals summarize independent replicate uncertainty. Generate figures directly from raw outputs. Counts are raw sampled input-label outcomes aggregated by their finite sufficient categories; seeds reconstruct the draws. Conclusions concern fixed-size inference. Repeated inspection across the plotted sizes is not authorized as an anytime-valid gate by these bounds.

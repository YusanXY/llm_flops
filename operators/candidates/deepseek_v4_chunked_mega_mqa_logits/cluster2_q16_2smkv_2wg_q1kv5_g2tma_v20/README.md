# Cluster2 Q16 2SM-KV / group2-TMA Q1-KV5 V20

V20 changes V17's KV load/readiness path to explicit SM100 `cta_group::2` TMA.
Both CTAs load disjoint M128 halves with rank-dependent coordinates while the
hardware merges completion bytes into rank 0's transaction barrier.  Rank 0
then broadcasts readiness to rank 1's epilogue consumers; the old rank-1 local
TMA wait followed by a rank-1-to-rank-0 software arrival is removed.  A two-CTA cluster
owns sixteen adjacent causal queries.  Each CTA TMA-loads one disjoint M128 KV
half and real `tcgen05.mma.cta_group::2` instructions gather them into a
logical M256N256K32 operation.

Four N256 passes reuse the gathered KV tile for Q16.  Two consumer warpgroups
alternate Q pairs on two fixed TMEM stages, so TCGen05 issue for one stage can
overlap ReLU/reduction/store work on the other.  This doubles query reuse per
KV load without changing the 168-register math budget or 384-thread block.
Per-head ReLU, FP32 accumulation/reduction, scale placement, masking and output
ordering remain unchanged.  The purpose is to amortize Q16/weight TMA and
task-boundary barriers without changing the TCGen05 or epilogue schedule.

Promotion requires all nine formal cases at the unchanged `rtol=1e-5`,
`atol=1e-6`, clean same-run benchmark improvement, explicit multicast and
2SM TCGen05 instructions in PTX/SASS, and a complete `ncu --set full` report.  A
compile, launch, synchronization or numerical failure rejects the experiment.

# Cluster2 Q32 2SM-KV / two-warpgroup Q1-KV5 chunk64 V21

V21 doubles the validated V17 query-reuse window.  A two-CTA cluster owns
thirty-two adjacent causal queries.  Each CTA TMA-loads one disjoint M128 KV
half and real `tcgen05.mma.cta_group::2` instructions gather them into a
logical M256N256K32 operation.

Eight N256 passes reuse the gathered KV tile for Q32.  Two consumer warpgroups
alternate Q pairs on two fixed TMEM stages, so TCGen05 issue for one stage can
overlap ReLU/reduction/store work on the other.  This doubles query reuse per
KV load without changing the 168-register math budget or 384-thread block.
Per-head ReLU, FP32 accumulation/reduction, scale placement, masking and output
ordering remain unchanged.  The purpose is to halve KV stage traffic per query
and amortize task-boundary barriers while preserving the TCGen05 and epilogue
arithmetic.

Promotion requires all nine formal cases at the unchanged `rtol=1e-5`,
`atol=1e-6`, clean same-run benchmark improvement, explicit multicast and
2SM TCGen05 instructions in PTX/SASS, and a complete `ncu --set full` report.  A
compile, launch, synchronization or numerical failure rejects the experiment.

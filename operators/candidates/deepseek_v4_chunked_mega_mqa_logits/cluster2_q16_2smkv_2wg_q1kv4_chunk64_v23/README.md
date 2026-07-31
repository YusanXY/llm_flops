# Cluster2 Q16 2SM-KV / two-warpgroup Q1-KV4 chunk64 V23

V23 changes only V17's KV ring depth from five stages to four.  A two-CTA cluster
owns sixteen adjacent causal queries.  Each CTA TMA-loads one disjoint M128 KV
half and real `tcgen05.mma.cta_group::2` instructions gather them into a
logical M256N256K32 operation.

Four N256 passes reuse the gathered KV tile for Q16.  Two consumer warpgroups
alternate Q pairs on two fixed TMEM stages, so TCGen05 issue for one stage can
overlap ReLU/reduction/store work on the other.  This doubles query reuse per
KV load without changing the 168-register math budget or 384-thread block.
Per-head ReLU, FP32 accumulation/reduction, scale placement, masking and output
ordering remain unchanged.  The purpose is to amortize Q16/weight TMA and
task-boundary barriers without changing the TCGen05 or epilogue schedule.  The
experiment removes 16,896 bytes of dynamic SMEM per CTA and tests whether V17's
fifth KV stage is necessary at Q16.

Promotion requires all nine formal cases at the unchanged `rtol=1e-5`,
`atol=1e-6`, clean same-run benchmark improvement, explicit multicast and
2SM TCGen05 instructions in PTX/SASS, and a complete `ncu --set full` report.  A
compile, launch, synchronization or numerical failure rejects the experiment.

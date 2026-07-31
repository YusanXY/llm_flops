# Cluster2 Q8 multicast-KV / Q1-KV5 V10

V10 is a structural experiment derived from the validated V9/V4 path.  A
portable two-CTA cluster owns eight adjacent causal queries.  Each CTA keeps
the proven BLOCK_Q4, M128N256K32 TCGen05/TMEM, two-math-warp-group pipeline,
while CTA rank selects one disjoint Q4 half.

Both CTAs walk the same KV chunks.  They arm independent local transaction
barriers, then rank 0 issues explicit
`cp.async.bulk.tensor.*.shared::cluster.global...multicast::cluster` loads with
mask `0x3`.  This is intended to retain V4's math parallelism while approaching
V6's nearly halved KV L2-to-SM traffic.

Promotion requires all nine formal cases at the unchanged `rtol=1e-5`,
`atol=1e-6`, clean same-run benchmark improvement, explicit multicast and
TCGen05 instructions in PTX/SASS, and a complete `ncu --set full` report.  A
compile, launch, synchronization or numerical failure rejects the experiment.

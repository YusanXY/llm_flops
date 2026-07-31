# Cluster2 Q8 2SM-KV / two-warpgroup Q1-KV5 V14

V14 is derived from the strict-correctness V13 2SM path.  A two-CTA cluster
owns eight adjacent causal queries.  Each CTA TMA-loads one disjoint M128 KV
half and real `tcgen05.mma.cta_group::2` instructions gather them into a
logical M256N256K32 operation.

Two N256 passes reuse the gathered KV tile for Q8.  Unlike V13's single
consumer warpgroup, V14 assigns one math warpgroup to each Q pair and keeps it
on one TMEM stage across KV iterations.  The split epilogue targets 168 math
registers and restores a 384-thread block while preserving V13's reduced TMA
traffic and exact per-head ReLU/FP32 reduction semantics.

Promotion requires all nine formal cases at the unchanged `rtol=1e-5`,
`atol=1e-6`, clean same-run benchmark improvement, explicit multicast and
2SM TCGen05 instructions in PTX/SASS, and a complete `ncu --set full` report.  A
compile, launch, synchronization or numerical failure rejects the experiment.

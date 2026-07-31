# BLOCK_Q8 / N256x2 / Q1-KV8 V6

This candidate preserves the canonical single-request ABI: Q is
`[1,m,64,128]`, causal lengths are `[1,m]`, and the physical page table has
one row.  Eight adjacent causal positions are grouped only inside the device
scheduler; all eight retain their own length, weight row, mask and output row.

The logical TC tile is `M128N512K32`.  It is emitted as two explicit legal
`tcgen05.mma` `M128N256K32` instructions that share the same KV descriptor and
write the low and high 256-column halves of one 512-column TMEM accumulator.
`SPLIT_KV=128` allows one math warpgroup and one TMEM stage.  One Q plus eight
KV shared-memory stages remain below the dynamic-SMEM limit.

Because the public DeepGEMM metadata helper partitions work in stock split
units, V6 builds a cached `(q_token_start, split_start)` boundary table in
128-token units before timed steady state.  No benchmark, tolerance, reference,
FP8 decode, FP32 accumulation, reduction order, scale, mask, or physical-cache
semantics are changed.

The experiment is promoted only if it passes every formal case and improves
clean same-run timing over V4.  Full NCU must also show the intended TMA traffic
reduction with no local or shared spills.

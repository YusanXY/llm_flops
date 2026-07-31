# BLOCK_Q4 / LDS.V4 / Q1-KV5 V8

V8 is derived from the validated V4 winner and changes only FP32 weight loads
in the per-head epilogue.  The single-request ABI, BLOCK_Q4 scheduler,
M128N256K32 TCGen05/TMEM path, Q1/KV5 TMA stages, TMEM32x2 loads, FP8 decode,
FP32 arithmetic order, masks, scales and output addresses are unchanged.

V4 broadcasts weights from shared memory and avoids all spills, but its
unrolled reduction expresses four scalar shared loads for each group of four
heads.  V8 uses explicit `ld.shared.v4.b32` PTX to fetch those same four
contiguous FP32 values in one vector instruction, then performs the identical
two float2 FMA chains.  It is promoted only after full correctness, clean
same-run timing and spill-free NCU evidence.

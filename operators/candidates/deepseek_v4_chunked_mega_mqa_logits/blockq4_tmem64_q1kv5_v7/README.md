# BLOCK_Q4 / TMEM64 / Q1-KV5 V7

V7 is a one-change experiment derived from the validated V4 winner.  It keeps
the canonical one-request ABI, BLOCK_Q4 scheduling, M128N256K32 TCGen05 MMA,
Q1/KV5 TMA pipeline, two TMEM stages, spill-free shared-weight reduction,
FP8 decode, FP32 accumulation, masks, scales and output order unchanged.

Full NCU on V4 reports TMEM as the highest-utilized pipeline (61.2%) and
54.64% tensor-pipe activity.  For H=64, V4 loads each result with two
32-register TMEM instructions and fences after both halves.  V7 explicitly
uses the native `SM100_TMEM_LOAD_32dp32b64x` instruction and a single
completion fence.  It is promoted only if all formal correctness cases pass,
clean timing improves, and NCU confirms no spill regression.

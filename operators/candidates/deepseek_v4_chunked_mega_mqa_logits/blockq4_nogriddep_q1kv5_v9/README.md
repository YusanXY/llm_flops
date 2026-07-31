# BLOCK_Q4 / no GRIDDEPSYNC / Q1-KV5 V9

V9 is a one-change experiment derived from the validated V4 winner.  It keeps
the canonical single-request ABI, BLOCK_Q4 scheduler, M128N256K32 TCGen05/TMEM
path, Q1/KV5 TMA stages, spill-free reduction, FP8 decode, FP32 arithmetic,
masks, scales and output order unchanged.

The schedule tensor is generated before the candidate launch and both
operations execute on the same CUDA stream.  The binding uses a normal kernel
launch and does not set a programmatic-dependent-launch attribute.  Therefore
the device-side `cudaGridDependencySynchronize()` has no dependency to resolve;
V9 removes it and relies on standard stream ordering.  It is promoted only if
all formal cases pass and clean timings improve over V4.

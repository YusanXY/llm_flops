# BLOCK_Q4 / explicit shared-weight broadcast V3

This is one candidate for `deepseek_v4_chunked_mega_mqa_logits`.  Its public
ABI is the canonical single-request layout: Q `[1,m,64,128]`, causal lengths
`[1,m]`, and one physical page-table row `[1,256]`.

Internally it groups four adjacent causal query positions for device
scheduling; this does not create four-request semantics.  The positions share
the one physical page table.  The kernel changes the SM100 MQA tile from
`M128N128K32` to
`M128N256K32`.  The kernel explicitly instantiates the TCGen05/TMEM path;
the reduced two-stage TMEM ring fits two 256-column accumulator stages in the
architectural 512-column budget.

NCU on V2 showed that keeping four FP32 weight rows live caused ptxas to spill
them: 18.95M local-memory spill requests and 4.96M shared-spill requests.  V3
removes the register-resident `weights[4][64]` tile.  During the head reduction
all lanes explicitly read the same weight address from shared memory, which is
a hardware broadcast and preserves the FP32 arithmetic exactly.  The TMEM,
TCGen05, TMA, mask, scale, output order, benchmark scripts, tolerances and
reference implementation are unchanged.

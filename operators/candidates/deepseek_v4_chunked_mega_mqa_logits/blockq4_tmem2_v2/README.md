# BLOCK_Q4 / TMEM2 V2

This is one candidate for `deepseek_v4_chunked_mega_mqa_logits`.  Its public
ABI is the canonical single-request layout: Q `[1,m,64,128]`, causal lengths
`[1,m]`, and one physical page-table row `[1,256]`.

Internally it groups four adjacent causal query positions for device
scheduling; this does not create four-request semantics.  The positions share
the one physical page table.  The kernel changes the SM100 MQA tile from
`M128N128K32` to
`M128N256K32`.  The kernel explicitly instantiates the TCGen05/TMEM path;
the reduced two-stage TMEM ring fits two 256-column accumulator stages in the
architectural 512-column budget.  Benchmark scripts, tolerances and the
reference implementation are unchanged.

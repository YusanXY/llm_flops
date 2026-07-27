# DeepSeek V4 Flash TileLang sparse attention

The pinned SGLang source exposes `dpsk_v4_fp8_attention_fwd`, including its
explicit gfx942 launch configuration. The current TileLang build rejects that
kernel during software-pipeline injection, so this reference defaults to
SGLang's supported `triton_fp8_attention_fwd` fallback. Set
`SGLANG_HACK_FLASHMLA_BACKEND=tilelang` to revalidate TileLang after an
upgrade; only the identified compiler error may then trigger the fallback.
Backend selection happens before steady-state samples. The cases cover
SWA-only (C0), SWA+C4, and SWA+C128 forms in both prefill and decode.

The contract records packed MODEL1 FP8 KV caches, logical page tables,
translated physical indices, valid lengths, compression ratio, and attention
sink. Cache/page/index state is observed and must remain unchanged. Small cases
use an independent PyTorch gather-softmax-value oracle.

Performance uses HIP events with graph capture disabled. The indexed
cache-attention path is not graph-state invariant on this pinned build; direct
launch timing avoids measuring an invalid or alternating captured state.

Source evidence:

- `python/sglang/srt/layers/attention/hip_flash_mla.py`
- `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py`
- `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`

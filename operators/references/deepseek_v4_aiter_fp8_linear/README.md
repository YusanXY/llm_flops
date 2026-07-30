# DeepSeek V4 Flash fused WQ_A + WKV projection

This compatibility reference now measures only the fused WQ_A + WKV decode
projection on the current SGLang/AITER gfx942 dynamic block-FP8 path.

- Logical projection: `fused_wq_a_wkv`
- Shape: `[M, 4096] @ [1536, 4096]^T -> [M, 1536]`
- Decode `M`: `16`, `32`
- Raw context metadata: `65536`

C4 indexer query projection and WO_B projection are intentionally excluded.
They are registered and optimized independently as:

- `deepseek_v4_aiter_c4_indexer_q_projection`
- `deepseek_v4_aiter_wo_b_projection`

Source evidence:

- SGLang: `python/sglang/srt/layers/quantization/fp8_utils.py`,
  `aiter_w8a8_block_fp8_linear`
- AITER: `aiter/ops/quant.py`, `per_group_quant_hip`
- AITER: `aiter/ops/triton/gemm_a8w8_blockscale.py`

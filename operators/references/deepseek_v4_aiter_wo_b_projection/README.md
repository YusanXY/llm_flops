# DeepSeek V4 Flash WO_B projection

This reference measures only the WO_B attention-output projection on the
current SGLang/AITER gfx942 dynamic block-FP8 path.

- Logical projection: `wo_b_projection`
- Shape: `[M, 8192] @ [4096, 8192]^T -> [M, 4096]`
- Prefill `M`: `1024`, `2048`, `4096`
- Decode `M`: `16`, `32`
- Raw context metadata: `65536`

It deliberately does not contain C4 indexer query-projection or fused
WQ_A + WKV cases. Those projections have independent operator contracts and
optimization directories.

Source evidence:

- SGLang: `python/sglang/srt/layers/quantization/fp8_utils.py`,
  `aiter_w8a8_block_fp8_linear`
- AITER: `aiter/ops/quant.py`, `per_group_quant_hip`
- AITER: `aiter/ops/triton/gemm_a8w8_blockscale.py`

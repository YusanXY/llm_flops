# DeepSeek V4 Flash SGLang/AITER block-FP8 linear

This reference measures the current SGLang gfx942 linear path, including
dynamic per-1x128 activation quantization and the AITER Triton
`gemm_a8w8_blockscale` kernel. It covers the Flash model's fused WQ_A + WKV
projection (`K=4096, N=1536`), C4 indexer query projection
(`K=1024, N=8192`), and attention output WO_B projection
(`K=8192, N=4096`).

Source evidence:

- SGLang: `python/sglang/srt/layers/quantization/fp8_utils.py`,
  `aiter_w8a8_block_fp8_linear`
- AITER: `aiter/ops/quant.py`, `per_group_quant_hip`
- AITER: `aiter/ops/triton/gemm_a8w8_blockscale.py`

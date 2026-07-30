# DeepSeek V4 Flash Q RMSNorm + WQ_B

This reference isolates the production Q normalization and wide query
projection path.

On gfx942, the current SGLang model executes AITER RMSNorm first, then its
dynamic per-1x128 activation quantization and AITER Triton block-FP8 linear.
The gfx950-only fused RMSNorm/FP8-quant path is intentionally not used.

Flash dimensions are `K=1024` (Q-LoRA rank) and `N=32768`
(`64 heads * 512 head_dim`).

Source evidence:

- SGLang: `python/sglang/srt/models/deepseek_v4.py`
- SGLang: `python/sglang/srt/layers/layernorm.py`
- SGLang: `python/sglang/srt/layers/quantization/fp8_utils.py`
- AITER: `aiter/ops/rmsnorm.py`

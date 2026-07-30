# DeepSeek V4 Flash WO_A grouped BF16 projection

This reference isolates the production grouped attention-output projection.

The current gfx942 default has `SGLANG_OPT_FP8_WO_A_GEMM` disabled. SGLang
therefore reshapes the attention output to `[T, 8, 4096]` and executes
`torch.einsum("tgd,grd->tgr", o, wo_a)` with BF16 weights
`[8, 1024, 4096]`.

Source evidence:

- SGLang: `python/sglang/srt/models/deepseek_v4.py`, attention output path

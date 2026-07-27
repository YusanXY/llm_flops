# DeepSeek V4 Flash SGLang AITER FP8 fused MoE

This reference measures the current SGLang AMD routed-expert path with dynamic
block-FP8 activations, E4M3FNUZ weights, float32 128x128 inverse scales, and
the DeepSeek V4 SwiGLU clamp of 10. The representative model shape is
`E=256, H=4096, I=2048, topK=6`.

The representative cases call `aiter.fused_moe.fused_moe` directly with the
same `Silu`, `per_128x128`, interleaved gate/up, and SwiGLU-limit arguments
that SGLang's `AiterRunnerCore` supplies on gfx942. The measured call therefore
does not construct SGLang `MoeRunner`, config, dispatch-output, or quant-info
objects. It does not silently fall back to Triton. An independent SGLang
Triton block-FP8 result is sampled as the correctness oracle before the
weights are preshuffled for AITER.

The spec manually prepares the BF16 activations, preshuffled FP8 expert
weights, 128x128 float32 scales, float32 route weights, and int32 expert IDs
before timing. The AITER fused operator returns the `[tokens, hidden]` BF16
output directly; the wrapper deliberately adds no output allocation or copy
beyond the workspace/output behavior of the production AITER API itself.

Routing is outside the fused-MoE operator, so the synthetic expert IDs use a
deterministic balanced dispatch. This keeps random expert-load skew out of the
kernel timing while preserving the model's sqrt-softplus scores, normalized
top-6 weights, and `routed_scaling_factor=1.5`. No model weights are
downloaded.

This operator uses HIP event timing because MoE sorting and workspace
allocation are not graph-capture safe. The benchmark engine retains every
asynchronous output in an inner-iteration group until the end event
synchronizes, matching the lifetime required by AITER's internal workspaces.

Small cases use an explicit PyTorch routed-expert oracle. Representative
prefill/decode cases preserve the full model weight shape and use the
independent SGLang Triton kernel as their oracle.

Source evidence:

- SGLang: `python/sglang/srt/layers/moe/moe_runner/aiter.py`
- SGLang: `python/sglang/srt/layers/quantization/fp8.py`
- AITER: `aiter/fused_moe.py`
- AITER: `op_tests/test_moe_blockscale.py`

"""DeepSeek V4 Flash MI300X high-priority prefill/decode projection.

This projection intentionally covers the seven operator categories requested
for comparison. It is not presented as full end-to-end model latency.
"""

from benchmark_engine.models import CaseSpec

from .base import ProjectionMapping


MODEL_LAYERS = 43
C4_LAYERS = 21
C128_LAYERS = 20
PREFILL_INPUTS = (1024, 2048, 4096)
DECODE_INPUTS = (16,)
RAW_CONTEXT = 65536
FP8_PROFILE = "fp8_block_gfx942"


def _m(
    adapter_id,
    name,
    backend,
    instances,
    operator_id,
    kind,
    shape=None,
    phase="prefill",
):
    return ProjectionMapping(
        adapter_id,
        phase,
        name,
        backend,
        instances,
        operator_id,
        kind,
        shape,
    )


_PREFILL = (
    _m(
        "sparse_prefill_attention_c4",
        "Sparse Prefill Attention C4",
        "SGLang DSV4 Triton sparse_fwd",
        C4_LAYERS,
        "deepseek_v4_tilelang_sparse_attention",
        "sparse_attention",
        (64, 512, 512),
    ),
    _m(
        "sparse_prefill_attention_c128",
        "Sparse Prefill Attention C128",
        "SGLang DSV4 Triton sparse_fwd",
        C128_LAYERS,
        "deepseek_v4_tilelang_sparse_attention",
        "sparse_attention",
        (64, 512, 512),
    ),
    _m(
        "routed_expert_fused_moe",
        "Routed Expert Fused MoE",
        "direct AITER block-FP8 fused_moe",
        MODEL_LAYERS,
        "deepseek_v4_aiter_fp8_fused_moe",
        "moe_fp8",
        (256, 4096, 2048, 6),
    ),
    _m(
        "c4_fp8_paged_mqa_logits",
        "C4 FP8 Paged MQA Logits",
        "AITER fp8_paged_mqa_logits",
        C4_LAYERS,
        "deepseek_v4_aiter_c4_paged_mqa_logits",
        "fp8_logits",
        (64, 128, RAW_CONTEXT // 4),
    ),
    _m(
        "q_rmsnorm_wq_b",
        "Q RMSNorm + WQ_B",
        "AITER RMSNorm + dynamic 1x128 FP8 + AITER Triton GEMM",
        MODEL_LAYERS,
        "deepseek_v4_q_rmsnorm_wqb",
        "rmsnorm_fp8_linear",
        (1024, 32768),
    ),
    _m(
        "wo_a_grouped_projection",
        "WO_A Grouped Projection",
        "PyTorch/rocBLAS grouped BF16 einsum",
        MODEL_LAYERS,
        "deepseek_v4_wo_a_grouped_bf16",
        "grouped_bf16",
        (8, 4096, 1024),
    ),
    _m(
        "wo_b_projection",
        "WO_B Projection",
        "SGLang dynamic 1x128 FP8 + AITER Triton GEMM",
        MODEL_LAYERS,
        "deepseek_v4_aiter_fp8_linear",
        "fp8_linear",
        (8192, 4096),
    ),
    _m(
        "c4_indexer_q_projection",
        "C4 Indexer Q Projection",
        "SGLang dynamic 1x128 FP8 + AITER Triton GEMM",
        C4_LAYERS,
        "deepseek_v4_aiter_fp8_linear",
        "fp8_linear",
        (1024, 8192),
    ),
)


def _d(adapter_id, name, backend, instances, operator_id, kind, shape=None):
    return _m(
        adapter_id,
        name,
        backend,
        instances,
        operator_id,
        kind,
        shape,
        phase="decode",
    )


_DECODE = (
    _d(
        "routed_expert_fused_moe",
        "Routed Expert Fused MoE",
        "direct AITER block-FP8 fused_moe",
        MODEL_LAYERS,
        "deepseek_v4_aiter_fp8_fused_moe",
        "moe_fp8",
        (256, 4096, 2048, 6),
    ),
    _d(
        "q_rmsnorm_wq_b",
        "Q RMSNorm + WQ_B",
        "AITER RMSNorm + dynamic 1x128 FP8 + AITER Triton GEMM",
        MODEL_LAYERS,
        "deepseek_v4_q_rmsnorm_wqb",
        "rmsnorm_fp8_linear",
        (1024, 32768),
    ),
    _d(
        "wo_a_grouped_projection",
        "WO_A Grouped Projection",
        "PyTorch/rocBLAS grouped BF16 einsum",
        MODEL_LAYERS,
        "deepseek_v4_wo_a_grouped_bf16",
        "grouped_bf16",
        (8, 4096, 1024),
    ),
    _d(
        "wo_b_projection",
        "WO_B Projection",
        "SGLang dynamic 1x128 FP8 + AITER Triton GEMM",
        MODEL_LAYERS,
        "deepseek_v4_aiter_fp8_linear",
        "fp8_linear",
        (8192, 4096),
    ),
    _d(
        "c4_indexer_q_projection",
        "C4 Indexer Q Projection",
        "SGLang dynamic 1x128 FP8 + AITER Triton GEMM",
        C4_LAYERS,
        "deepseek_v4_aiter_fp8_linear",
        "fp8_linear",
        (1024, 8192),
    ),
    _d(
        "fused_wq_a_wkv",
        "Fused WQ_A + WKV",
        "SGLang dynamic 1x128 FP8 + AITER Triton GEMM",
        MODEL_LAYERS,
        "deepseek_v4_aiter_fp8_linear",
        "fp8_linear",
        (4096, 1536),
    ),
)


class DeepSeekV4FlashMi300xProjection:
    projection_id = "deepseek_v4_flash_mi300x_high_priority"
    display_name = "DeepSeek V4 Flash MI300X high-priority operator subset"

    def mappings(self, phase: str, quant_profile: str):
        if quant_profile == FP8_PROFILE:
            if phase == "prefill":
                return _PREFILL
            if phase == "decode":
                return _DECODE
        return ()

    def mapping_for_case(self, operator_id: str, case: CaseSpec):
        symbols = case.symbols
        adapter_id, phase, profile = (
            symbols.get(name)
            for name in ("projection_adapter_id", "phase", "quant_profile")
        )
        if not all(
            isinstance(value, str)
            for value in (adapter_id, phase, profile)
        ):
            return None
        return next(
            (
                mapping
                for mapping in self.mappings(phase, profile)
                if mapping.adapter_id == adapter_id
                and mapping.operator_id == operator_id
            ),
            None,
        )


DEEPSEEK_V4_FLASH_MI300X_PROJECTION = DeepSeekV4FlashMi300xProjection()


__all__ = [
    "C4_LAYERS",
    "C128_LAYERS",
    "DECODE_INPUTS",
    "DEEPSEEK_V4_FLASH_MI300X_PROJECTION",
    "DeepSeekV4FlashMi300xProjection",
    "FP8_PROFILE",
    "MODEL_LAYERS",
    "PREFILL_INPUTS",
    "RAW_CONTEXT",
]

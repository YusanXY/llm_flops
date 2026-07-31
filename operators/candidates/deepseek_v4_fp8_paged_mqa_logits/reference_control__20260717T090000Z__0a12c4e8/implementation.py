"""DeepGEMM FP8 paged-MQA baseline extracted from the legacy benchmark."""

import deep_gemm
from sglang.jit_kernel.dsa import deepgemm_paged_mqa_logits_split


def operator(
    q_fp8,
    kv_fused,
    weights,
    context_lens,
    page_table,
    schedule,
    max_context,
    q_offset,
):
    if q_fp8.ndim == 4:
        return deep_gemm.fp8_paged_mqa_logits(
            q_fp8,
            kv_fused,
            weights,
            context_lens,
            page_table,
            schedule,
            max_context,
            False,
        )
    result = deepgemm_paged_mqa_logits_split(
        deep_gemm.fp8_paged_mqa_logits,
        q_fp8,
        kv_fused,
        weights,
        context_lens,
        page_table,
        schedule,
        max_context,
        q_offset=q_offset,
    )
    return result[0] if isinstance(result, tuple) else result

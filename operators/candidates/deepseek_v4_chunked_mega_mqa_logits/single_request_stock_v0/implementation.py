"""Unmodified DeepGEMM control for the canonical one-request CaseSpec."""

from __future__ import annotations

import deep_gemm


HEADS = 64
HEAD_DIM = 128
PAGE_SIZE = 64
COMPRESSED_CONTEXT = 16384


def operator(
    history_hidden,
    wkv_gate,
    ape,
    norm_weight,
    rope_cos,
    rope_sin,
    q_fp8,
    weights,
    kv_fused,
    c4_seq_lens,
    page_table,
    schedule,
    raw_context,
    recompute_eligible_raw_start,
    rms_eps,
):
    del history_hidden, wkv_gate, ape, norm_weight, rope_cos, rope_sin, rms_eps

    if q_fp8.ndim != 4 or q_fp8.shape[0] != 1:
        raise ValueError("control requires exactly one request")
    m = q_fp8.shape[1]
    if q_fp8.shape != (1, m, HEADS, HEAD_DIM):
        raise ValueError("Q must be [1,m,64,128]")
    if weights.shape != (m, HEADS):
        raise ValueError("weights must be [m,64]")
    if c4_seq_lens.shape != (1, m):
        raise ValueError("causal lengths must be [1,m]")
    if page_table.ndim != 2 or page_table.shape[0] != 1:
        raise ValueError("page table must be [1,pages]")
    if raw_context != 65536 or recompute_eligible_raw_start != 0:
        raise ValueError("unsupported formal boundary")

    return deep_gemm.fp8_paged_mqa_logits(
        q_fp8,
        kv_fused,
        weights,
        c4_seq_lens,
        page_table,
        schedule,
        COMPRESSED_CONTEXT,
        False,
    )

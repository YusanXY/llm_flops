"""DeepSeek-V4 full-KV-reuse control for causal paged-MQA logits.

The formal input bundle intentionally carries two mathematically equivalent
representations of Indexer K:

* ``kv_fused`` is the complete physical 8448-byte/page C4 Indexer cache.
* ``history_hidden`` plus the frozen Indexer producer parameters can recreate
  every byte of that cache.

This reference is the *full reuse* control.  Its timed path reads the supplied
cache and calls paged MQA; it does not project or rewrite historical K.  An
optimized candidate may instead recompute an arbitrary historical suffix from
the equivalent activation/weight inputs and fuse that suffix into the MQA
consumer, provided that it preserves the exact DeepSeek-V4 C4 semantics and
the physical cache contents.
"""

from __future__ import annotations

import torch


HEAD_DIM = 128
HEADS = 64
PAGE_SIZE = 64
COMPRESSION_RATIO = 4
WKV_GATE_DIM = 4 * HEAD_DIM
FP8_MAX = 448.0


def indexer_kv_projection_prefill(
    hidden_states: torch.Tensor, wkv_gate: torch.Tensor
) -> torch.Tensor:
    """BF16 x BF16^T -> FP32, matching SGLang ``linear_bf16_fp32``."""

    return torch.mm(hidden_states, wkv_gate.t(), out_dtype=torch.float32)


def c4_full_history_compress(
    projected_records: torch.Tensor,
    ape: torch.Tensor,
) -> torch.Tensor:
    """Recreate all C4 Indexer K rows from a sequence starting at token zero.

    Output group ``g`` represents raw sequence length ``4*(g+1)``:

    * regular fields come from raw records ``4g .. 4g+3``;
    * for ``g>0``, overlap fields come from ``4g-4 .. 4g-1``;
    * for ``g==0``, SGLang's missing overlap values are zero and their scores
      are negative infinity, so only the four regular records contribute.

    The explicit maximum and reduction order mirrors ``c4_v2.cuh``.
    """

    if projected_records.ndim != 2 or projected_records.shape[1] != WKV_GATE_DIM:
        raise ValueError("projected records must be [raw_context, 512]")
    if projected_records.shape[0] % COMPRESSION_RATIO:
        raise ValueError("raw history must be C4 aligned")
    if ape.shape != (8, HEAD_DIM):
        raise ValueError("APE must be [8, 128]")

    groups = projected_records.reshape(-1, COMPRESSION_RATIO, WKV_GATE_DIM)
    older = torch.empty_like(groups)
    older[0].zero_()
    older[1:].copy_(groups[:-1])

    values = torch.cat(
        (
            older[..., :HEAD_DIM],
            groups[..., HEAD_DIM : 2 * HEAD_DIM],
        ),
        dim=1,
    )
    scores = torch.cat(
        (
            older[..., 2 * HEAD_DIM : 3 * HEAD_DIM],
            groups[..., 3 * HEAD_DIM : 4 * HEAD_DIM],
        ),
        dim=1,
    )
    scores[0, :COMPRESSION_RATIO].fill_(-torch.inf)
    logits = scores + ape.unsqueeze(0)

    maximum = logits[:, 0]
    for index in range(1, 8):
        maximum = torch.maximum(maximum, logits[:, index])
    sum_exp = torch.zeros_like(maximum)
    sum_product = torch.zeros_like(maximum)
    for index in range(8):
        factor = torch.exp(logits[:, index] - maximum)
        sum_exp = sum_exp + factor
        sum_product = sum_product + values[:, index] * factor
    return sum_product / sum_exp


def c4_overlap_compress_prefill(
    c4_prefix_carry: torch.Tensor,
    projected_records: torch.Tensor,
    ape: torch.Tensor,
) -> torch.Tensor:
    """Recompute an aligned suffix when the preceding four records are known."""

    if c4_prefix_carry.shape != (4, WKV_GATE_DIM):
        raise ValueError("C4 prefix carry must be [4, 512]")
    if projected_records.ndim != 2 or projected_records.shape[1] != WKV_GATE_DIM:
        raise ValueError("projected records must be [m, 512]")
    if projected_records.shape[0] % COMPRESSION_RATIO:
        raise ValueError("prefill extension must be C4 aligned")

    all_records = torch.cat((c4_prefix_carry, projected_records), dim=0)
    windows = all_records.unfold(0, 8, COMPRESSION_RATIO).permute(0, 2, 1)
    older = windows[:, :4]
    newer = windows[:, 4:]
    values = torch.cat(
        (
            older[..., :HEAD_DIM],
            newer[..., HEAD_DIM : 2 * HEAD_DIM],
        ),
        dim=1,
    )
    scores = torch.cat(
        (
            older[..., 2 * HEAD_DIM : 3 * HEAD_DIM],
            newer[..., 3 * HEAD_DIM : 4 * HEAD_DIM],
        ),
        dim=1,
    )
    logits = scores + ape.unsqueeze(0)
    maximum = logits[:, 0]
    for index in range(1, 8):
        maximum = torch.maximum(maximum, logits[:, index])
    sum_exp = torch.zeros_like(maximum)
    sum_product = torch.zeros_like(maximum)
    for index in range(8):
        factor = torch.exp(logits[:, index] - maximum)
        sum_exp = sum_exp + factor
        sum_product = sum_product + values[:, index] * factor
    return sum_product / sum_exp


def rms_norm_prefill(
    values: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    variance = values.square().mean(dim=-1, keepdim=True)
    return values * torch.rsqrt(variance + eps) * weight


def apply_compressed_rope_prefill(
    values: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply SGLang DSV4 complex RoPE to the trailing 64 dimensions."""

    output = torch.empty_like(values)
    output[:, :64].copy_(values[:, :64])
    rope = values[:, 64:].reshape(values.shape[0], 32, 2)
    real = rope[..., 0]
    imag = rope[..., 1]
    output_rope = output[:, 64:].reshape(values.shape[0], 32, 2)
    output_rope[..., 0].copy_(real * rope_cos - imag * rope_sin)
    output_rope[..., 1].copy_(real * rope_sin + imag * rope_cos)
    return output


def normalized_fwht_128_prefill(values: torch.Tensor) -> torch.Tensor:
    """Normalized Sylvester FWHT matching the DSV4 Indexer rotation."""

    transformed = values
    width = 1
    while width < HEAD_DIM:
        groups = transformed.reshape(-1, HEAD_DIM // (2 * width), 2, width)
        left = groups[:, :, 0]
        right = groups[:, :, 1]
        transformed = torch.cat((left + right, left - right), dim=-1).reshape(
            values.shape[0], HEAD_DIM
        )
        width *= 2
    return transformed * (HEAD_DIM**-0.5)


def quantize_indexer_k_prefill(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One E4M3 group and one FP32 scale per 128-wide Indexer K."""

    scale = torch.clamp(values.abs().amax(dim=-1, keepdim=True), min=1e-4)
    scale = scale / FP8_MAX
    quantized = torch.clamp(values / scale, -FP8_MAX, FP8_MAX).to(
        torch.float8_e4m3fn
    )
    return quantized, scale


def store_indexer_k_prefill(
    kv_fused: torch.Tensor,
    cache_write_locs: torch.Tensor,
    quantized_k: torch.Tensor,
    k_scale: torch.Tensor,
) -> None:
    """Scatter exact K+scale records into SGLang's 8448-byte page ABI."""

    page_bytes = PAGE_SIZE * (HEAD_DIM + 4)
    cache_bytes = kv_fused.view(kv_fused.shape[0], page_bytes)
    pages = torch.div(cache_write_locs, PAGE_SIZE, rounding_mode="floor").long()
    slots = torch.remainder(cache_write_locs, PAGE_SIZE).long()

    value_view = cache_bytes[:, : PAGE_SIZE * HEAD_DIM].view(torch.float8_e4m3fn)
    columns = slots.unsqueeze(1) * HEAD_DIM + torch.arange(
        HEAD_DIM, device=slots.device, dtype=torch.long
    ).unsqueeze(0)
    value_view[pages.unsqueeze(1), columns] = quantized_k

    scale_view = cache_bytes[:, PAGE_SIZE * HEAD_DIM :].view(torch.float32)
    scale_view[pages, slots] = k_scale.squeeze(-1)


def rebuild_full_indexer_cache(
    history_hidden: torch.Tensor,
    wkv_gate: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    rms_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact FP8 K codes and FP32 scales for the full history.

    This helper is deliberately not called by the baseline ``operator``.  It
    documents and enables the candidate-side alternative representation.
    """

    projected = indexer_kv_projection_prefill(history_hidden, wkv_gate)
    compressed = c4_full_history_compress(projected, ape)
    normalized = rms_norm_prefill(compressed, norm_weight, rms_eps)
    roped = apply_compressed_rope_prefill(normalized, rope_cos, rope_sin)
    rotated = normalized_fwht_128_prefill(roped)
    return quantize_indexer_k_prefill(rotated)


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
    """Execute the full-physical-KV-reuse DeepSeek-V4 MQA control."""

    import deep_gemm

    if q_fp8.ndim != 4 or q_fp8.shape[0] != 1:
        raise ValueError("Q must describe exactly one request")
    m = q_fp8.shape[1]
    if history_hidden.shape != (raw_context, 7168):
        raise ValueError("history_hidden must contain the complete raw context")
    if wkv_gate.shape != (WKV_GATE_DIM, 7168):
        raise ValueError("wkv_gate must be [512, 7168]")
    if q_fp8.shape != (1, m, HEADS, HEAD_DIM):
        raise ValueError("Q must be [1, m, 64, 128]")
    if weights.shape != (m, HEADS):
        raise ValueError("weights must be [m, 64]")
    if c4_seq_lens.shape != (1, m):
        raise ValueError("causal lengths must be [1, m]")
    if page_table.ndim != 2 or page_table.shape[0] != 1:
        raise ValueError("page table must have exactly one request row")
    if recompute_eligible_raw_start != 0:
        raise ValueError("formal alternative representation covers all history")
    if raw_context % COMPRESSION_RATIO:
        raise ValueError("raw context must be C4 aligned")
    if kv_fused.shape[0] * PAGE_SIZE != raw_context // COMPRESSION_RATIO:
        raise ValueError("physical cache must cover the complete C4 context")

    # Full-reuse control: producer-side tensors are intentionally untouched.
    return deep_gemm.fp8_paged_mqa_logits(
        q_fp8,
        kv_fused,
        weights,
        c4_seq_lens,
        page_table,
        schedule,
        raw_context // COMPRESSION_RATIO,
        False,
    )

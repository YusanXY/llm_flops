"""Sequential baseline for the closed C4 Indexer-K-to-MQA-logits pipeline.

This module deliberately imports no SGLang operator.  It spells out every
stage separately so later fused candidates have one self-contained semantic
reference and one physical-cache ABI.
"""

from __future__ import annotations

import torch


HEAD_DIM = 128
HEADS = 64
PAGE_SIZE = 64
FP8_MAX = 448.0


def indexer_kv_projection(
    hidden_states: torch.Tensor, wkv_gate: torch.Tensor
) -> torch.Tensor:
    """BF16 x BF16^T -> FP32, matching Indexer Compressor.wkv_gate."""

    return torch.mm(hidden_states, wkv_gate.t(), out_dtype=torch.float32)


def build_c4_windows(
    c4_state: torch.Tensor, kv_score: torch.Tensor
) -> torch.Tensor:
    """Build aligned 8-token C4 windows for one contiguous request.

    ``c4_state`` contains the eight raw records immediately preceding the
    extend.  Since all formal prefix lengths are four-token aligned, the first
    newly completed C4 window needs the final four prefix records followed by
    the first four extend records.  Subsequent windows advance by four raw
    tokens.  The persistent state is updated to the final eight records.
    """

    if c4_state.shape != (8, 4 * HEAD_DIM):
        raise ValueError(f"expected C4 state [8,512], got {tuple(c4_state.shape)}")
    if kv_score.shape[0] % 4 != 0:
        raise ValueError("single-request C4 extend length must be divisible by 4")

    records = torch.cat((c4_state[-4:], kv_score), dim=0)
    windows = records.unfold(0, 8, 4).permute(0, 2, 1).contiguous()
    c4_state.copy_(records[-8:])
    return windows


def c4_overlap_compress(windows: torch.Tensor, ape: torch.Tensor) -> torch.Tensor:
    """Eight-token C4 overlap compression with per-dimension softmax.

    State layout per raw token is:
      [overlap_K(128), regular_K(128), overlap_score(128), regular_score(128)].
    At an aligned full window, the first four records use overlap fields and
    the last four records use regular fields.
    """

    overlap = windows[:, :4]
    regular = windows[:, 4:]
    values = torch.cat(
        (overlap[..., :HEAD_DIM], regular[..., HEAD_DIM : 2 * HEAD_DIM]),
        dim=1,
    )
    scores = torch.cat(
        (
            overlap[..., 2 * HEAD_DIM : 3 * HEAD_DIM],
            regular[..., 3 * HEAD_DIM : 4 * HEAD_DIM],
        ),
        dim=1,
    )
    probabilities = torch.softmax(scores + ape.unsqueeze(0), dim=1)
    return torch.sum(values * probabilities, dim=1)


def rms_norm(
    values: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    variance = values.square().mean(dim=-1, keepdim=True)
    return values * torch.rsqrt(variance + eps) * weight


def apply_rope(
    values: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor
) -> torch.Tensor:
    """Apply complex RoPE to the trailing 64 dimensions."""

    output = torch.empty_like(values)
    output[:, :64].copy_(values[:, :64])
    rope = values[:, 64:].reshape(values.shape[0], 32, 2)
    real = rope[..., 0]
    imag = rope[..., 1]
    output_rope = output[:, 64:].reshape(values.shape[0], 32, 2)
    output_rope[..., 0].copy_(real * rope_cos - imag * rope_sin)
    output_rope[..., 1].copy_(real * rope_sin + imag * rope_cos)
    return output


def normalized_fwht_128(values: torch.Tensor) -> torch.Tensor:
    """Normalized Sylvester FWHT over the Indexer head dimension."""

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


def quantize_indexer_k(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token E4M3 K quantization with one FP32 scale."""

    scale = torch.clamp(values.abs().amax(dim=-1, keepdim=True), min=1e-4) / FP8_MAX
    quantized = torch.clamp(values / scale, -FP8_MAX, FP8_MAX).to(
        torch.float8_e4m3fn
    )
    return quantized, scale


def store_indexer_k(
    kv_fused: torch.Tensor,
    cache_write_locs: torch.Tensor,
    quantized_k: torch.Tensor,
    k_scale: torch.Tensor,
) -> None:
    """Scatter K and scale into SGLang's 8448-byte physical page layout.

    A page is not token-interleaved: all 64x128 FP8 K bytes come first,
    followed by 64 FP32 scales.
    """

    page_bytes = PAGE_SIZE * (HEAD_DIM + 4)
    cache_bytes = kv_fused.view(kv_fused.shape[0], page_bytes)
    pages = torch.div(cache_write_locs, PAGE_SIZE, rounding_mode="floor").long()
    slots = torch.remainder(cache_write_locs, PAGE_SIZE).long()

    value_bytes = cache_bytes[:, : PAGE_SIZE * HEAD_DIM]
    value_view = value_bytes.view(torch.float8_e4m3fn)
    columns = slots.unsqueeze(1) * HEAD_DIM + torch.arange(
        HEAD_DIM, device=slots.device, dtype=torch.long
    ).unsqueeze(0)
    value_view[pages.unsqueeze(1), columns] = quantized_k

    scale_bytes = cache_bytes[:, PAGE_SIZE * HEAD_DIM :]
    scale_view = scale_bytes.view(torch.float32)
    scale_view[pages, slots] = k_scale.squeeze(-1)


def operator(
    hidden_states,
    wkv_gate,
    c4_state,
    ape,
    norm_weight,
    rope_cos,
    rope_sin,
    q_fp8,
    weights,
    kv_fused,
    context_lens,
    page_table,
    schedule,
    cache_write_locs,
    raw_context,
    compressed_context,
    rms_eps,
):
    """Execute every baseline stage sequentially and return FP32 logits."""

    import deep_gemm

    m = hidden_states.shape[0]
    if (
        raw_context % 8 != 0
        or raw_context // 4 != compressed_context
        or m % 4 != 0
    ):
        raise ValueError("baseline requires an aligned C4 context")
    if q_fp8.shape != (1, m, HEADS, HEAD_DIM):
        raise ValueError("single-request Q must be [1,m,64,128]")
    if context_lens.shape != (1, m) or page_table.shape[0] != 1:
        raise ValueError("single-request metadata must be [1,m] with one page row")
    if cache_write_locs.numel() != m // 4:
        raise ValueError("C4 producer must write exactly m/4 compressed K records")

    kv_score = indexer_kv_projection(hidden_states, wkv_gate)
    windows = build_c4_windows(c4_state, kv_score)
    compressed_k = c4_overlap_compress(windows, ape)
    normalized_k = rms_norm(compressed_k, norm_weight, rms_eps)
    rope_k = apply_rope(normalized_k, rope_cos, rope_sin)
    rotated_k = normalized_fwht_128(rope_k)
    quantized_k, k_scale = quantize_indexer_k(rotated_k)
    store_indexer_k(kv_fused, cache_write_locs, quantized_k, k_scale)

    return deep_gemm.fp8_paged_mqa_logits(
        q_fp8,
        kv_fused,
        weights,
        context_lens,
        page_table,
        schedule,
        compressed_context,
        False,
    )

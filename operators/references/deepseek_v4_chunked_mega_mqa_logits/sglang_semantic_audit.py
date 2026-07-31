"""Cross-check full-history and suffix-recompute helpers against SGLang DSV4."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

from sglang.jit_kernel.dsv4 import (
    CompressorPrefillPlan,
    compress_forward,
    compress_norm_rope_store,
    linear_bf16_fp32,
)


ROOT = Path("/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops")
REFERENCE = ROOT / "operators/references/deepseek_v4_chunked_mega_mqa_logits"


def load_impl():
    spec = importlib.util.spec_from_file_location(
        "full_reuse_impl", REFERENCE / "implementation.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load standalone reference")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_segment(impl, generator, raw_prefix: int, m: int):
    device = torch.device("cuda:0")
    raw_context = raw_prefix + m
    hidden_size = 7168
    hidden = torch.randn(
        (m, hidden_size),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.05)
    wkv = torch.randn(
        (512, hidden_size),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    carry = torch.randn(
        (4, 512), device=device, dtype=torch.float32, generator=generator
    ).mul_(0.1)
    ape = torch.randn(
        (8, 128), device=device, dtype=torch.float32, generator=generator
    ).mul_(0.05)
    norm_weight = torch.randn(
        (128,), device=device, dtype=torch.float32, generator=generator
    ).mul_(0.05).add_(1.0)

    projected_sglang = linear_bf16_fp32(hidden, wkv)
    projected_reference = impl.indexer_kv_projection_prefill(hidden, wkv)
    projection_max_abs = float(
        (projected_sglang - projected_reference).abs().max().item()
    )

    plan = CompressorPrefillPlan.generate_legacy(
        compress_ratio=4,
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=device),
        seq_lens=torch.tensor([raw_context], dtype=torch.int64),
        extend_lens=torch.tensor([m], dtype=torch.int64),
        num_q_tokens=m,
        device=device,
    )
    torch.cuda.synchronize()
    plan_c_words = plan.plan_c.view(torch.int32).cpu()
    plan_w_words = plan.plan_w.view(torch.int32).cpu()
    plan_c_words = plan_c_words[plan_c_words[:, 0] != -1]
    read_pages = plan_c_words[:, 2:4]
    write_locs = plan_w_words[:, 1]
    max_page = max(
        0,
        int(read_pages.clamp(min=0).max().item()),
        int(
            torch.div(
                write_locs.clamp(min=0), 4, rounding_mode="floor"
            ).max().item()
        ),
    )
    state = torch.zeros(
        (max_page + 1, 4, 512), device=device, dtype=torch.float32
    )
    if raw_prefix:
        first_read_page = int(plan_c_words[0, 2].item())
        state[first_read_page].copy_(carry)

    compressed_sglang = compress_forward(
        state,
        projected_sglang,
        ape,
        plan,
        head_dim=128,
        compress_ratio=4,
    )
    if raw_prefix:
        compressed_reference = impl.c4_overlap_compress_prefill(
            carry.clone(), projected_reference, ape
        )
        mode = "suffix_with_four_record_carry"
    else:
        compressed_reference = impl.c4_full_history_compress(
            projected_reference, ape
        )
        mode = "sequence_start_missing_overlap"
    c4_max_abs = float(
        (compressed_sglang - compressed_reference).abs().max().item()
    )

    frequency = 1.0 / (
        40000.0
        ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64.0)
    )
    phase = torch.outer(torch.arange(raw_context, dtype=torch.float32), frequency)
    freq_cis = torch.polar(torch.ones_like(phase), phase).to(device)
    emitted_positions = torch.arange(
        raw_prefix, raw_context, 4, device=device, dtype=torch.long
    )
    rope_rows = freq_cis[emitted_positions]
    rope_cos = rope_rows.real
    rope_sin = rope_rows.imag

    page_size = 64
    page_bytes = page_size * (128 + 4)
    cache_sglang = torch.empty(
        (1, page_bytes), device=device, dtype=torch.uint8
    )
    cache_reference = torch.empty_like(cache_sglang)
    for cache in (cache_sglang, cache_reference):
        cache[:, : page_size * 128].view(torch.float8_e4m3fn).fill_(1.0)
        cache[:, page_size * 128 :].view(torch.float32).fill_(1.0)

    out_loc = torch.zeros(m, device=device, dtype=torch.int64)
    ragged_ids = torch.bitwise_and(plan_c_words[:, 1], 0xFFFF).long().to(device)
    logical_locs = torch.arange(
        raw_prefix // 4, raw_context // 4, device=device, dtype=torch.int64
    )
    out_loc[ragged_ids] = logical_locs
    compress_norm_rope_store(
        compressed_sglang.clone(),
        plan,
        norm_weight=norm_weight,
        norm_eps=1e-6,
        freq_cis=freq_cis,
        out_loc=out_loc,
        kvcache=cache_sglang,
        page_size=page_size,
    )

    normalized = impl.rms_norm_prefill(
        compressed_reference.clone(), norm_weight, 1e-6
    )
    roped = impl.apply_compressed_rope_prefill(normalized, rope_cos, rope_sin)
    rotated = impl.normalized_fwht_128_prefill(roped)
    quantized, scale = impl.quantize_indexer_k_prefill(rotated)
    impl.store_indexer_k_prefill(
        cache_reference.view(1, page_size, 1, 132),
        logical_locs,
        quantized,
        scale,
    )

    k_sglang = cache_sglang[:, : page_size * 128]
    k_reference = cache_reference[:, : page_size * 128]
    scale_sglang = cache_sglang[:, page_size * 128 :].view(torch.float32)
    scale_reference = cache_reference[:, page_size * 128 :].view(torch.float32)
    k_codes_equal = bool(torch.equal(k_sglang, k_reference))
    scale_max_abs = float(
        (scale_sglang - scale_reference).abs().max().item()
    )
    report = {
        "mode": mode,
        "raw_prefix": raw_prefix,
        "m": m,
        "projection_max_abs": projection_max_abs,
        "c4_compress_max_abs": c4_max_abs,
        "fp8_cache_codes_equal": k_codes_equal,
        "fp32_cache_scale_max_abs": scale_max_abs,
        "num_compressed_rows": int(compressed_sglang.shape[0]),
        "plan_seq_lens": plan_c_words[:, 0].tolist(),
        "plan_ragged_ids": ragged_ids.cpu().tolist(),
    }
    if projection_max_abs != 0.0:
        raise AssertionError(report)
    if c4_max_abs > 1e-6:
        raise AssertionError(report)
    if not k_codes_equal or scale_max_abs > 1e-7:
        raise AssertionError(report)
    return report


def main() -> None:
    impl = load_impl()
    torch.cuda.set_device(0)
    generator = torch.Generator(device="cuda:0").manual_seed(90210)
    report = {
        "locked_sglang_paths": {
            "projection": "sglang/jit_kernel/dsv4.linear_bf16_fp32",
            "compressor": "sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh",
            "postprocess_store": (
                "sglang/jit_kernel/csrc/deepseek_v4/"
                "fused_norm_rope_v2.cuh"
            ),
        },
        "audits": [
            audit_segment(impl, generator, raw_prefix=0, m=16),
            audit_segment(impl, generator, raw_prefix=64, m=16),
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

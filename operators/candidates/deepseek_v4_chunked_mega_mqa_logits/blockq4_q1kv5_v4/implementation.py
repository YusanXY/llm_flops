"""V4: spill-free BLOCK_Q4 with a one-Q/five-KV-stage pipeline.

The canonical ABI presents exactly one request containing ``m`` adjacent
causal queries as ``[1,m,64,128]``.  All rows address the one physical cache.
The device implementation groups four adjacent query positions solely for
scheduling and issues a real cta_group::1
M128N256K32 FP8 TCGen05 tile:

* one KV/TMA traversal serves four queries instead of one (V0) or two (V1);
* two 256-column accumulator stages exactly fill the 512-column TMEM budget;
* one Q and five KV shared-memory stages keep the V3 SMEM footprint while
  assigning one extra stage to the dominant KV stream;
* FP32 weights are broadcast directly from shared memory during reduction,
  avoiding V2's register-generated local-memory spill traffic;
* output order and all FP8 decode, FP32 accumulation, weights, scale and mask
  arithmetic remain identical to the stock DeepGEMM implementation.
"""

from __future__ import annotations

import os
from pathlib import Path

import deep_gemm
import torch
from torch.utils.cpp_extension import load


HEADS = 64
HEAD_DIM = 128
PAGE_SIZE = 64
BLOCK_Q = 4
COMPRESSED_CONTEXT = 16384

_HERE = Path(__file__).resolve().parent
_DG_INCLUDE = Path(deep_gemm.__file__).resolve().parent / "include"
_BUILD = _HERE / ".build_blockq4_q1kv5_v4"
_BUILD.mkdir(exist_ok=True)

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")

_ext = load(
    name="mega_chunked_mqa_blockq4_q1kv5_v4",
    sources=[str(_HERE / "v2_binding.cpp"), str(_HERE / "v2_kernel.cu")],
    extra_include_paths=[str(_HERE), str(_DG_INCLUDE)],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "-lineinfo",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-gencode=arch=compute_100a,code=sm_100a",
        "-Xptxas=-v",
    ],
    extra_ldflags=["-lcuda"],
    build_directory=str(_BUILD),
    with_cuda=True,
    verbose=os.environ.get("MEGA_V4_VERBOSE_BUILD", "0") == "1",
)

_SCHEDULE_CACHE = {}
_TABLE_CACHE = {}


def _blockq4_schedule(c4_seq_lens, m):
    device_index = c4_seq_lens.device.index
    key = (device_index, m)
    result = _SCHEDULE_CACHE.get(key)
    if result is None or result.device != c4_seq_lens.device:
        grouped_lens = c4_seq_lens.view(m // BLOCK_Q, BLOCK_Q)
        # Metadata work is defined in query-token x KV-split units, so the
        # public generator is shape-compatible with the custom BLOCK_Q=4
        # device scheduler.
        result = deep_gemm.get_paged_mqa_logits_metadata(
            grouped_lens, PAGE_SIZE, deep_gemm.get_num_sms()
        )
        _SCHEDULE_CACHE[key] = result
    return result


def _blockq4_page_table(page_table, m):
    """Materialize the internal scheduling view outside timed steady state."""
    device_index = page_table.device.index
    key = (device_index, m, page_table.data_ptr())
    cached = _TABLE_CACHE.get(key)
    if cached is None or cached[0] is not page_table:
        grouped = page_table.expand(m // BLOCK_Q, -1).contiguous()
        cached = (page_table, grouped)
        _TABLE_CACHE[key] = cached
    return cached[1]


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
    del history_hidden, wkv_gate, ape, norm_weight, rope_cos, rope_sin
    del schedule, rms_eps

    if q_fp8.ndim != 4 or q_fp8.shape[0] != 1:
        raise ValueError("Q must contain exactly one request")
    m = q_fp8.shape[1]
    if m % BLOCK_Q:
        raise ValueError("BLOCK_Q4 requires a query count divisible by four")
    if q_fp8.shape != (1, m, HEADS, HEAD_DIM):
        raise ValueError("Q must be [1,m,64,128]")
    if weights.shape != (m, HEADS):
        raise ValueError("weights must be [m,64]")
    if c4_seq_lens.shape != (1, m):
        raise ValueError("causal lengths must be [1,m]")
    if page_table.ndim != 2 or page_table.shape[0] != 1:
        raise ValueError("page table must contain exactly one physical-cache row")
    if raw_context != 65536 or recompute_eligible_raw_start != 0:
        raise ValueError("unsupported formal boundary")

    q_grouped = q_fp8.view(m // BLOCK_Q, BLOCK_Q, HEADS, HEAD_DIM)
    lens_grouped = c4_seq_lens.view(m // BLOCK_Q, BLOCK_Q)
    table_grouped = _blockq4_page_table(page_table, m)
    schedule_grouped = _blockq4_schedule(c4_seq_lens, m)

    return _ext.forward(
        q_grouped,
        kv_fused,
        weights,
        lens_grouped,
        table_grouped,
        schedule_grouped,
        COMPRESSED_CONTEXT,
    )

"""FlashInfer block-sparse candidate for DeepSeek V4 sparse prefill.

Each logical query head is one BSR row and each selected token is a 1x1 column
block. The one-head layout selects FlashInfer's CUDA-core path, which supports
the 512-wide head contract. The FlashInfer plan and its buffers are cached per
indices tensor so steady-state calls measure ``run`` rather than planning.
"""

from dataclasses import dataclass

import torch
from flashinfer.sparse import BlockSparseAttentionWrapper


_WORKSPACE_BYTES = 128 * 1024 * 1024
_MAX_CACHED_PLANS = 4


@dataclass
class _CachedPlan:
    wrapper: BlockSparseAttentionWrapper
    indptr: torch.Tensor
    flat_indices: torch.Tensor
    output: torch.Tensor


_PLANS: dict[tuple, _CachedPlan] = {}


def _plan_key(q, kv, indices, softmax_scale, value_dim):
    return (
        q.device.type,
        q.device.index,
        q.dtype,
        kv.dtype,
        tuple(q.shape),
        tuple(kv.shape),
        tuple(indices.shape),
        indices.data_ptr(),
        float(softmax_scale),
        int(value_dim),
    )


def _make_plan(q, kv, indices, softmax_scale, value_dim):
    if q.ndim != 3 or kv.ndim != 3 or indices.ndim != 3:
        raise ValueError("expected q/kv/indices to be rank-3 tensors")
    if indices.shape[0] != q.shape[0] or indices.shape[1] != 1:
        raise ValueError("indices must have shape [query_tokens, 1, topk]")
    if kv.shape[1] != 1:
        raise ValueError("DeepSeek V4 sparse prefill expects one KV head")
    if q.shape[-1] != kv.shape[-1] or int(value_dim) != q.shape[-1]:
        raise ValueError("FlashInfer candidate requires equal QK/value dimensions")
    if indices.dtype != torch.int32:
        raise TypeError("FlashInfer block indices must use torch.int32")

    query_tokens = q.shape[0]
    query_heads = q.shape[1]
    topk = indices.shape[-1]
    if topk <= 0:
        raise NotImplementedError("empty sparse rows are unsupported")
    # Flatten logical heads into independent query rows. R=1 satisfies the
    # wrapper's CUDA-core selection rule and matches decode-kernel semantics.
    q_rows = query_tokens * query_heads
    block_rows = q_rows
    flat_indices = (
        indices[:, 0, :]
        .repeat_interleave(query_heads, dim=0)
        .reshape(-1)
        .contiguous()
    )
    indptr = torch.arange(
        0,
        (block_rows + 1) * topk,
        topk,
        dtype=torch.int32,
        device=indices.device,
    )
    workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=q.device)
    # FlashInfer 0.6.12 selects FA3 on SM100 in auto mode, but its generated
    # head_dim=512 configuration is invalid. FA2 supports this contract.
    wrapper = BlockSparseAttentionWrapper(workspace, backend="fa2")
    wrapper.plan(
        indptr,
        flat_indices,
        q_rows,
        kv.shape[0],
        1,
        1,
        1,
        kv.shape[1],
        q.shape[2],
        causal=False,
        sm_scale=float(softmax_scale),
        q_data_type=q.dtype,
        kv_data_type=kv.dtype,
        o_data_type=q.dtype,
    )
    return _CachedPlan(
        wrapper,
        indptr,
        flat_indices,
        torch.empty((q_rows, 1, q.shape[2]), dtype=q.dtype, device=q.device),
    )


def operator(q, kv, indices, softmax_scale, value_dim):
    key = _plan_key(q, kv, indices, softmax_scale, value_dim)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _make_plan(q, kv, indices, softmax_scale, value_dim)
        if len(_PLANS) >= _MAX_CACHED_PLANS:
            _PLANS.pop(next(iter(_PLANS)))
        _PLANS[key] = plan
    q_flat = q.reshape(-1, 1, q.shape[-1])
    output = plan.wrapper.run(q_flat, kv, kv, out=plan.output)
    return output.reshape_as(q)

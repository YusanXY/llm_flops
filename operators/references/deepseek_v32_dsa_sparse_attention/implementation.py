"""Official MLSys 2026 FlashInfer baseline for DeepSeek-V3.2 DSA attention.

Source: ``flashinfer_wrapper_5af199`` from the official
``flashinfer-ai/mlsys26-contest`` dataset.  The wrapper intentionally uses the
TensorRT-LLM MLA decode entry point: this is the exact API chosen by the
contest baseline, despite the benchmark definition being named DSA attention.
"""

import torch
import flashinfer.decode


_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
_workspace_cache = {}

QK_NOPE_HEAD_DIM = 128
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
TOPK = 2048


def _get_workspace(device):
    key = str(device)
    buf = _workspace_cache.get(key)
    if buf is None:
        buf = torch.zeros(
            _WORKSPACE_SIZE_BYTES, dtype=torch.uint8, device=device
        )
        _workspace_cache[key] = buf
    return buf


def operator(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    device = q_nope.device

    if isinstance(sm_scale, torch.Tensor):
        bmm1_scale = float(sm_scale.item())
    else:
        bmm1_scale = float(sm_scale)

    query = torch.cat([q_nope, q_pe], dim=-1).unsqueeze(1)
    kv_cache = torch.cat([ckv_cache, kpe_cache], dim=-1)
    block_tables = sparse_indices.unsqueeze(1)

    # Invalid entries are contiguous at the end of each row, matching the
    # official contest workload contract.
    seq_lens = (sparse_indices != -1).sum(dim=1).to(torch.int32)
    max_seq_len = int(seq_lens.max().item())
    workspace = _get_workspace(device)

    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        sparse_mla_top_k=TOPK,
        bmm1_scale=bmm1_scale,
    )
    output = output.squeeze(1)

    # Preserve the official destination_passing_style=false return contract.
    return (output,)

"""Contract for the gfx942 DeepSeek V4 TileLang sparse-attention kernel."""

import importlib

from benchmark_engine.correctness import InputBundle
from benchmark_engine.models import CaseSpec
from benchmark_engine.workloads.deepseek_v4_flash import (
    NumericPath,
    OracleStateComparator,
    normalize_named_tensors,
    sample_tensor,
)


HEADS = 64
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
SWA_TOKENS = 128
TOKEN_BYTES = 584
VALUE_BYTES = 576
PREFILL_M = (1024, 2048, 4096)
PROFILE = "fp8_block_gfx942"
UNIFIED_TAG = "deepseek_v4_flash_prefill_unified"


def _padded_cache(torch, blocks, block_size, device):
    logical_bytes = block_size * TOKEN_BYTES
    row_bytes = ((logical_bytes + 575) // 576) * 576
    raw = torch.zeros((blocks, row_bytes), dtype=torch.uint8, device=device)
    return raw[:, :logical_bytes].view(blocks, block_size, 1, TOKEN_BYTES)


def _clone_padded_cache(torch, cache):
    clone = torch.empty_strided(
        cache.shape,
        cache.stride(),
        dtype=cache.dtype,
        device=cache.device,
    )
    clone.copy_(cache)
    return clone


def _pack_cache(torch, logical, block_size):
    token_count = logical.shape[0]
    blocks = (token_count + block_size - 1) // block_size
    padded_tokens = blocks * block_size
    if padded_tokens != token_count:
        logical = torch.cat(
            (
                logical,
                torch.zeros(
                    (padded_tokens - token_count, HEAD_DIM),
                    dtype=logical.dtype,
                    device=logical.device,
                ),
            ),
            dim=0,
        )
    logical = logical.view(blocks, block_size, HEAD_DIM)
    cache = _padded_cache(torch, blocks, block_size, logical.device)
    raw = cache.as_strided(
        (blocks, cache.stride(0)),
        (cache.stride(0), 1),
    )
    values_region = raw[:, : block_size * VALUE_BYTES].view(
        blocks, block_size, VALUE_BYTES
    )
    scales_region = raw[
        :, block_size * VALUE_BYTES : block_size * TOKEN_BYTES
    ].view(blocks, block_size, 8)

    nope = logical[..., :NOPE_DIM].float().view(
        blocks, block_size, NOPE_DIM // 64, 64
    )
    maximum = nope.abs().amax(dim=-1).clamp_min(1.0e-8)
    exponent = torch.ceil(torch.log2(maximum / 240.0))
    scale = torch.exp2(exponent)
    # Both the TileLang loader and the current Triton fallback bitcast these
    # bytes as tl.float8e4nv (the OCP E4M3 encoding), even on gfx942.
    quantized = (nope / scale.unsqueeze(-1)).clamp(-240.0, 240.0).to(
        torch.float8_e4m3fn
    )
    values_region[..., :NOPE_DIM] = quantized.reshape(
        blocks, block_size, NOPE_DIM
    ).view(torch.uint8)
    values_region[..., NOPE_DIM:] = (
        logical[..., NOPE_DIM:]
        .contiguous()
        .view(torch.uint8)
        .reshape(blocks, block_size, ROPE_DIM * 2)
    )
    scales_region[..., :7] = (exponent.to(torch.int32) + 127).to(torch.uint8)
    dequantized = torch.cat(
        (
            (quantized.float() * scale.unsqueeze(-1)).reshape(
                blocks, block_size, NOPE_DIM
            ),
            logical[..., NOPE_DIM:].float(),
        ),
        dim=-1,
    ).reshape(padded_tokens, HEAD_DIM)
    return cache, dequantized


def _physical_indices(torch, logical_indices, page_table, page_size):
    logical_page = torch.div(logical_indices, page_size, rounding_mode="floor")
    page_offset = logical_indices % page_size
    batch = logical_indices.shape[0]
    expanded = page_table[:, None, :].expand(
        batch, logical_indices.shape[1], page_table.shape[1]
    )
    physical_page = torch.gather(expanded, 2, logical_page.long())
    return (physical_page * page_size + page_offset).to(torch.int32)


def _semantic_oracle(
    torch,
    q,
    swa_values,
    swa_indices,
    swa_lengths,
    extra_values,
    extra_indices,
    extra_lengths,
    sink,
):
    output = torch.empty_like(q, dtype=torch.float32)
    for batch_index in range(q.shape[0]):
        for query_index in range(q.shape[1]):
            primary_count = int(swa_lengths[batch_index].item())
            keys = [
                swa_values[
                    swa_indices[batch_index, query_index, :primary_count].long()
                ]
            ]
            if extra_values is not None:
                extra_count = int(extra_lengths[batch_index].item())
                keys.append(
                    extra_values[
                        extra_indices[
                            batch_index, query_index, :extra_count
                        ].long()
                    ]
                )
            selected = torch.cat(keys, dim=0).float()
            scores = torch.mm(q[batch_index, query_index].float(), selected.t())
            scores = scores * (HEAD_DIM**-0.5)
            all_scores = torch.cat((scores, sink[:, None]), dim=-1)
            probabilities = torch.softmax(all_scores, dim=-1)[
                :, : selected.shape[0]
            ]
            output[batch_index, query_index] = torch.mm(probabilities, selected)
    return output


def _cache_state(prefix, cache):
    return {
        f"{prefix}_head": cache[0, 0, 0, :128],
        f"{prefix}_tail": cache[-1, -1, 0, -128:],
    }


class DeepSeekV4TilelangSparseAttentionSpec:
    operator_id = "deepseek_v4_tilelang_sparse_attention"

    def cases(self):
        cases = []
        seed = 4500
        for phase in ("prefill", "decode"):
            for ratio in (0, 4, 128):
                seed += 1
                cases.append(
                    CaseSpec(
                        f"{phase}_smoke_c{ratio}",
                        {
                            "phase": phase,
                            "batch": 1 if phase == "prefill" else 2,
                            "queries": 2 if phase == "prefill" else 1,
                            "raw_context": 257,
                            "compression_ratio": ratio,
                        },
                        seed,
                        frozenset(
                            {
                                "smoke",
                                "oracle",
                                f"c{ratio}",
                                f"deepseek_v4_{phase}",
                            }
                        ),
                        1200,
                    )
                )
        cases.extend(
            (
                CaseSpec(
                    "prefill_representative_c4_m64_context4096",
                    {
                        "phase": "prefill",
                        "batch": 1,
                        "queries": 64,
                        "raw_context": 4096,
                        "compression_ratio": 4,
                    },
                    4510,
                    frozenset(
                        {
                            "representative",
                            "performance_only",
                            "c4",
                            "deepseek_v4_prefill",
                        }
                    ),
                    2400,
                ),
                CaseSpec(
                    "prefill_representative_c128_m64_context65536",
                    {
                        "phase": "prefill",
                        "batch": 1,
                        "queries": 64,
                        "raw_context": 65536,
                        "compression_ratio": 128,
                    },
                    4511,
                    frozenset(
                        {
                            "representative",
                            "performance_only",
                            "c128",
                            "deepseek_v4_prefill",
                        }
                    ),
                    2400,
                ),
                CaseSpec(
                    "decode_representative_c4_b16_context65536",
                    {
                        "phase": "decode",
                        "batch": 16,
                        "queries": 1,
                        "raw_context": 65536,
                        "compression_ratio": 4,
                        "model_input": 16,
                        "quant_profile": PROFILE,
                        "projection_adapter_id": (
                            "sparse_decode_attention_c4"
                        ),
                    },
                    4512,
                    frozenset(
                        {
                            "representative",
                            "performance_only",
                            "c4",
                            "deepseek_v4_decode",
                            "deepseek_v4_flash_decode_unified",
                        }
                    ),
                    2400,
                ),
                CaseSpec(
                    "decode_representative_c128_b16_context65536",
                    {
                        "phase": "decode",
                        "batch": 16,
                        "queries": 1,
                        "raw_context": 65536,
                        "compression_ratio": 128,
                        "model_input": 16,
                        "quant_profile": PROFILE,
                        "projection_adapter_id": (
                            "sparse_decode_attention_c128"
                        ),
                    },
                    4513,
                    frozenset(
                        {
                            "representative",
                            "performance_only",
                            "c128",
                            "deepseek_v4_decode",
                            "deepseek_v4_flash_decode_unified",
                        }
                    ),
                    2400,
                ),
            )
        )
        for m_index, queries in enumerate(PREFILL_M, start=1):
            for ratio in (4, 128):
                cases.append(
                    CaseSpec(
                        f"prefill_unified_c{ratio}_m{queries}_context65536",
                        {
                            "phase": "prefill",
                            "batch": 1,
                            "queries": queries,
                            "raw_context": 65536,
                            "compression_ratio": ratio,
                            "model_input": queries,
                            "quant_profile": PROFILE,
                            "projection_adapter_id": (
                                f"sparse_prefill_attention_c{ratio}"
                            ),
                        },
                        4550 + m_index * 10 + ratio,
                        frozenset(
                            {
                                "representative",
                                "performance_only",
                                f"c{ratio}",
                                UNIFIED_TAG,
                            }
                        ),
                        3600,
                    )
                )
        return tuple(cases)

    def make_inputs(self, case, context):
        torch = importlib.import_module("torch")
        batch = int(case.symbols["batch"])
        queries = int(case.symbols["queries"])
        raw_context = int(case.symbols["raw_context"])
        ratio = int(case.symbols["compression_ratio"])
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("TileLang sparse attention requires a GPU generator")
        q = torch.randn(
            (batch, queries, HEADS, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).clamp_(-1.0, 1.0)

        swa_logical = torch.randn(
            (SWA_TOKENS, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).clamp_(-1.0, 1.0)
        swa_cache, swa_values = _pack_cache(torch, swa_logical, SWA_TOKENS)
        swa_page_table = torch.zeros((batch, 1), dtype=torch.int32, device=device)
        swa_logical_indices = (
            torch.arange(SWA_TOKENS, dtype=torch.int32, device=device)
            .view(1, 1, SWA_TOKENS)
            .expand(batch, queries, SWA_TOKENS)
        )
        swa_indices = _physical_indices(
            torch, swa_logical_indices, swa_page_table, SWA_TOKENS
        )
        swa_lengths = torch.full(
            (batch,), min(raw_context, SWA_TOKENS), dtype=torch.int32, device=device
        )

        extra_cache = extra_values = extra_indices = extra_lengths = None
        extra_page_table = extra_logical_indices = None
        if ratio:
            valid_extra = min(512, (raw_context + ratio - 1) // ratio)
            padded_extra = max(32, ((valid_extra + 31) // 32) * 32)
            page_size = 64 if ratio == 4 else 2
            pages = (padded_extra + page_size - 1) // page_size
            extra_logical = torch.randn(
                (pages * page_size, HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).clamp_(-1.0, 1.0)
            extra_cache, extra_values = _pack_cache(
                torch, extra_logical, page_size
            )
            base_pages = torch.arange(pages, dtype=torch.int32, device=device)
            extra_page_table = torch.stack(
                [
                    torch.roll(base_pages, shifts=index % max(1, pages))
                    for index in range(batch)
                ]
            )
            extra_logical_indices = (
                torch.arange(padded_extra, dtype=torch.int32, device=device)
                .view(1, 1, padded_extra)
                .expand(batch, queries, padded_extra)
            )
            extra_indices = _physical_indices(
                torch, extra_logical_indices, extra_page_table, page_size
            )
            extra_lengths = torch.full(
                (batch,), valid_extra, dtype=torch.int32, device=device
            )
        sink = torch.linspace(
            -3.0, -1.0, HEADS, dtype=torch.float32, device=device
        )
        observed = {
            **_cache_state("swa_cache", swa_cache),
            "swa_page_table": swa_page_table,
            "swa_indices_head": swa_indices[:1, :1, :32],
            "compression_ratio": ratio,
        }
        if extra_cache is not None:
            observed.update(
                {
                    **_cache_state("extra_cache", extra_cache),
                    "extra_page_table_head": extra_page_table[
                        : min(2, batch), : min(16, extra_page_table.shape[1])
                    ],
                    "extra_indices_head": extra_indices[:1, :1, :32],
                }
            )
        if "performance_only" not in case.tags:
            oracle = _semantic_oracle(
                torch,
                q,
                swa_values,
                swa_indices,
                swa_lengths,
                extra_values,
                extra_indices,
                extra_lengths,
                sink,
            )
            observed["semantic_oracle"] = sample_tensor(oracle)
        return InputBundle(
            args=(
                q,
                swa_cache,
                swa_indices,
                swa_lengths,
                extra_cache,
                extra_indices,
                extra_lengths,
                sink,
            ),
            observed_state=observed,
        )

    def clone_inputs(self, inputs):
        torch = importlib.import_module("torch")
        (
            q,
            swa_cache,
            swa_indices,
            swa_lengths,
            extra_cache,
            extra_indices,
            extra_lengths,
            sink,
        ) = inputs.args
        swa2 = _clone_padded_cache(torch, swa_cache)
        extra2 = (
            None
            if extra_cache is None
            else _clone_padded_cache(torch, extra_cache)
        )
        observed = {
            **_cache_state("swa_cache", swa2),
            "swa_page_table": inputs.observed_state["swa_page_table"].clone(),
            "swa_indices_head": swa_indices[:1, :1, :32].clone(),
            "compression_ratio": inputs.observed_state["compression_ratio"],
        }
        if extra2 is not None:
            observed.update(
                {
                    **_cache_state("extra_cache", extra2),
                    "extra_page_table_head": inputs.observed_state[
                        "extra_page_table_head"
                    ].clone(),
                    "extra_indices_head": extra_indices[:1, :1, :32].clone(),
                }
            )
        if "semantic_oracle" in inputs.observed_state:
            observed["semantic_oracle"] = inputs.observed_state[
                "semantic_oracle"
            ].clone()
        return InputBundle(
            args=(
                q.clone(),
                swa2,
                swa_indices.clone(),
                swa_lengths.clone(),
                extra2,
                None if extra_indices is None else extra_indices.clone(),
                None if extra_lengths is None else extra_lengths.clone(),
                sink.clone(),
            ),
            observed_state=observed,
        )

    def normalize_output(self, output):
        return normalize_named_tensors(output)

    def comparator(self, case):
        numeric = [
            NumericPath("output", 0.08, 0.02, "state.semantic_oracle"),
            NumericPath("state.swa_cache_head", 0.0),
            NumericPath("state.swa_cache_tail", 0.0),
        ]
        immutable = [
            "state.swa_page_table",
            "state.swa_indices_head",
            "state.compression_ratio",
        ]
        if int(case.symbols["compression_ratio"]):
            numeric.extend(
                (
                    NumericPath("state.extra_cache_head", 0.0),
                    NumericPath("state.extra_cache_tail", 0.0),
                )
            )
            immutable.extend(
                ("state.extra_page_table_head", "state.extra_indices_head")
            )
        return OracleStateComparator(
            numeric_paths=numeric,
            immutable_paths=immutable,
            require_oracle="performance_only" not in case.tags,
            name="tilelang_sparse_attention_oracle",
        )

    def cost_model(self, case):
        batch = int(case.symbols["batch"])
        queries = int(case.symbols["queries"])
        raw = int(case.symbols["raw_context"])
        ratio = int(case.symbols["compression_ratio"])
        attended = min(raw, SWA_TOKENS)
        if ratio:
            attended += min(512, (raw + ratio - 1) // ratio)
        return {
            "flops": 4 * batch * queries * HEADS * attended * HEAD_DIM,
            "estimated_bytes": (
                batch * queries * HEADS * HEAD_DIM * 2
                + attended * TOKEN_BYTES
                + batch * queries * HEADS * HEAD_DIM * 2
            ),
            "throughput_units": batch * queries * attended,
        }

    def layout_contract(self, case):
        return {
            "q": [
                int(case.symbols["batch"]),
                int(case.symbols["queries"]),
                HEADS,
                HEAD_DIM,
            ],
            "kv_layout": "MODEL1_FP8Sparse: 448 FP8 + 64 BF16 + 7 UE8M0 + pad",
            "compression_ratio": int(case.symbols["compression_ratio"]),
            "page_table": "logical pages translated to physical token indices",
            "cache_mutation": "forbidden and observed",
            "backend": (
                "SGLang DSV4 sparse attention: Triton on pinned gfx942 "
                "build, TileLang opt-in with compiler-failure fallback"
            ),
        }


SPEC = DeepSeekV4TilelangSparseAttentionSpec()


def cost_model(case):
    return SPEC.cost_model(case)

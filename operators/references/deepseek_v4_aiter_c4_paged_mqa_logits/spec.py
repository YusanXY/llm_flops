"""Contract for AITER C4 FP8 paged-MQA index logits on MI300X."""

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
HEAD_DIM = 128
PAGE_SIZE = 64
RATIO = 4
PREFILL_M = (1024, 2048, 4096)
PROFILE = "fp8_block_gfx942"
UNIFIED_TAG = "deepseek_v4_flash_prefill_unified"


def _unified_cases():
    return tuple(
        CaseSpec(
            f"prefill_unified_m{queries}_context65536",
            {
                "phase": "prefill",
                "queries": queries,
                "raw_context": 65536,
                "model_input": queries,
                "quant_profile": PROFILE,
                "projection_adapter_id": "c4_fp8_paged_mqa_logits",
            },
            4450 + index,
            frozenset(
                {
                    "representative",
                    "performance_only",
                    UNIFIED_TAG,
                }
            ),
            3600,
        )
        for index, queries in enumerate(PREFILL_M, start=1)
    )


def _pack_and_shuffle(torch, logical_cache):
    shuffle = importlib.import_module("aiter.ops.shuffle").shuffle_weight
    pages = logical_cache.shape[0]
    maximum = logical_cache.float().abs().amax(dim=-1).clamp_min(1.0e-4)
    scale = maximum / 240.0
    quantized = (logical_cache.float() / scale.unsqueeze(-1)).to(
        torch.float8_e4m3fnuz
    )
    packed = torch.empty(
        (pages, PAGE_SIZE * (HEAD_DIM + 4)),
        dtype=torch.uint8,
        device=logical_cache.device,
    )
    value_bytes = PAGE_SIZE * HEAD_DIM
    packed[:, :value_bytes] = quantized.reshape(pages, value_bytes).view(
        torch.uint8
    )
    packed[:, value_bytes:] = scale.contiguous().view(torch.uint8).reshape(
        pages, PAGE_SIZE * 4
    )
    values = packed[:, :value_bytes].contiguous().view(
        pages, PAGE_SIZE, HEAD_DIM
    )
    packed[:, :value_bytes] = shuffle(values).reshape(pages, value_bytes)
    return (
        packed.view(pages, PAGE_SIZE, 1, HEAD_DIM + 4),
        quantized.float() * scale.unsqueeze(-1),
    )


def _semantic_oracle(torch, q, cache, weights, lengths, page_table, max_len):
    rows = []
    for batch_index in range(q.shape[0]):
        logical = cache[page_table[batch_index].long()].reshape(-1, HEAD_DIM)
        scores = torch.mm(logical.float(), q[batch_index, 0].float().t())
        scores = torch.relu(scores)
        scores = (scores * weights[batch_index].float().unsqueeze(0)).sum(dim=-1)
        valid = int(lengths[batch_index].item())
        row = torch.zeros((max_len,), dtype=torch.float32, device=q.device)
        row[:valid] = scores[:valid]
        rows.append(row)
    return torch.stack(rows)


def _cache_views(cache):
    raw = cache.reshape(cache.shape[0], -1)
    return {
        "cache_head": raw[:1, :256],
        "cache_tail": raw[-1:, -256:],
    }


class DeepSeekV4AiterC4PagedMqaSpec:
    operator_id = "deepseek_v4_aiter_c4_paged_mqa_logits"

    def cases(self):
        return (
            CaseSpec(
                "prefill_smoke_m4_context257",
                {"phase": "prefill", "queries": 4, "raw_context": 257},
                4401,
                frozenset(
                    {"smoke", "oracle", "tail_page", "deepseek_v4_prefill"}
                ),
                900,
            ),
            CaseSpec(
                "decode_smoke_b2_context257",
                {"phase": "decode", "queries": 2, "raw_context": 257},
                4402,
                frozenset(
                    {"smoke", "oracle", "tail_page", "deepseek_v4_decode"}
                ),
                900,
            ),
            CaseSpec(
                "prefill_representative_m64_context4096",
                {"phase": "prefill", "queries": 64, "raw_context": 4096},
                4403,
                frozenset(
                    {"representative", "performance_only", "deepseek_v4_prefill"}
                ),
                1800,
            ),
            CaseSpec(
                "decode_representative_b16_context65536",
                {
                    "phase": "decode",
                    "queries": 16,
                    "raw_context": 65536,
                    "model_input": 16,
                    "quant_profile": PROFILE,
                    "projection_adapter_id": "c4_fp8_paged_mqa_logits",
                },
                4404,
                frozenset(
                    {
                        "representative",
                        "performance_only",
                        "deepseek_v4_decode",
                        "deepseek_v4_flash_decode_unified",
                    }
                ),
                1800,
            ),
        ) + _unified_cases()

    def make_inputs(self, case, context):
        torch = importlib.import_module("torch")
        queries = int(case.symbols["queries"])
        raw_context = int(case.symbols["raw_context"])
        compressed = (raw_context + RATIO - 1) // RATIO
        pages = (compressed + PAGE_SIZE - 1) // PAGE_SIZE
        max_len = pages * PAGE_SIZE
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("paged-MQA logits requires a GPU generator")
        q = torch.randn(
            (queries, 1, HEADS, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).clamp_(-2.0, 2.0)
        q_fp8 = q.to(torch.float8_e4m3fnuz)
        logical_cache = torch.randn(
            (pages, PAGE_SIZE, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).clamp_(-2.0, 2.0)
        packed, dequantized = _pack_and_shuffle(torch, logical_cache)
        base = torch.arange(pages, dtype=torch.int32, device=device)
        page_table = torch.stack(
            [
                torch.roll(base, shifts=index % max(1, pages))
                for index in range(queries)
            ]
        )
        context_lens = torch.full(
            (queries,), compressed, dtype=torch.int32, device=device
        )
        weights = torch.randn(
            (queries, HEADS),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        output = torch.empty(
            (queries, max_len), dtype=torch.float32, device=device
        )
        observed = {
            **_cache_views(packed),
            "page_table_head": page_table[: min(2, queries), : min(16, pages)],
            "page_table_tail": page_table[-1:, -min(16, pages) :],
            "context_lens": context_lens,
        }
        if "performance_only" not in case.tags:
            oracle = _semantic_oracle(
                torch,
                q_fp8,
                dequantized,
                weights,
                context_lens,
                page_table,
                max_len,
            )
            observed["semantic_oracle"] = sample_tensor(oracle[:, :compressed])
        return InputBundle(
            args=(
                q_fp8,
                packed,
                weights,
                output,
                context_lens,
                page_table,
                max_len,
                compressed,
            ),
            observed_state=observed,
        )

    def clone_inputs(self, inputs):
        q, cache, weights, output, lengths, table, max_len, valid_width = inputs.args
        cache2, lengths2, table2 = cache.clone(), lengths.clone(), table.clone()
        observed = {
            **_cache_views(cache2),
            "page_table_head": table2[
                : min(2, table2.shape[0]), : min(16, table2.shape[1])
            ],
            "page_table_tail": table2[-1:, -min(16, table2.shape[1]) :],
            "context_lens": lengths2,
        }
        if "semantic_oracle" in inputs.observed_state:
            observed["semantic_oracle"] = inputs.observed_state[
                "semantic_oracle"
            ].clone()
        return InputBundle(
            args=(
                q.clone(),
                cache2,
                weights.clone(),
                output.clone(),
                lengths2,
                table2,
                max_len,
                valid_width,
            ),
            observed_state=observed,
        )

    def normalize_output(self, output):
        return normalize_named_tensors(output)

    def comparator(self, case):
        return OracleStateComparator(
            numeric_paths=(
                NumericPath(
                    "output", 0.5, 0.02, "state.semantic_oracle"
                ),
                NumericPath("state.cache_head", 0.0),
                NumericPath("state.cache_tail", 0.0),
            ),
            immutable_paths=(
                "state.page_table_head",
                "state.page_table_tail",
                "state.context_lens",
            ),
            require_oracle="performance_only" not in case.tags,
            name="aiter_c4_paged_mqa_oracle",
        )

    def cost_model(self, case):
        queries = int(case.symbols["queries"])
        raw_context = int(case.symbols["raw_context"])
        compressed = (raw_context + RATIO - 1) // RATIO
        return {
            "flops": 2 * queries * HEADS * compressed * HEAD_DIM,
            "estimated_bytes": (
                queries * HEADS * HEAD_DIM
                + compressed * (HEAD_DIM + 4)
                + queries * compressed * 4
            ),
            "throughput_units": queries * compressed,
        }

    def layout_contract(self, case):
        queries = int(case.symbols["queries"])
        compressed = (int(case.symbols["raw_context"]) + RATIO - 1) // RATIO
        pages = (compressed + PAGE_SIZE - 1) // PAGE_SIZE
        return {
            "q": [queries, 1, HEADS, HEAD_DIM],
            "cache": [pages, PAGE_SIZE, 1, HEAD_DIM + 4],
            "page_table": [queries, pages],
            "compressed_context": compressed,
            "cache_mutation": "forbidden and observed",
            "backend": "AITER deepgemm_fp8_paged_mqa_logits Preshuffle=True",
        }


SPEC = DeepSeekV4AiterC4PagedMqaSpec()


def cost_model(case):
    return SPEC.cost_model(case)

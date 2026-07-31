"""Contract for the closed DeepSeek V4 Indexer-K -> paged-MQA mega operator."""

from __future__ import annotations

import importlib
import math

from benchmark_engine.correctness import (
    ExactComparator,
    FloatingComparator,
    InputBundle,
    Tolerance,
)
from benchmark_engine.correctness.models import ComparisonResult, OutputBundle, OutputLeaf
from benchmark_engine.models import CaseSpec


OPERATOR_ID = "deepseek_v4_mega_mqa_logits"
RAW_CONTEXT = 65536
COMPRESSION_RATIO = 4
COMPRESSED_CONTEXT = RAW_CONTEXT // COMPRESSION_RATIO
HIDDEN_SIZE = 7168
HEADS = 64
HEAD_DIM = 128
WKV_GATE_DIM = 4 * HEAD_DIM
PAGE_SIZE = 64
C4_WINDOW = 8
RMS_EPS = 1e-6
RTOL = 1e-5
ATOL = 1e-6
TOLERANCE = Tolerance(RTOL, ATOL, False, "mega_operator")


def _formal_case(m: int, seed: int) -> CaseSpec:
    return CaseSpec(
        case_id=f"formal__mega_mqa_logits__m{m}__ctx65536",
        symbols={
            "m": m,
            "raw_context": RAW_CONTEXT,
            "compressed_context": COMPRESSED_CONTEXT,
            "compression_ratio": COMPRESSION_RATIO,
            "hidden_size": HIDDEN_SIZE,
            "wkv_gate_dim": WKV_GATE_DIM,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "page_size": PAGE_SIZE,
            "c4_window": C4_WINDOW,
            "mode": "single_request_causal_prefill",
            "request_count": 1,
            "query_tokens": m,
        },
        seed=seed,
        tags=frozenset(
            {
                "formal",
                "representative",
                "mega",
                f"m_{m}",
                "context_65536",
            }
        ),
        timeout_s=3600,
    )


def _runtime():
    torch = importlib.import_module("torch")
    deep_gemm = importlib.import_module("deep_gemm")
    return torch, deep_gemm


def _sample_indices(torch, total, device, limit=256):
    total = int(total)
    count = min(total, int(limit))
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    positions = torch.arange(count, device=device, dtype=torch.long)
    span = total - 1
    intervals = count - 1
    return positions * (span // intervals) + positions * (span % intervals) // intervals


def _bounded_output(output, limit=256):
    torch = importlib.import_module("torch")
    m, max_context = output.shape
    rows = _sample_indices(torch, m, output.device, limit)
    causal_raw_lens = RAW_CONTEXT - m + rows + 1
    valid_lens = torch.div(
        causal_raw_lens, COMPRESSION_RATIO, rounding_mode="floor"
    )
    sample_ordinals = torch.arange(
        rows.numel(), device=output.device, dtype=torch.long
    )
    columns = torch.remainder(
        sample_ordinals * 2654435761 + rows * 2246822519,
        valid_lens.to(torch.long),
    )
    values = tuple(output.detach()[rows, columns].float().cpu().tolist())
    return OutputBundle(
        (
            OutputLeaf(
                "output",
                values,
                str(output.dtype),
                (m, max_context),
                tuple(output.stride()),
                "strided",
                str(output.device),
            ),
        )
    )


def _subset(bundle, predicate):
    return OutputBundle(tuple(leaf for leaf in bundle.leaves if predicate(leaf.path)))


class MegaMqaComparator:
    """Strict floating math plus exact FP8 codes and integer metadata."""

    @staticmethod
    def _is_exact(path):
        return path.startswith("state.cache_k_") or path.startswith(
            "state.block_table_"
        )

    def compare(self, reference, candidate, **_):
        exact = ExactComparator().compare(
            _subset(reference, self._is_exact),
            _subset(candidate, self._is_exact),
        )
        floating = FloatingComparator(
            operator_default=TOLERANCE, require_explicit=True
        ).compare(
            _subset(reference, lambda path: not self._is_exact(path)),
            _subset(candidate, lambda path: not self._is_exact(path)),
        )
        passed = exact.passed and floating.passed
        return ComparisonResult(
            passed,
            "mega_mqa_strict_math_and_exact_physical_cache",
            {"floating": floating.metrics, "exact_state": exact.metrics},
            tuple((floating.diagnostics + exact.diagnostics)[:8]),
            None if passed else (floating.failed_path or exact.failed_path),
        )


def _validate(symbols):
    expected = {
        "raw_context": RAW_CONTEXT,
        "compressed_context": COMPRESSED_CONTEXT,
        "compression_ratio": COMPRESSION_RATIO,
        "hidden_size": HIDDEN_SIZE,
        "wkv_gate_dim": WKV_GATE_DIM,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "page_size": PAGE_SIZE,
        "c4_window": C4_WINDOW,
        "mode": "single_request_causal_prefill",
        "request_count": 1,
    }
    m = int(symbols["m"])
    if m not in {1024, 2048, 4096}:
        raise NotImplementedError("formal mega baseline supports m=1024/2048/4096")
    if symbols.get("query_tokens") != m:
        raise NotImplementedError(
            f"query_tokens must equal m, got {symbols.get('query_tokens')!r}"
        )
    for name, value in expected.items():
        if symbols.get(name) != value:
            raise NotImplementedError(f"unsupported {name}: {symbols.get(name)!r}")
    return m


def _cache_views(cache):
    pages = cache.shape[0]
    middle = pages // 2
    page_bytes = PAGE_SIZE * (HEAD_DIM + 4)
    cache_2d = cache.view(pages, page_bytes)

    def page(prefix, index):
        page_bytes_view = cache_2d[index : index + 1]
        return {
            f"cache_k_{prefix}": page_bytes_view[:, : PAGE_SIZE * HEAD_DIM],
            f"cache_scale_{prefix}": page_bytes_view[
                :, PAGE_SIZE * HEAD_DIM :
            ].view(importlib.import_module("torch").float32),
        }

    return {
        **page("head", 0),
        **page("middle", middle),
        **page("tail", pages - 1),
    }


def _state_views(c4_state, page_table):
    middle = c4_state.shape[0] // 2
    return {
        "compressor_state_head": c4_state[:1],
        "compressor_state_middle": c4_state[middle : middle + 1],
        "compressor_state_tail": c4_state[-1:],
        "block_table_head": page_table[:1],
        "block_table_tail": page_table[-1:],
    }


def _observed_state(cache, c4_state, page_table):
    return {**_cache_views(cache), **_state_views(c4_state, page_table)}


def _check_available_memory(required_bytes, available_bytes):
    if required_bytes > int(available_bytes * 0.85):
        raise MemoryError(
            f"estimated canonical-plus-two-clones allocation {required_bytes} "
            f"exceeds safe CUDA memory budget {available_bytes}"
        )


class DeepSeekV4MegaMqaLogitsSpec:
    operator_id = OPERATOR_ID

    def cases(self):
        return (
            _formal_case(1024, 401),
            _formal_case(2048, 403),
            _formal_case(4096, 409),
        )

    def layout_contract(self, case):
        m = _validate(case.symbols)
        pages_per_sequence = COMPRESSED_CONTEXT // PAGE_SIZE
        return {
            "hidden_states": [m, HIDDEN_SIZE],
            "wkv_gate": [WKV_GATE_DIM, HIDDEN_SIZE],
            "c4_state": [C4_WINDOW, WKV_GATE_DIM],
            "request_count": 1,
            "query_tokens": m,
            "q_fp8": [1, m, HEADS, HEAD_DIM],
            "physical_cache": [
                pages_per_sequence,
                PAGE_SIZE,
                1,
                HEAD_DIM + 4,
            ],
            "physical_page_bytes": PAGE_SIZE * (HEAD_DIM + 4),
            "cache_layout": "8192 FP8 K bytes followed by 256 FP32 scale bytes",
            "page_table": [1, pages_per_sequence],
            "page_table_semantics": "one physical page-table row for one request",
            "context_lens": [1, m],
            "context_lens_semantics": "floor((65536-m+j+1)/4) for query j",
            "cache_mutation": "m/4 contiguous compressed tail tokens",
            "compressor_state_mutation": "final eight raw records of one request",
            "output": [m, COMPRESSED_CONTEXT],
            "raw_context": RAW_CONTEXT,
            "compressed_context": COMPRESSED_CONTEXT,
            "relu": "per_head_before_weighted_head_reduction",
            "workspace_lifecycle": "all sequential intermediates are inside the timed callable",
        }

    def make_inputs(self, case, context):
        torch, deep_gemm = _runtime()
        m = _validate(case.symbols)
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("mega MQA logits requires a CUDA generator")

        pages_per_sequence = COMPRESSED_CONTEXT // PAGE_SIZE
        blocks = pages_per_sequence
        produced_k = m // COMPRESSION_RATIO
        prefix_raw_context = RAW_CONTEXT - m
        prefix_compressed_context = prefix_raw_context // COMPRESSION_RATIO
        page_bytes = PAGE_SIZE * (HEAD_DIM + 4)
        cache_bytes = blocks * page_bytes
        output_bytes = m * COMPRESSED_CONTEXT * 4
        producer_bytes = (
            m * HIDDEN_SIZE * 2
            + WKV_GATE_DIM * HIDDEN_SIZE * 2
            + C4_WINDOW * WKV_GATE_DIM * 4
            + m * HEADS * HEAD_DIM
            + m * HEADS * 4
        )
        estimated_three_way = 3 * (cache_bytes + output_bytes + producer_bytes)
        free, _ = torch.cuda.mem_get_info(device)
        _check_available_memory(estimated_three_way, free)

        hidden_states = torch.randn(
            (m, HIDDEN_SIZE), device=device, dtype=torch.bfloat16, generator=generator
        ).mul_(0.05)
        wkv_gate = torch.randn(
            (WKV_GATE_DIM, HIDDEN_SIZE),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        ).mul_(0.02)
        c4_state = torch.randn(
            (C4_WINDOW, WKV_GATE_DIM),
            device=device,
            dtype=torch.float32,
            generator=generator,
        ).mul_(0.1)
        ape = torch.randn(
            (C4_WINDOW, HEAD_DIM),
            device=device,
            dtype=torch.float32,
            generator=generator,
        ).mul_(0.05)
        norm_weight = torch.ones((HEAD_DIM,), device=device, dtype=torch.float32)

        rope_frequency = 1.0 / (
            10000.0
            ** (
                torch.arange(0, 64, 2, device=device, dtype=torch.float32)
                / 64.0
            )
        )
        compressed_raw_positions = torch.arange(
            prefix_raw_context,
            RAW_CONTEXT,
            COMPRESSION_RATIO,
            device=device,
            dtype=torch.float32,
        )
        if compressed_raw_positions.numel() != produced_k:
            raise AssertionError("incorrect number of C4 producer positions")
        rope_phase = compressed_raw_positions.unsqueeze(1) * rope_frequency.unsqueeze(0)
        rope_cos = torch.cos(rope_phase)
        rope_sin = torch.sin(rope_phase)

        q_codes = torch.randint(
            -4,
            5,
            (1, m, HEADS, HEAD_DIM),
            device=device,
            dtype=torch.int8,
            generator=generator,
        )
        q_fp8 = q_codes.to(torch.float8_e4m3fn)
        weights = torch.randn(
            (m, HEADS), device=device, dtype=torch.float32, generator=generator
        ).mul_(HEADS**-0.5)

        request_page_table = torch.arange(
            blocks, device=device, dtype=torch.int32
        ).view(1, pages_per_sequence)
        page_table = request_page_table.contiguous()
        causal_raw_lens = torch.arange(
            prefix_raw_context + 1,
            RAW_CONTEXT + 1,
            device=device,
            dtype=torch.int32,
        )
        context_lens = torch.div(
            causal_raw_lens, COMPRESSION_RATIO, rounding_mode="floor"
        ).view(1, m)
        schedule = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, PAGE_SIZE, deep_gemm.get_num_sms()
        )

        kv_fused = torch.empty(
            (blocks, PAGE_SIZE, 1, HEAD_DIM + 4),
            device=device,
            dtype=torch.uint8,
        )
        cache_2d = kv_fused.view(blocks, page_bytes)
        cache_2d[:, : PAGE_SIZE * HEAD_DIM].view(torch.float8_e4m3fn).fill_(1.0)
        cache_2d[:, PAGE_SIZE * HEAD_DIM :].view(torch.float32).fill_(1.0)

        cache_write_locs = torch.arange(
            prefix_compressed_context,
            COMPRESSED_CONTEXT,
            device=device,
            dtype=torch.int64,
        )
        observed = _observed_state(kv_fused, c4_state, page_table)
        return InputBundle(
            args=(
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
                RAW_CONTEXT,
                COMPRESSED_CONTEXT,
                RMS_EPS,
            ),
            observed_state=observed,
        )

    def clone_inputs(self, inputs):
        torch, deep_gemm = _runtime()
        (
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
            _,
            cache_write_locs,
            raw_context,
            compressed_context,
            rms_eps,
        ) = inputs.args
        cloned = (
            hidden_states.clone(),
            wkv_gate.clone(),
            c4_state.clone(),
            ape.clone(),
            norm_weight.clone(),
            rope_cos.clone(),
            rope_sin.clone(),
            q_fp8.clone(),
            weights.clone(),
            kv_fused.clone(),
            context_lens.clone(),
            page_table.clone(),
        )
        schedule = deep_gemm.get_paged_mqa_logits_metadata(
            cloned[10], PAGE_SIZE, deep_gemm.get_num_sms()
        )
        locs = cache_write_locs.clone()
        args = (
            *cloned,
            schedule,
            locs,
            raw_context,
            compressed_context,
            rms_eps,
        )
        observed = _observed_state(cloned[9], cloned[2], cloned[11])
        return InputBundle(args=args, observed_state=observed)

    def normalize_output(self, output):
        return _bounded_output(output)

    def comparator(self, case):
        _validate(case.symbols)
        return MegaMqaComparator()

    def cost_model(self, case):
        m = _validate(case.symbols)
        wkv_flops = 2 * m * HIDDEN_SIZE * WKV_GATE_DIM
        produced_k = m // COMPRESSION_RATIO
        prefix_raw_context = RAW_CONTEXT - m
        causal_raw_lens = range(prefix_raw_context + 1, RAW_CONTEXT + 1)
        total_visible_k = sum(length // COMPRESSION_RATIO for length in causal_raw_lens)
        c4_softmax_and_reduce = produced_k * HEAD_DIM * C4_WINDOW * 6
        rms_rope_fwht = produced_k * HEAD_DIM * (
            5 + 3 + int(math.log2(HEAD_DIM))
        )
        mqa_dot = 2 * HEADS * total_visible_k * HEAD_DIM
        mqa_head_reduce = 2 * HEADS * total_visible_k
        flops = (
            wkv_flops
            + c4_softmax_and_reduce
            + rms_rope_fwht
            + mqa_dot
            + mqa_head_reduce
        )
        estimated_bytes = (
            m * HIDDEN_SIZE * 2
            + WKV_GATE_DIM * HIDDEN_SIZE * 2
            + C4_WINDOW * WKV_GATE_DIM * 4
            + m * HEADS * HEAD_DIM
            + m * HEADS * 4
            + total_visible_k * (HEAD_DIM + 4)
            + produced_k * (HEAD_DIM + 4)
            + m * COMPRESSED_CONTEXT * 4
        )
        return {
            "flops": flops,
            "estimated_bytes": estimated_bytes,
            "throughput_units": m * COMPRESSED_CONTEXT,
        }

    def workspace_bytes(self, case):
        _validate(case.symbols)
        return None


SPEC = DeepSeekV4MegaMqaLogitsSpec()


def cost_model(case):
    return SPEC.cost_model(case)

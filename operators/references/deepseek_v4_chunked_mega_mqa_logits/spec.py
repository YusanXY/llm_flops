"""Contract and independent oracle for DSV4 full-KV-reuse paged MQA.

The CaseSpec carries both the complete physical C4 Indexer cache and the
upstream activation/parameter representation that reconstructs the same
cache.  The reference times only the full-cache reuse path.  This makes the
control mathematically identical to DeepSeek V4 while giving candidates a
legal boundary for hybrid historical-K recomputation.
"""

from __future__ import annotations

import importlib
import math

from benchmark_engine.correctness import (
    ExactComparator,
    FloatingComparator,
    InputBundle,
    Tolerance,
)
from benchmark_engine.correctness.models import (
    ComparisonResult,
    OutputBundle,
    OutputLeaf,
)
from benchmark_engine.models import CaseSpec


OPERATOR_ID = "deepseek_v4_chunked_mega_mqa_logits"
RAW_CONTEXT = 65536
COMPRESSION_RATIO = 4
COMPRESSED_CONTEXT = RAW_CONTEXT // COMPRESSION_RATIO
HIDDEN_SIZE = 7168
HEADS = 64
HEAD_DIM = 128
WKV_GATE_DIM = 4 * HEAD_DIM
PAGE_SIZE = 64
RMS_EPS = 1e-6
COMPRESS_ROPE_THETA = 40000.0
RTOL = 1e-5
ATOL = 1e-6
FP8_MAX = 448.0
TOLERANCE = Tolerance(RTOL, ATOL, False, "dsv4_full_kv_reuse")


def _formal_case(m: int, seed: int) -> CaseSpec:
    raw_prefix = RAW_CONTEXT - m
    return CaseSpec(
        case_id=f"formal__full_kv_reuse_mqa__m{m}__ctx65536",
        symbols={
            "m": m,
            "raw_prefix": raw_prefix,
            "raw_context": RAW_CONTEXT,
            "compressed_context": COMPRESSED_CONTEXT,
            "compression_ratio": COMPRESSION_RATIO,
            "hidden_size": HIDDEN_SIZE,
            "wkv_gate_dim": WKV_GATE_DIM,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "page_size": PAGE_SIZE,
            "mode": "single_request_causal_prefill_full_kv_reuse",
        },
        seed=seed,
        tags=frozenset(
            {
                "formal",
                "representative",
                "mega",
                "causal_prefill",
                "full_kv_reuse_control",
                "hybrid_recompute_boundary",
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
        "mode": "single_request_causal_prefill_full_kv_reuse",
    }
    m = int(symbols["m"])
    if m not in {1024, 2048, 4096}:
        raise NotImplementedError("formal full-reuse MQA supports m=1024/2048/4096")
    raw_prefix = RAW_CONTEXT - m
    if raw_prefix % COMPRESSION_RATIO:
        raise NotImplementedError("formal raw prefix must be C4 aligned")
    if symbols.get("raw_prefix") != raw_prefix:
        raise NotImplementedError("raw_prefix does not match raw_context-m")
    for name, value in expected.items():
        if symbols.get(name) != value:
            raise NotImplementedError(f"unsupported {name}: {symbols.get(name)!r}")
    return m, raw_prefix


def _sample_valid_output_indices(torch, m, device, limit=256):
    """Spread samples over rows and the prefix valid for every query."""

    count = min(int(limit), m * COMPRESSED_CONTEXT)
    ordinal = torch.arange(count, device=device, dtype=torch.long)
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    rows = ordinal * (m - 1) // (count - 1)
    minimum_c4_len = (RAW_CONTEXT - m) // COMPRESSION_RATIO
    positions = ordinal * (minimum_c4_len - 1) // (count - 1)
    return rows * COMPRESSED_CONTEXT + positions


def _bounded_output(output, limit=256):
    torch = importlib.import_module("torch")
    flat = output.detach().reshape(-1)
    indices = _sample_valid_output_indices(
        torch, output.shape[0], flat.device, limit
    )
    values = tuple(flat[indices].float().cpu().tolist())
    return OutputBundle(
        (
            OutputLeaf(
                "output",
                values,
                str(output.dtype),
                tuple(output.shape),
                tuple(output.stride()),
                "strided",
                str(output.device),
            ),
        )
    )


def _subset(bundle, predicate):
    return OutputBundle(tuple(leaf for leaf in bundle.leaves if predicate(leaf.path)))


def _maximum_abs(left, right):
    if len(left) != len(right):
        return float("inf")
    return max(
        (abs(float(a) - float(b)) for a, b in zip(left, right)),
        default=0.0,
    )


class FullKvReuseComparator:
    """Require candidate/reference agreement and independent semantic checks."""

    @staticmethod
    def _is_exact(path):
        return (
            path.startswith("state.cache_sample_pages")
            or path.startswith("state.expected_cache_sample_pages")
            or path.startswith("state.block_table_")
        )

    @staticmethod
    def _self_check(role, bundle):
        leaves = bundle.by_path()
        failures = []
        metrics = {}

        output = leaves.get("output")
        oracle = leaves.get("state.semantic_oracle")
        if output is None or oracle is None:
            failures.append({"role": role, "error": "missing semantic oracle"})
        else:
            error = _maximum_abs(output.value, oracle.value)
            bound = ATOL + RTOL * max(
                (abs(float(v)) for v in oracle.value), default=0.0
            )
            metrics[f"{role}_output_oracle_max_abs"] = error
            metrics[f"{role}_output_oracle_bound"] = bound
            if error > bound:
                failures.append(
                    {"role": role, "path": "output", "max_abs_error": error}
                )

        actual = leaves.get("state.cache_sample_pages")
        expected = leaves.get("state.expected_cache_sample_pages")
        if actual is None or expected is None:
            failures.append({"role": role, "error": "missing cache immutability state"})
        elif actual.value != expected.value:
            failures.append(
                {
                    "role": role,
                    "path": "state.cache_sample_pages",
                    "error": "physical cache changed",
                }
            )
        return failures, metrics

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
        ref_failures, ref_metrics = self._self_check("reference", reference)
        cand_failures, cand_metrics = self._self_check("candidate", candidate)
        failures = list(ref_failures + cand_failures)
        if not exact.passed:
            failures.extend(exact.diagnostics)
        if not floating.passed:
            failures.extend(floating.diagnostics)
        passed = exact.passed and floating.passed and not failures
        return ComparisonResult(
            passed,
            "dsv4_full_kv_reuse_cross_and_independent_oracle",
            {
                "floating": floating.metrics,
                "exact_state": exact.metrics,
                **ref_metrics,
                **cand_metrics,
            },
            tuple(failures[:8]),
            None if passed else "output_or_cache",
        )


def _oracle_full_history_cache(
    torch,
    history_hidden,
    wkv_gate,
    ape,
    norm_weight,
    rope_cos,
    rope_sin,
):
    """Independent vectorized reconstruction of the complete physical cache."""

    projected = torch.mm(history_hidden, wkv_gate.t(), out_dtype=torch.float32)
    groups = projected.reshape(
        COMPRESSED_CONTEXT, COMPRESSION_RATIO, WKV_GATE_DIM
    )
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
    transformed = sum_product / sum_exp

    variance = transformed.square().mean(dim=-1, keepdim=True)
    transformed = transformed * torch.rsqrt(variance + RMS_EPS) * norm_weight
    rope = transformed[:, 64:].reshape(-1, 32, 2)
    real, imag = rope[..., 0].clone(), rope[..., 1].clone()
    rope[..., 0] = real * rope_cos - imag * rope_sin
    rope[..., 1] = real * rope_sin + imag * rope_cos

    width = 1
    while width < HEAD_DIM:
        fwht_groups = transformed.reshape(
            -1, HEAD_DIM // (2 * width), 2, width
        )
        left = fwht_groups[:, :, 0]
        right = fwht_groups[:, :, 1]
        transformed = torch.cat((left + right, left - right), dim=-1).reshape(
            COMPRESSED_CONTEXT, HEAD_DIM
        )
        width *= 2
    transformed = transformed * (HEAD_DIM**-0.5)

    scale = torch.clamp(
        transformed.abs().amax(dim=-1, keepdim=True), min=1e-4
    ) / FP8_MAX
    quantized = torch.clamp(
        transformed / scale, -FP8_MAX, FP8_MAX
    ).to(torch.float8_e4m3fn)

    pages = COMPRESSED_CONTEXT // PAGE_SIZE
    page_bytes = PAGE_SIZE * (HEAD_DIM + 4)
    cache = torch.empty(
        (pages, PAGE_SIZE, 1, HEAD_DIM + 4),
        device=history_hidden.device,
        dtype=torch.uint8,
    )
    cache_2d = cache.view(pages, page_bytes)
    cache_2d[:, : PAGE_SIZE * HEAD_DIM].view(torch.float8_e4m3fn).copy_(
        quantized.reshape(pages, PAGE_SIZE * HEAD_DIM)
    )
    cache_2d[:, PAGE_SIZE * HEAD_DIM :].view(torch.float32).copy_(
        scale.reshape(pages, PAGE_SIZE)
    )
    return cache


def _oracle_mqa_samples(
    torch,
    q_fp8,
    cache,
    weights,
    c4_seq_lens,
    page_table,
):
    if q_fp8.ndim != 4 or q_fp8.shape[0] != 1:
        raise ValueError("oracle expects one request with Q [1,m,H,D]")
    m = q_fp8.shape[1]
    indices = _sample_valid_output_indices(torch, m, q_fp8.device, 256)
    rows = torch.div(indices, COMPRESSED_CONTEXT, rounding_mode="floor")
    logical = torch.remainder(indices, COMPRESSED_CONTEXT)
    valid = logical < c4_seq_lens[0, rows].long()
    page_columns = torch.div(logical, PAGE_SIZE, rounding_mode="floor")
    pages = page_table[0, page_columns].long()
    slots = torch.remainder(logical, PAGE_SIZE).long()

    page_bytes = PAGE_SIZE * (HEAD_DIM + 4)
    cache_2d = cache.view(cache.shape[0], page_bytes)
    cache_k = cache_2d[:, : PAGE_SIZE * HEAD_DIM].view(torch.float8_e4m3fn)
    columns = slots.unsqueeze(1) * HEAD_DIM + torch.arange(
        HEAD_DIM, device=slots.device, dtype=torch.long
    ).unsqueeze(0)
    k = cache_k[pages.unsqueeze(1), columns].float()
    cache_scale = cache_2d[:, PAGE_SIZE * HEAD_DIM :].view(torch.float32)
    scale = cache_scale[pages, slots]

    q = q_fp8[0, rows].float()
    per_head = (q * k.unsqueeze(1)).sum(dim=-1)
    per_head = torch.relu(per_head)
    scores = (per_head * weights[rows].float()).sum(dim=-1) * scale
    return torch.where(valid, scores, torch.zeros_like(scores))


def _observed_state(torch, cache, page_table, semantic_oracle, expected_cache):
    """Use strided page views so post-call mutations remain observable."""

    page_bytes = PAGE_SIZE * (HEAD_DIM + 4)
    cache_2d = cache.view(cache.shape[0], page_bytes)
    expected_2d = expected_cache.view(expected_cache.shape[0], page_bytes)
    return {
        # 16 pages spanning the complete cache, exact as raw bytes.
        "cache_sample_pages": cache_2d[::17],
        "expected_cache_sample_pages": expected_2d[::17],
        "block_table_head": page_table[:1],
        "block_table_tail": page_table[-1:],
        "semantic_oracle": semantic_oracle,
    }


def _check_available_memory(required_bytes, available_bytes):
    if required_bytes > int(available_bytes * 0.85):
        raise MemoryError(
            f"estimated canonical/clones/oracle allocation {required_bytes} "
            f"exceeds safe CUDA memory budget {available_bytes}"
        )


class DeepSeekV4ChunkedMegaMqaLogitsSpec:
    operator_id = OPERATOR_ID

    def cases(self):
        return (
            _formal_case(1024, 421),
            _formal_case(2048, 423),
            _formal_case(4096, 429),
        )

    def layout_contract(self, case):
        m, raw_prefix = _validate(case.symbols)
        pages = COMPRESSED_CONTEXT // PAGE_SIZE
        return {
            "m_prefill_queries": m,
            "request_count": 1,
            "raw_prefix": raw_prefix,
            "raw_context": RAW_CONTEXT,
            "alternative_history_hidden_bf16": [RAW_CONTEXT, HIDDEN_SIZE],
            "alternative_wkv_gate_bf16": [WKV_GATE_DIM, HIDDEN_SIZE],
            "alternative_recompute_coverage_raw": [0, RAW_CONTEXT],
            "baseline_path": "complete_physical_c4_kv_reuse",
            "q_fp8": [1, m, HEADS, HEAD_DIM],
            "c4_seq_lens": [1, m],
            "c4_seq_lens_values": (
                f"floor(({raw_prefix}+i+1)/4) for i in [0,{m})"
            ),
            "page_table": [1, pages],
            "page_table_sharing": "one physical page-table row for one request",
            "physical_cache": [pages, PAGE_SIZE, 1, HEAD_DIM + 4],
            "physical_page_bytes": PAGE_SIZE * (HEAD_DIM + 4),
            "cache_layout": "8192 FP8 K bytes followed by 256 FP32 scale bytes",
            "output": [m, COMPRESSED_CONTEXT],
            "relu": "per_head_before_weighted_head_reduction",
            "causal_visibility": "C4 length is floor(raw query length/4)",
            "workspace_lifecycle": "history producer is outside baseline timing",
        }

    def make_inputs(self, case, context):
        torch, deep_gemm = _runtime()
        m, raw_prefix = _validate(case.symbols)
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("full-reuse mega MQA requires a CUDA generator")

        pages = COMPRESSED_CONTEXT // PAGE_SIZE
        page_bytes = PAGE_SIZE * (HEAD_DIM + 4)
        history_bytes = RAW_CONTEXT * HIDDEN_SIZE * 2
        fixed_bytes = (
            WKV_GATE_DIM * HIDDEN_SIZE * 2
            + 8 * HEAD_DIM * 4
            + HEAD_DIM * 4
            + 2 * COMPRESSED_CONTEXT * 32 * 4
            + pages * page_bytes
        )
        per_case_bytes = (
            m * HEADS * HEAD_DIM
            + m * HEADS * 4
            + pages * 4
            + m * 4
        )
        output_bytes = m * COMPRESSED_CONTEXT * 4
        producer_workspace = (
            RAW_CONTEXT * WKV_GATE_DIM * 4
            + 2 * COMPRESSED_CONTEXT * 8 * HEAD_DIM * 4
            + 5 * COMPRESSED_CONTEXT * HEAD_DIM * 4
        )
        estimated_peak = (
            3 * (history_bytes + fixed_bytes + per_case_bytes)
            + 2 * output_bytes
            + producer_workspace
        )
        free, _ = torch.cuda.mem_get_info(device)
        _check_available_memory(estimated_peak, free)

        history_hidden = torch.randn(
            (RAW_CONTEXT, HIDDEN_SIZE),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        ).mul_(0.05)
        wkv_gate = torch.randn(
            (WKV_GATE_DIM, HIDDEN_SIZE),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        ).mul_(0.02)
        ape = torch.randn(
            (8, HEAD_DIM),
            device=device,
            dtype=torch.float32,
            generator=generator,
        ).mul_(0.05)
        norm_weight = torch.randn(
            (HEAD_DIM,),
            device=device,
            dtype=torch.float32,
            generator=generator,
        ).mul_(0.05).add_(1.0)

        rope_frequency = 1.0 / (
            COMPRESS_ROPE_THETA
            ** (
                torch.arange(0, 64, 2, device=device, dtype=torch.float32)
                / 64.0
            )
        )
        rope_positions = torch.arange(
            0,
            RAW_CONTEXT,
            COMPRESSION_RATIO,
            device=device,
            dtype=torch.float32,
        )
        rope_phase = rope_positions.unsqueeze(1) * rope_frequency.unsqueeze(0)
        rope_cos = torch.cos(rope_phase)
        rope_sin = torch.sin(rope_phase)

        # The complete physical cache is generated once, outside all timed
        # reference/candidate calls, from the equivalent upstream boundary.
        kv_fused = _oracle_full_history_cache(
            torch,
            history_hidden,
            wkv_gate,
            ape,
            norm_weight,
            rope_cos,
            rope_sin,
        )

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
            (m, HEADS),
            device=device,
            dtype=torch.float32,
            generator=generator,
        ).mul_((HEADS * HEAD_DIM) ** -0.5)

        request_page_table = torch.arange(
            pages, device=device, dtype=torch.int32
        )
        page_table = request_page_table.unsqueeze(0).contiguous()
        raw_causal_lens = raw_prefix + torch.arange(
            1, m + 1, device=device, dtype=torch.int32
        )
        c4_seq_lens = torch.div(
            raw_causal_lens, COMPRESSION_RATIO, rounding_mode="floor"
        ).unsqueeze(0)
        schedule = deep_gemm.get_paged_mqa_logits_metadata(
            c4_seq_lens, PAGE_SIZE, deep_gemm.get_num_sms()
        )
        semantic_oracle = _oracle_mqa_samples(
            torch,
            q_fp8,
            kv_fused,
            weights,
            c4_seq_lens,
            page_table,
        )
        expected_cache = kv_fused.clone()
        observed = _observed_state(
            torch, kv_fused, page_table, semantic_oracle, expected_cache
        )
        return InputBundle(
            args=(
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
                RAW_CONTEXT,
                0,
                RMS_EPS,
            ),
            observed_state=observed,
        )

    def clone_inputs(self, inputs):
        torch, deep_gemm = _runtime()
        (
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
            _,
            raw_context,
            recompute_eligible_raw_start,
            rms_eps,
        ) = inputs.args
        cloned = (
            history_hidden.clone(),
            wkv_gate.clone(),
            ape.clone(),
            norm_weight.clone(),
            rope_cos.clone(),
            rope_sin.clone(),
            q_fp8.clone(),
            weights.clone(),
            kv_fused.clone(),
            c4_seq_lens.clone(),
            page_table.clone(),
        )
        schedule = deep_gemm.get_paged_mqa_logits_metadata(
            cloned[9], PAGE_SIZE, deep_gemm.get_num_sms()
        )
        expected_cache = cloned[8].clone()
        semantic_oracle = _oracle_mqa_samples(
            torch,
            cloned[6],
            expected_cache,
            cloned[7],
            cloned[9],
            cloned[10],
        )
        observed = _observed_state(
            torch, cloned[8], cloned[10], semantic_oracle, expected_cache
        )
        return InputBundle(
            args=(
                *cloned,
                schedule,
                raw_context,
                recompute_eligible_raw_start,
                rms_eps,
            ),
            observed_state=observed,
        )

    def normalize_output(self, output):
        return _bounded_output(output)

    def comparator(self, case):
        _validate(case.symbols)
        return FullKvReuseComparator()

    def cost_model(self, case):
        m, _ = _validate(case.symbols)
        causal_c4_elements = sum(
            (RAW_CONTEXT - m + index + 1) // COMPRESSION_RATIO
            for index in range(m)
        )
        mqa_dot = 2 * HEADS * HEAD_DIM * causal_c4_elements
        mqa_head_reduce = 2 * HEADS * causal_c4_elements
        flops = mqa_dot + mqa_head_reduce
        estimated_bytes = (
            m * HEADS * HEAD_DIM
            + m * HEADS * 4
            + causal_c4_elements * (HEAD_DIM + 4)
            + (COMPRESSED_CONTEXT // PAGE_SIZE) * 4
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


SPEC = DeepSeekV4ChunkedMegaMqaLogitsSpec()


def cost_model(case):
    return SPEC.cost_model(case)

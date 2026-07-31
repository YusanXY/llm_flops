"""Reference-owned contract for DeepSeek V4 indexer FP8 quantization."""

import importlib
import math

from benchmark_engine.correctness import (
    FloatingComparator,
    InputBundle,
    QuantizationParameters,
    QuantizedComparator,
    Tolerance,
)
from benchmark_engine.correctness.models import ComparisonResult, OutputBundle, OutputLeaf
from benchmark_engine.models import CaseSpec


HEADS = 64
HEAD_DIM = 128
ROPE_DIM = 64
WEIGHT_SCALE = HEAD_DIM**-0.5 * HEADS**-0.5
WEIGHT_TOLERANCE = Tolerance(1e-5, 1e-6, source="fused_quant_weight")
EFFECTIVE_TOLERANCE = Tolerance(1e-4, 1e-5, source="fused_quant_effective")
LEGACY_INDEXER_FP8 = {
    "adapter_name": "C4 Indexer FP8 Quant",
    "backend": "SGLang fused RoPE/Hadamard FP8",
    "instances": 30,
    "prefill_m": (1024, 2048, 4096),
    "decode_m": (16, 32),
    "context": 65536,
    "heads": HEADS,
    "head_dim": HEAD_DIM,
    "weight_scale": WEIGHT_SCALE,
}


def _projection_cases():
    return tuple(CaseSpec(
        f"{phase}__c4_indexer_fp8_quant__m{m}__ctx65536__fp8_mxfp8",
        {"batch": 1 if phase == "prefill" else m,
         "query_tokens": m if phase == "prefill" else 1,
         "request_count": 1 if phase == "prefill" else m,
         "context": 65536, "position": 65535, "pattern": "random",
         "weight_layout": "contiguous", "phase": phase, "quant_profile": "fp8_mxfp8",
         "raw_context": 65536, "model_input": m, "projection_adapter_id": "c4_indexer_fp8_quant"},
        223, frozenset({"model_projection", f"deepseek_v4_{phase}", f"phase_{phase}",
                        "quant_profile_fp8_mxfp8", f"m_{m}", "context_65536", "performance_only"}), 1800)
        for phase, inputs in (("prefill", (1024, 2048, 4096)), ("decode", (16, 32))) for m in inputs)


def _only(bundle, paths):
    wanted = set(paths)
    return OutputBundle(tuple(leaf for leaf in bundle.leaves if leaf.path in wanted))


class IndexerQuantComparator:
    """Audit FP8 bytes, weight scale and their downstream effective values."""

    def compare(self, reference, candidate, **_):
        code_path = "output.codes"
        weight_path = "output.weights"
        effective_path = "output.effective"
        codes = QuantizedComparator(
            parameters=QuantizationParameters(
                scale=1.0,
                zero_point=0,
                quant_min=0,
                quant_max=255,
            ),
            mode="raw_code",
        ).compare(_only(reference, (code_path,)), _only(candidate, (code_path,)))
        weights = FloatingComparator(operator_default=WEIGHT_TOLERANCE, require_explicit=True).compare(
            _only(reference, (weight_path,)), _only(candidate, (weight_path,))
        )
        effective = FloatingComparator(operator_default=EFFECTIVE_TOLERANCE, require_explicit=True).compare(
            _only(reference, (effective_path,)), _only(candidate, (effective_path,))
        )
        passed = codes.passed and weights.passed and effective.passed
        return ComparisonResult(
            passed,
            "indexer_fp8_quantized",
            {"raw_codes": codes.metrics, "weights": weights.metrics, "effective_dequant": effective.metrics},
            tuple((codes.diagnostics + weights.diagnostics + effective.diagnostics)[:8]),
            None if passed else (codes.failed_path or weights.failed_path or effective.failed_path),
        )


def _torch():
    return importlib.import_module("torch")


def _freqs_cis(torch, context, device):
    frequencies = 1.0 / (
        10000.0 ** (torch.arange(0, ROPE_DIM, 2, device=device, dtype=torch.float32) / ROPE_DIM)
    )
    positions = torch.arange(context, device=device, dtype=torch.float32)
    phases = torch.outer(positions, frequencies)
    return torch.polar(torch.ones_like(phases), phases)


class DeepSeekV4IndexerFp8QuantSpec:
    operator_id = "deepseek_v4_indexer_fp8_quant"

    def cases(self):
        return (
            CaseSpec("smoke_b2", {"batch": 2, "context": 257, "position": 17, "pattern": "random", "weight_layout": "contiguous"}, 59, frozenset({"smoke"}), 600),
            CaseSpec("boundary_zero_clamp", {"batch": 1, "context": 2, "position": 0, "pattern": "zero", "weight_layout": "contiguous"}, 61, frozenset({"boundary", "adversarial"}), 600),
            CaseSpec("boundary_noncontiguous_weight", {"batch": 3, "context": 257, "position": 128, "pattern": "random", "weight_layout": "noncontiguous"}, 67, frozenset({"boundary"}), 600),
            CaseSpec("boundary_position_65535", {"batch": 1, "context": 65536, "position": 65535, "pattern": "random", "weight_layout": "contiguous"}, 71, frozenset({"boundary"}), 600),
            CaseSpec("representative_decode_b16_context65536", {"batch": 16, "context": 65536, "position": 65535, "pattern": "random", "weight_layout": "contiguous"}, 73, frozenset({"representative", "legacy", "decode"}), 1800),
        ) + _projection_cases()

    def legacy_mappings(self):
        return LEGACY_INDEXER_FP8

    @staticmethod
    def _validate(symbols):
        batch, context, position = (int(symbols[n]) for n in ("batch", "context", "position"))
        query_tokens = int(symbols.get("query_tokens", 1))
        if batch <= 0 or query_tokens <= 0 or context <= 0:
            raise ValueError("batch, query_tokens and context must be positive")
        if position < 0 or position >= context:
            raise ValueError("position must satisfy 0 <= position < context")
        if query_tokens > context:
            raise ValueError("query_tokens cannot exceed context")
        if symbols.get("weight_layout") not in {"contiguous", "noncontiguous"}:
            raise ValueError("unsupported weight layout")
        if symbols.get("phase") == "prefill" and batch != 1:
            raise ValueError("prefill CaseSpec must describe exactly one request")
        return batch, query_tokens, context, position

    def make_inputs(self, case, context):
        torch = _torch()
        batch, query_tokens, sequence, position = self._validate(case.symbols)
        rows = batch * query_tokens
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("deepseek_v4_indexer_fp8_quant requires a CUDA generator")
        q = torch.randn((rows, HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16, generator=generator)
        if case.symbols.get("pattern") == "zero":
            q.zero_()
        if case.symbols.get("weight_layout") == "noncontiguous":
            storage = torch.randn((rows, HEADS * 2), device=device, dtype=torch.bfloat16, generator=generator)
            weight = storage[:, ::2]
        else:
            weight = torch.randn((rows, HEADS), device=device, dtype=torch.bfloat16, generator=generator)
        if query_tokens > 1:
            positions = torch.arange(
                sequence - query_tokens,
                sequence,
                device=device,
                dtype=torch.int32,
            )
        else:
            positions = torch.full((rows,), position, device=device, dtype=torch.int32)
        freqs_cis = _freqs_cis(torch, sequence, device)
        return InputBundle(args=(q, weight, WEIGHT_SCALE, freqs_cis, positions))

    def clone_inputs(self, inputs):
        q, weight, weight_scale, freqs_cis, positions = inputs.args
        if weight.is_contiguous():
            weight_clone = weight.clone()
        else:
            storage = weight.new_empty((weight.shape[0], weight.shape[1] * 2))
            weight_clone = storage[:, ::2]
            weight_clone.copy_(weight)
        return InputBundle(args=(q.clone(), weight_clone, weight_scale, freqs_cis.clone(), positions.clone()))

    def normalize_output(self, output):
        q_fp8, weights = output
        torch = _torch()

        def leaf(path, tensor):
            flat = tensor.detach().reshape(-1)
            count = min(flat.numel(), 256)
            if count <= 1:
                indices = torch.zeros(count, device=flat.device, dtype=torch.long)
            else:
                positions = torch.arange(count, device=flat.device, dtype=torch.long)
                span, intervals = flat.numel() - 1, count - 1
                indices = positions * (span // intervals) + positions * (span % intervals) // intervals
            sampled = flat[indices].cpu().tolist()
            values = tuple(int(value) for value in sampled) if tensor.dtype == torch.uint8 else tuple(float(value) for value in sampled)
            return OutputLeaf(path, values, str(tensor.dtype),
                              tuple(tensor.shape), tuple(tensor.stride()), "strided", str(tensor.device))

        values, weights_float = q_fp8.float(), weights.float()
        return OutputBundle((leaf("output.codes", q_fp8.view(torch.uint8)),
                             leaf("output.values", values), leaf("output.weights", weights_float),
                             leaf("output.effective", values * weights_float)))

    def comparator(self, case):
        del case
        return IndexerQuantComparator()

    def cost_model(self, case):
        batch, query_tokens, _, _ = self._validate(case.symbols)
        rows = batch * query_tokens
        values = rows * HEADS * HEAD_DIM
        rope_flops = rows * HEADS * ROPE_DIM * 6
        hadamard_flops = values * int(math.log2(HEAD_DIM))
        return {"flops": rope_flops + hadamard_flops, "estimated_bytes": values * 3 + rows * HEADS * 4, "throughput_units": values}


SPEC = DeepSeekV4IndexerFp8QuantSpec()


def cost_model(case):
    return SPEC.cost_model(case)

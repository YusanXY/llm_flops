"""Contract for SGLang's fused WQ_A plus WKV block-FP8 linear on MI300X."""

import importlib

from benchmark_engine.correctness import InputBundle, clone_input_bundle
from benchmark_engine.models import CaseSpec
from benchmark_engine.workloads.deepseek_v4_flash import (
    NumericPath,
    OracleStateComparator,
    normalize_named_tensors,
    sample_tensor,
)


BLOCK = 128
K = 4096
N = 1536
PROFILE = "fp8_block_gfx942"
ADAPTER_ID = "fused_wq_a_wkv"


def _quantize_weight(torch, value):
    n, k = value.shape
    blocks = value.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    maximum = blocks.abs().amax(dim=(1, 3)).clamp_min(1.0e-8)
    scale = torch.exp2(torch.ceil(torch.log2(maximum / 240.0)))
    quantized = (blocks / scale[:, None, :, None]).clamp(
        -240.0, 240.0
    ).to(torch.float8_e4m3fnuz)
    return quantized.view(n, k), scale


def _dequantize_weight(torch, weight, scale):
    n, k = weight.shape
    return (
        weight.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
        * scale[:, None, :, None]
    ).view(n, k)


def _representative_cases():
    cases = []
    seed = 4710
    for m in (16, 32):
        seed += 1
        cases.append(
            CaseSpec(
                f"decode_fused_wq_a_wkv_m{m}_context65536",
                {
                    "phase": "decode",
                    "m": m,
                    "k": K,
                    "n": N,
                    "model_input": m,
                    "raw_context": 65536,
                    "quant_profile": PROFILE,
                    "projection_adapter_id": ADAPTER_ID,
                },
                seed,
                frozenset(
                    {
                        "representative",
                        "performance_only",
                        "deepseek_v4_flash_decode_unified",
                    }
                ),
                2400,
            )
        )
    return tuple(cases)


class DeepSeekV4AiterFp8LinearSpec:
    operator_id = "deepseek_v4_aiter_fp8_linear"

    def cases(self):
        return (
            CaseSpec(
                "decode_smoke_m16_context1024",
                {
                    "phase": "decode",
                    "m": 16,
                    "k": K,
                    "n": N,
                    "model_input": 16,
                    "raw_context": 1024,
                    "quant_profile": PROFILE,
                    "projection_adapter_id": ADAPTER_ID,
                },
                4701,
                frozenset({"smoke", "oracle", "deepseek_v4_decode"}),
                900,
            ),
        ) + _representative_cases()

    def make_inputs(self, case, context):
        torch = importlib.import_module("torch")
        m, k, n = (int(case.symbols[name]) for name in ("m", "k", "n"))
        if k % BLOCK or n % BLOCK:
            raise ValueError("K and N must be divisible by 128")
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("fused WQ_A plus WKV projection requires a GPU generator")
        x = torch.randn(
            (m, k), device=device, dtype=torch.bfloat16, generator=generator
        ).clamp_(-2.0, 2.0)
        source_weight = torch.randn(
            (n, k), device=device, dtype=torch.bfloat16, generator=generator
        ).clamp_(-0.25, 0.25)
        weight, scale = _quantize_weight(torch, source_weight)
        observed = {
            "weight_scale_head": scale[:1, : min(4, scale.shape[1])],
        }
        if "performance_only" not in case.tags:
            oracle = torch.mm(x.float(), _dequantize_weight(torch, weight, scale).t())
            observed["semantic_oracle"] = sample_tensor(oracle)
        return InputBundle(
            args=(x, weight, scale),
            observed_state=observed,
        )

    def clone_inputs(self, inputs):
        return clone_input_bundle(inputs)

    def normalize_output(self, output):
        return normalize_named_tensors(output)

    def comparator(self, case):
        return OracleStateComparator(
            numeric_paths=(
                NumericPath(
                    "output",
                    atol=1.0,
                    rtol=0.05,
                    oracle_path="state.semantic_oracle",
                ),
            ),
            immutable_paths=("state.weight_scale_head",),
            require_oracle="performance_only" not in case.tags,
            name="sglang_aiter_fused_wq_a_wkv_oracle",
        )

    def cost_model(self, case):
        m, k, n = (int(case.symbols[name]) for name in ("m", "k", "n"))
        return {
            "flops": 2 * m * n * k,
            "estimated_bytes": (
                2 * m * k
                + n * k
                + 4 * m * (k // BLOCK)
                + 4 * (n // BLOCK) * (k // BLOCK)
                + 2 * m * n
            ),
            "throughput_units": m * n,
        }

    def layout_contract(self, case):
        m, k, n = (int(case.symbols[name]) for name in ("m", "k", "n"))
        return {
            "x": [m, k],
            "weight": [n, k],
            "weight_scale": [n // BLOCK, k // BLOCK],
            "output": [m, n],
            "dtype": "bfloat16 x float8_e4m3fnuz -> bfloat16",
            "logical_operator": ADAPTER_ID,
            "backend": (
                "SGLang aiter_w8a8_block_fp8_linear: dynamic 1x128 "
                "activation quant + AITER Triton block-FP8 GEMM"
            ),
        }


SPEC = DeepSeekV4AiterFp8LinearSpec()


def cost_model(case):
    return SPEC.cost_model(case)

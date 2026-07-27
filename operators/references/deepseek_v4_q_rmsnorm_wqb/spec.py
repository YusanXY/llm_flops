"""Contract for the DeepSeek V4 Flash gfx942 Q RMSNorm + WQ_B path."""

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
K = 1024
N = 32768
EPSILON = 1.0e-6
PREFILL_M = (1024, 2048, 4096)
PROFILE = "fp8_block_gfx942"
UNIFIED_TAG = "deepseek_v4_flash_prefill_unified"


def _quantize_weight(torch, value):
    blocks = value.float().view(N // BLOCK, BLOCK, K // BLOCK, BLOCK)
    maximum = blocks.abs().amax(dim=(1, 3)).clamp_min(1.0e-8)
    scale = torch.exp2(torch.ceil(torch.log2(maximum / 240.0)))
    quantized = (blocks / scale[:, None, :, None]).clamp(
        -240.0, 240.0
    ).to(torch.float8_e4m3fnuz)
    return quantized.view(N, K), scale


def _dequantize_weight(torch, weight, scale):
    return (
        weight.float().view(N // BLOCK, BLOCK, K // BLOCK, BLOCK)
        * scale[:, None, :, None]
    ).view(N, K)


def _unified_cases():
    prefill = tuple(
        CaseSpec(
            f"prefill_q_rmsnorm_wqb_m{m}_context65536",
            {
                "phase": "prefill",
                "m": m,
                "k": K,
                "n": N,
                "model_input": m,
                "raw_context": 65536,
                "quant_profile": PROFILE,
                "projection_adapter_id": "q_rmsnorm_wq_b",
            },
            4800 + index,
            frozenset(
                {
                    "representative",
                    "performance_only",
                    UNIFIED_TAG,
                }
            ),
            2400,
        )
        for index, m in enumerate(PREFILL_M, start=1)
    )
    decode = tuple(
        CaseSpec(
            f"decode_q_rmsnorm_wqb_m{m}_context65536",
            {
                "phase": "decode",
                "m": m,
                "k": K,
                "n": N,
                "model_input": m,
                "raw_context": 65536,
                "quant_profile": PROFILE,
                "projection_adapter_id": "q_rmsnorm_wq_b",
            },
            4803 + index,
            frozenset(
                {
                    "representative",
                    "performance_only",
                    "deepseek_v4_flash_decode_unified",
                }
            ),
            2400,
        )
        for index, m in enumerate((16, 32), start=1)
    )
    return prefill + decode


class DeepSeekV4QRmsnormWqbSpec:
    operator_id = "deepseek_v4_q_rmsnorm_wqb"

    def cases(self):
        return (
            CaseSpec(
                "prefill_smoke_m8_k1024_n32768",
                {"phase": "prefill", "m": 8, "k": K, "n": N},
                4800,
                frozenset({"smoke", "oracle", "deepseek_v4_prefill"}),
                1200,
            ),
        ) + _unified_cases()

    def make_inputs(self, case, context):
        torch = importlib.import_module("torch")
        m = int(case.symbols["m"])
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("Q RMSNorm + WQ_B requires a GPU generator")
        q_lora = torch.randn(
            (m, K), device=device, dtype=torch.bfloat16, generator=generator
        ).clamp_(-2.0, 2.0)
        norm_weight = (
            torch.randn(
                (K,), device=device, dtype=torch.bfloat16, generator=generator
            )
            .mul_(0.05)
            .add_(1.0)
        )
        source_weight = torch.randn(
            (N, K), device=device, dtype=torch.bfloat16, generator=generator
        ).clamp_(-0.125, 0.125)
        weight, scale = _quantize_weight(torch, source_weight)
        observed = {
            "norm_weight_head": norm_weight[:32],
            "weight_scale_head": scale[:1, :4],
        }
        if "performance_only" not in case.tags:
            x = q_lora.float()
            normalized = (
                x
                * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + EPSILON)
                * norm_weight.float()
            )
            oracle = torch.mm(
                normalized, _dequantize_weight(torch, weight, scale).t()
            )
            observed["semantic_oracle"] = sample_tensor(oracle)
        return InputBundle(
            args=(q_lora, norm_weight, weight, scale, EPSILON),
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
                    atol=1.5,
                    rtol=0.08,
                    oracle_path="state.semantic_oracle",
                ),
            ),
            immutable_paths=("state.norm_weight_head", "state.weight_scale_head"),
            require_oracle="performance_only" not in case.tags,
            name="q_rmsnorm_wqb_oracle",
        )

    def cost_model(self, case):
        m = int(case.symbols["m"])
        rms_flops = 5 * m * K
        gemm_flops = 2 * m * K * N
        return {
            "flops": rms_flops + gemm_flops,
            "estimated_bytes": (
                2 * m * K
                + 2 * K
                + N * K
                + 4 * (N // BLOCK) * (K // BLOCK)
                + 4 * m * (K // BLOCK)
                + 2 * m * N
            ),
            "throughput_units": m * N,
        }

    def layout_contract(self, case):
        m = int(case.symbols["m"])
        return {
            "q_lora": [m, K],
            "norm_weight": [K],
            "wq_b": [N, K],
            "wq_b_scale": [N // BLOCK, K // BLOCK],
            "output": [m, N],
            "dtype": "bfloat16 RMSNorm + dynamic block-FP8 GEMM -> bfloat16",
            "backend": (
                "AITER rmsnorm2d_fwd + SGLang/AITER dynamic 1x128 quant "
                "+ AITER Triton block-FP8 GEMM"
            ),
        }


SPEC = DeepSeekV4QRmsnormWqbSpec()


def cost_model(case):
    return SPEC.cost_model(case)

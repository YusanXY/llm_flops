"""Contract for DeepSeek V4 Flash WO_A grouped BF16 projection."""

import importlib

from benchmark_engine.correctness import InputBundle, clone_input_bundle
from benchmark_engine.models import CaseSpec
from benchmark_engine.workloads.deepseek_v4_flash import (
    NumericPath,
    OracleStateComparator,
    normalize_named_tensors,
    sample_tensor,
)


GROUPS = 8
GROUP_DIM = 4096
RANK = 1024
PREFILL_M = (1024, 2048, 4096)
PROFILE = "fp8_block_gfx942"
UNIFIED_TAG = "deepseek_v4_flash_prefill_unified"


def _unified_cases():
    prefill = tuple(
        CaseSpec(
            f"prefill_wo_a_grouped_m{m}_context65536",
            {
                "phase": "prefill",
                "m": m,
                "groups": GROUPS,
                "group_dim": GROUP_DIM,
                "rank": RANK,
                "model_input": m,
                "raw_context": 65536,
                "quant_profile": PROFILE,
                "projection_adapter_id": "wo_a_grouped_projection",
            },
            4900 + index,
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
            f"decode_wo_a_grouped_m{m}_context65536",
            {
                "phase": "decode",
                "m": m,
                "groups": GROUPS,
                "group_dim": GROUP_DIM,
                "rank": RANK,
                "model_input": m,
                "raw_context": 65536,
                "quant_profile": PROFILE,
                "projection_adapter_id": "wo_a_grouped_projection",
            },
            4903 + index,
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


class DeepSeekV4WoAGroupedBf16Spec:
    operator_id = "deepseek_v4_wo_a_grouped_bf16"

    def cases(self):
        return (
            CaseSpec(
                "prefill_smoke_m4_g2_d128_r64",
                {
                    "phase": "prefill",
                    "m": 4,
                    "groups": 2,
                    "group_dim": 128,
                    "rank": 64,
                },
                4900,
                frozenset({"smoke", "oracle", "deepseek_v4_prefill"}),
                900,
            ),
        ) + _unified_cases()

    def make_inputs(self, case, context):
        torch = importlib.import_module("torch")
        m, groups, group_dim, rank = (
            int(case.symbols[name])
            for name in ("m", "groups", "group_dim", "rank")
        )
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("WO_A grouped BF16 requires a GPU generator")
        x = torch.randn(
            (m, groups, group_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).clamp_(-1.0, 1.0)
        weight = torch.randn(
            (groups, rank, group_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).mul_(0.02)
        observed = {"weight_head": weight[:1, :1, :32]}
        if "performance_only" not in case.tags:
            oracle = torch.einsum(
                "tgd,grd->tgr", x.float(), weight.float()
            )
            observed["semantic_oracle"] = sample_tensor(oracle)
        return InputBundle(args=(x, weight), observed_state=observed)

    def clone_inputs(self, inputs):
        return clone_input_bundle(inputs)

    def normalize_output(self, output):
        return normalize_named_tensors(output)

    def comparator(self, case):
        return OracleStateComparator(
            numeric_paths=(
                NumericPath(
                    "output",
                    atol=0.25,
                    rtol=0.02,
                    oracle_path="state.semantic_oracle",
                ),
            ),
            immutable_paths=("state.weight_head",),
            require_oracle="performance_only" not in case.tags,
            name="wo_a_grouped_bf16_oracle",
        )

    def cost_model(self, case):
        m, groups, group_dim, rank = (
            int(case.symbols[name])
            for name in ("m", "groups", "group_dim", "rank")
        )
        return {
            "flops": 2 * m * groups * group_dim * rank,
            "estimated_bytes": 2 * (
                m * groups * group_dim
                + groups * rank * group_dim
                + m * groups * rank
            ),
            "throughput_units": m * groups * rank,
        }

    def layout_contract(self, case):
        m, groups, group_dim, rank = (
            int(case.symbols[name])
            for name in ("m", "groups", "group_dim", "rank")
        )
        return {
            "attention_output": [m, groups, group_dim],
            "weight": [groups, rank, group_dim],
            "output": [m, groups, rank],
            "dtype": "bfloat16",
            "backend": 'PyTorch/rocBLAS torch.einsum("tgd,grd->tgr")',
        }


SPEC = DeepSeekV4WoAGroupedBf16Spec()


def cost_model(case):
    return SPEC.cost_model(case)

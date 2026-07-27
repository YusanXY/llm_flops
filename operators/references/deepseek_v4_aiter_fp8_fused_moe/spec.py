"""Contract for the DeepSeek V4 Flash AITER block-FP8 fused MoE."""

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
PREFILL_M = (1024, 2048, 4096)
PROFILE = "fp8_block_gfx942"
UNIFIED_TAG = "deepseek_v4_flash_prefill_unified"


def _unified_cases():
    prefill = tuple(
        CaseSpec(
            f"prefill_unified_m{tokens}_h4096_i2048_e256_top6_context65536",
            {
                "phase": "prefill",
                "tokens": tokens,
                "hidden": 4096,
                "intermediate": 2048,
                "experts": 256,
                "topk": 6,
                "model_input": tokens,
                "raw_context": 65536,
                "quant_profile": PROFILE,
                "projection_adapter_id": "routed_expert_fused_moe",
            },
            4650 + index,
            frozenset(
                {
                    "representative",
                    "performance_only",
                    UNIFIED_TAG,
                }
            ),
            3600,
        )
        for index, tokens in enumerate(PREFILL_M, start=1)
    )
    decode_m32 = CaseSpec(
        "decode_unified_m32_h4096_i2048_e256_top6_context65536",
        {
            "phase": "decode",
            "tokens": 32,
            "hidden": 4096,
            "intermediate": 2048,
            "experts": 256,
            "topk": 6,
            "model_input": 32,
            "raw_context": 65536,
            "quant_profile": PROFILE,
            "projection_adapter_id": "routed_expert_fused_moe",
        },
        4660,
        frozenset(
            {
                "representative",
                "performance_only",
                "deepseek_v4_flash_decode_unified",
            }
        ),
        3600,
    )
    return prefill + (decode_m32,)


def _quantize_weight(torch, value):
    experts, rows, columns = value.shape
    blocks = value.float().view(
        experts,
        rows // BLOCK,
        BLOCK,
        columns // BLOCK,
        BLOCK,
    )
    maximum = blocks.abs().amax(dim=(2, 4)).clamp_min(1.0e-8)
    scale_float = torch.exp2(torch.ceil(torch.log2(maximum / 240.0)))
    scale = scale_float
    quantized = (blocks / scale.float()[:, :, None, :, None]).clamp(
        -240.0, 240.0
    ).to(torch.float8_e4m3fnuz)
    return quantized.view(experts, rows, columns), scale


def _constant_weight(torch, shape, device):
    experts, rows, columns = shape
    weight = torch.full(
        shape, 0.5, dtype=torch.float8_e4m3fnuz, device=device
    )
    scale = torch.full(
        (experts, rows // BLOCK, columns // BLOCK),
        2.0**-7,
        dtype=torch.float32,
        device=device,
    )
    return weight, scale


def _dequantize_weight(torch, weight, scale):
    experts, rows, columns = weight.shape
    return (
        weight.float().view(
            experts,
            rows // BLOCK,
            BLOCK,
            columns // BLOCK,
            BLOCK,
        )
        * scale.float()[:, :, None, :, None]
    ).view(experts, rows, columns)


def _semantic_oracle(
    torch,
    hidden,
    w13,
    w2,
    topk_weights,
    topk_ids,
    w13_scale,
    w2_scale,
    intermediate,
):
    w13_value = _dequantize_weight(torch, w13, w13_scale)
    w2_value = _dequantize_weight(torch, w2, w2_scale)
    output = torch.zeros_like(hidden, dtype=torch.float32)
    for token in range(hidden.shape[0]):
        for route in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, route].item())
            projected = torch.mv(w13_value[expert], hidden[token].float())
            gate, up = projected.split(intermediate)
            gate = gate.clamp(max=10.0)
            up = up.clamp(min=-10.0, max=10.0)
            activated = torch.nn.functional.silu(gate) * up
            expert_output = torch.mv(w2_value[expert], activated)
            output[token] += topk_weights[token, route] * expert_output
    return output


def _triton_oracle(
    torch,
    hidden_states,
    w13,
    w2,
    topk_weights,
    topk_ids,
    w13_scale,
    w2_scale,
    experts,
    hidden,
    intermediate,
    topk,
):
    """Independent SGLang Triton oracle for representative AITER cases."""
    from types import SimpleNamespace

    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        fused_moe as triton_fused_moe,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.server_args import (
        get_global_server_args,
        set_global_server_args_for_scheduler,
    )

    # The fused clamp epilogue is CUDA-only. This setting does not affect AITER.
    envs.SGLANG_OPT_SWIGLU_CLAMP_FUSION.set(False)
    try:
        get_global_server_args()
    except ValueError:
        set_global_server_args_for_scheduler(
            SimpleNamespace(
                enable_deterministic_inference=False,
                enable_fused_moe_sum_all_reduce=False,
            )
        )
    config = MoeRunnerConfig(
        num_experts=experts,
        num_local_experts=experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        top_k=topk,
        activation="silu",
        is_gated=True,
        inplace=False,
        swiglu_limit=10.0,
    )
    output = triton_fused_moe(
        hidden_states=hidden_states,
        w1=w13,
        w2=w2,
        topk_output=StandardTopKOutput(topk_weights, topk_ids, None),
        moe_runner_config=config,
        use_fp8_w8a8=True,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=[128, 128],
    )
    torch.cuda.synchronize(hidden_states.device)
    return output


class DeepSeekV4AiterFp8FusedMoeSpec:
    operator_id = "deepseek_v4_aiter_fp8_fused_moe"

    def cases(self):
        return (
            CaseSpec(
                "prefill_smoke_m64_h256_i128_e8_top2",
                {
                    "phase": "prefill",
                    "tokens": 64,
                    "hidden": 256,
                    "intermediate": 128,
                    "experts": 8,
                    "topk": 2,
                },
                4601,
                frozenset({"smoke", "oracle", "deepseek_v4_prefill"}),
                1200,
            ),
            CaseSpec(
                "decode_smoke_m4_h256_i128_e8_top2",
                {
                    "phase": "decode",
                    "tokens": 4,
                    "hidden": 256,
                    "intermediate": 128,
                    "experts": 8,
                    "topk": 2,
                },
                4602,
                frozenset({"smoke", "oracle", "deepseek_v4_decode"}),
                1200,
            ),
            CaseSpec(
                "prefill_representative_m256_h4096_i2048_e256_top6",
                {
                    "phase": "prefill",
                    "tokens": 256,
                    "hidden": 4096,
                    "intermediate": 2048,
                    "experts": 256,
                    "topk": 6,
                },
                4603,
                frozenset({"representative", "oracle", "deepseek_v4_prefill"}),
                2400,
            ),
            CaseSpec(
                "decode_representative_m16_h4096_i2048_e256_top6",
                {
                    "phase": "decode",
                    "tokens": 16,
                    "hidden": 4096,
                    "intermediate": 2048,
                    "experts": 256,
                    "topk": 6,
                    "model_input": 16,
                    "raw_context": 65536,
                    "quant_profile": PROFILE,
                    "projection_adapter_id": "routed_expert_fused_moe",
                },
                4604,
                frozenset(
                    {
                        "representative",
                        "oracle",
                        "deepseek_v4_decode",
                        "deepseek_v4_flash_decode_unified",
                    }
                ),
                2400,
            ),
        ) + _unified_cases()

    def make_inputs(self, case, context):
        torch = importlib.import_module("torch")
        shuffle = importlib.import_module("aiter.ops.shuffle").shuffle_weight
        tokens, hidden, intermediate, experts, topk = (
            int(case.symbols[name])
            for name in ("tokens", "hidden", "intermediate", "experts", "topk")
        )
        if hidden % BLOCK or intermediate % BLOCK:
            raise ValueError("hidden and intermediate sizes must be 128-aligned")
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("AITER fused MoE requires a GPU generator")
        hidden_states = torch.randn(
            (tokens, hidden),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).clamp_(-1.0, 1.0)
        requires_oracle = "performance_only" not in case.tags
        use_small_oracle = requires_oracle and "representative" not in case.tags
        if use_small_oracle:
            w13_source = torch.randn(
                (experts, 2 * intermediate, hidden),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.02)
            w2_source = torch.randn(
                (experts, hidden, intermediate),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).mul_(0.02)
            w13, w13_scale = _quantize_weight(torch, w13_source)
            w2, w2_scale = _quantize_weight(torch, w2_source)
            oracle_w13, oracle_w2 = w13, w2
        else:
            w13, w13_scale = _constant_weight(
                torch, (experts, 2 * intermediate, hidden), device
            )
            w2, w2_scale = _constant_weight(
                torch, (experts, hidden, intermediate), device
            )
            oracle_w13 = oracle_w2 = None
        # Routing itself is outside the fused-MoE operator. Use a deterministic
        # balanced dispatch so this kernel benchmark measures expert compute
        # rather than a random expert-load tail. Keep DeepSeek V4's
        # sqrt-softplus, normalization, and routed_scaling_factor=1.5 semantics
        # for the per-route weights.
        router_logits = torch.randn(
            (tokens, topk),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        selected_scores = torch.sqrt(
            torch.nn.functional.softplus(router_logits)
        )
        topk_ids = (
            torch.arange(
                tokens * topk, dtype=torch.int32, device=device
            )
            .view(tokens, topk)
            .remainder(experts)
        )
        topk_weights = (
            selected_scores / selected_scores.sum(dim=-1, keepdim=True)
        ).mul_(1.5)
        observed = {
            "topk_ids": topk_ids,
            "topk_weights": topk_weights,
            "w13_scale_head": w13_scale[:1, :1, : min(4, w13_scale.shape[-1])],
            "w2_scale_head": w2_scale[:1, :1, : min(4, w2_scale.shape[-1])],
        }
        if use_small_oracle:
            oracle = _semantic_oracle(
                torch,
                hidden_states,
                oracle_w13,
                oracle_w2,
                topk_weights,
                topk_ids,
                w13_scale,
                w2_scale,
                intermediate,
            )
            observed["semantic_oracle"] = sample_tensor(oracle)
        elif requires_oracle:
            oracle = _triton_oracle(
                torch,
                hidden_states,
                w13,
                w2,
                topk_weights,
                topk_ids,
                w13_scale,
                w2_scale,
                experts,
                hidden,
                intermediate,
                topk,
            )
            observed["semantic_oracle"] = sample_tensor(oracle)
        w13 = shuffle(w13.contiguous(), (16, 16))
        w2 = shuffle(w2.contiguous(), (16, 16))
        observed["w13_scale_head"] = w13_scale[
            :1, :1, : min(4, w13_scale.shape[-1])
        ]
        observed["w2_scale_head"] = w2_scale[
            :1, :1, : min(4, w2_scale.shape[-1])
        ]
        return InputBundle(
            args=(
                hidden_states,
                w13,
                w2,
                topk_weights,
                topk_ids,
                w13_scale,
                w2_scale,
            ),
            observed_state=observed,
        )

    def clone_inputs(self, inputs):
        cloned = clone_input_bundle(inputs)
        # Tensor.clone() intentionally drops Python-side attributes. AITER uses
        # this marker to select the kernel contract for preshuffled expert
        # weights, so restore it on every independent benchmark input bundle.
        cloned.args[1].is_shuffled = True
        cloned.args[2].is_shuffled = True
        return cloned

    def normalize_output(self, output):
        return normalize_named_tensors(output)

    def comparator(self, case):
        return OracleStateComparator(
            numeric_paths=(
                NumericPath(
                    "output", 0.2, 0.15, "state.semantic_oracle"
                ),
            ),
            immutable_paths=(
                "state.topk_ids",
                "state.topk_weights",
                "state.w13_scale_head",
                "state.w2_scale_head",
            ),
            require_oracle="performance_only" not in case.tags,
            name="aiter_fp8_fused_moe_oracle",
        )

    def cost_model(self, case):
        tokens, hidden, intermediate, topk = (
            int(case.symbols[name])
            for name in ("tokens", "hidden", "intermediate", "topk")
        )
        return {
            "flops": 6 * tokens * topk * hidden * intermediate,
            "estimated_bytes": (
                tokens * hidden * 2
                + int(case.symbols["experts"])
                * (3 * hidden * intermediate)
                + tokens * hidden * 2
            ),
            "throughput_units": tokens * topk,
        }

    def layout_contract(self, case):
        return {
            "hidden_states": [
                int(case.symbols["tokens"]),
                int(case.symbols["hidden"]),
            ],
            "w13": [
                int(case.symbols["experts"]),
                2 * int(case.symbols["intermediate"]),
                int(case.symbols["hidden"]),
            ],
            "w2": [
                int(case.symbols["experts"]),
                int(case.symbols["hidden"]),
                int(case.symbols["intermediate"]),
            ],
            "topk": int(case.symbols["topk"]),
            "weight_dtype": "float8_e4m3fnuz",
            "scale_dtype": "float32 block inverse scale",
            "swiglu_limit": 10.0,
            "backend": "direct AITER block-FP8 fused MoE used by SGLang",
            "routing": "ungrouped sqrtsoftplus, normalized top6, scale=1.5",
            "oracle": "independent SGLang Triton block-FP8 fused MoE",
        }


SPEC = DeepSeekV4AiterFp8FusedMoeSpec()


def cost_model(case):
    return SPEC.cost_model(case)

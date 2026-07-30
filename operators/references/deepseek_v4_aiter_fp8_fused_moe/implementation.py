"""AITER FP8 fused-MoE kernel used by SGLang's DeepSeek V4 AMD path."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _backend_symbols():
    """Import AITER after the worker has initialized its selected HIP device."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    return fused_moe, ActivationType, QuantType, GateMode


def operator(
    hidden_states,
    w13,
    w2,
    topk_weights,
    topk_ids,
    w13_scale,
    w2_scale,
    expert_mask=None,
):
    """Run only the fused-MoE backend selected by SGLang's AITER runner.

    All inputs are prepared by the operator spec before timing. The arguments
    mirror ``AiterRunnerCore.run`` for the standard-dispatch block-FP8
    DeepSeek V4 path, without constructing SGLang runner, config, dispatch, or
    quant-info objects inside the measured call. ``expert_mask`` is optional
    so the llm_flops EP1 benchmark ABI remains unchanged while SGLang EP4/EP8
    can pass its global-to-local expert ownership mask to AITER.
    """
    fused_moe, activation_type, quant_type, gate_mode = _backend_symbols()
    return fused_moe(
        hidden_states=hidden_states,
        w1=w13,
        w2=w2,
        topk_weight=topk_weights,
        topk_ids=topk_ids,
        activation=activation_type.Silu,
        quant_type=quant_type.per_128x128,
        doweight_stage1=False,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        expert_mask=expert_mask,
        swiglu_limit=10.0,
        gate_mode=gate_mode.INTERLEAVE.value,
    )

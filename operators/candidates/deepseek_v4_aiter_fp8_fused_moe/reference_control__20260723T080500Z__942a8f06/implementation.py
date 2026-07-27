"""Control candidate for the DeepSeek V4 AITER FP8 fused-MoE kernel."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _backend_symbols():
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
):
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
        swiglu_limit=10.0,
        gate_mode=gate_mode.INTERLEAVE.value,
    )

"""Control candidate matching the current gfx942 Q RMSNorm + WQ_B path."""

from aiter import rmsnorm2d_fwd
from sglang.srt.layers.quantization.fp8_utils import (
    aiter_w8a8_block_fp8_linear,
)


def operator(q_lora, norm_weight, weight_fp8, weight_scale, epsilon):
    normalized = rmsnorm2d_fwd(q_lora, norm_weight, epsilon)
    return aiter_w8a8_block_fp8_linear(
        input=normalized,
        weight=weight_fp8,
        block_size=[128, 128],
        weight_scale=weight_scale,
        input_scale=None,
        bias=None,
    )

"""Control candidate matching the current SGLang gfx942 linear path."""

from sglang.srt.layers.quantization.fp8_utils import (
    aiter_w8a8_block_fp8_linear,
)


def operator(x, weight_fp8, weight_scale):
    return aiter_w8a8_block_fp8_linear(
        input=x,
        weight=weight_fp8,
        block_size=[128, 128],
        weight_scale=weight_scale,
        input_scale=None,
        bias=None,
    )

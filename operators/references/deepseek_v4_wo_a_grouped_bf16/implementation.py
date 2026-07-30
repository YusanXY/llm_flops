"""Current SGLang gfx942 WO_A grouped BF16 projection."""

import torch


def operator(attention_output, weight):
    return torch.einsum("tgd,grd->tgr", attention_output, weight)

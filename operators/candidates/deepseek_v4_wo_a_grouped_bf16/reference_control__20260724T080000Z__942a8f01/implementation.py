"""Control candidate matching the current gfx942 WO_A projection."""

import torch


def operator(attention_output, weight):
    return torch.einsum("tgd,grd->tgr", attention_output, weight)

import torch
import torch.nn as nn

from torch.autograd.function import once_differentiable
from core.nn import _C

__all__ = ['CollectAttention', 'DistributeAttention', 'psa_collect', 'psa_distribute']


class _PSACollect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hc):
        out = _C.psa_forward(hc, 1)

        ctx.save_for_backward(hc)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        hc = ctx.saved_tensors

        dhc = _C.psa_backward(dout, hc[0], 1)

        return dhc


class _PSADistribute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hc):
        out = _C.psa_forward(hc, 2)

        ctx.save_for_backward(hc)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        hc = ctx.saved_tensors

        dhc = _C.psa_backward(dout, hc[0], 2)

        return dhc


psa_collect = _PSACollect.apply
psa_distribute = _PSADistribute.apply


class CollectAttention(nn.Module):
    """Collect Attention Generation Module"""

    def __init__(self):
        super(CollectAttention, self).__init__()

    def forward(self, x):
        out = psa_collect(x)
        return out


class DistributeAttention(nn.Module):
    """Distribute Attention Generation Module"""

    def __init__(self):
        super(DistributeAttention, self).__init__()

    def forward(self, x):
        out = psa_distribute(x)
        return out

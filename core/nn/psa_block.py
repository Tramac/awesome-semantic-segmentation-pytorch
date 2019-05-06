import torch
import torch.nn as nn

from torch.autograd.function import once_differentiable
from core.nn import _C

__all__ = ['CollectAttention', 'DistributeAttention', 'psa']


class _PointwiseSpatialAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, forward_type):
        out = _C.psa_forward(h, forward_type)

        ctx.save_for_backward(h)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout, forward_type):
        h = ctx.save_for_backward

        dh = _C.psa_backward(dout, h, forward_type)

        return dh


psa = _PointwiseSpatialAttention.apply


class CollectAttention(nn.Module):
    """Collect Attention Generation Module"""

    def __init__(self):
        super(CollectAttention, self).__init__()

    def forward(self, x):
        out = psa(x, 1)
        return out


class DistributeAttention(nn.Module):
    """Distribute Attention Generation Module"""

    def __init__(self):
        super(DistributeAttention, self).__init__()

    def forward(self, x):
        out = psa(x, 2)
        return out

"""Point-wise Spatial Attention Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.segbase import SegBaseModel


class PSANet(SegBaseModel):
    r"""PSANet

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Hengshuang Zhao, et al. "PSANet: Point-wise Spatial Attention Network for Scene Parsing."
        ECCV-2018.
    """
    def __init__(self, nclass, backbone='resnet', aux=False, pretrained_base=True, **kwargs):
        super(PSANet, self).__init__(nclass, aux, backbone, pretrained_base, **kwargs)
        pass

    def forward(self, x):
        _, _, c3, c4 = self.base_forward(x)


class _PSAHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_PSAHead, self).__init__()
        pass

    def forward(self, x):
        pass

class _PSACollectModule(nn.Module):
    def __init__(self, in_channels, reduced_channels, feat_w, feat_h, norm_layer, **kwargs):
        super(_PSACollectModule, self).__init__()
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            norm_layer(reduced_channels),
            nn.ReLU(True))
        self.conv_adaption = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, 1, bias=False),
            norm_layer(reduced_channels),
            nn.ReLU(True),
            nn.Conv2d(reduced_channels, (feat_w - 1) * (feat_h), 1, bias=False))

    def forward(self, x):
        size = x.size()[2:]
        x = self.conv_reduce(x)
        # shrink
        x_shrink = F.interpolate(x, scale_factor=1 / 2, mode='bilinear', align_corners=True)
        x_adaption = self.conv_adaption(x_shrink)


class _PSADistributeModule(nn.Module):
    def __init__(self, in_channels, reduced_channels, feat_w, feat_h, norm_layer, **kwargs):
        super(_PSADistributeModule, self).__init__()
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            norm_layer(reduced_channels),
            nn.ReLU(True))
        self.conv_adaption = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, 1, bias=False),
            norm_layer(reduced_channels),
            nn.ReLU(True),
            nn.Conv2d(reduced_channels, (feat_w - 1) * (feat_h), 1, bias=False))

    def forward(self, x):
        x = self.conv_reduce(x)
        x_shrink = F.interpolate(x, scale_factor=1 / 2, mode='bilinear', align_corners=True)
        x_adaption = self.conv_adaption(x_shrink)
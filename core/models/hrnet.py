"""High-Resolution Representations for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class HRNet(nn.Module):
    """HRNet

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
        Ke Sun. "High-Resolution Representations for Labeling Pixels and Regions."
        arXiv preprint arXiv:1904.04514 (2019).
    """
    def __init__(self, nclass, backbone='', aux=False, pretrained_base=False, **kwargs):
        super(HRNet, self).__init__()

    def forward(self, x):
        pass
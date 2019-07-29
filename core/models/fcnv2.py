"""Fully Convolutional Network with Stride of 8"""
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel

__all__ = ['FCN', 'get_fcn', 'get_fcn_resnet50_voc',
           'get_fcn_resnet101_voc', 'get_fcn_resnet152_voc']


class FCN(SegBaseModel):
    def __init__(self, nclass, backbone='resnet50', aux=True, pretrained_base=True, **kwargs):
        super(FCN, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)
        self.head = _FCNHead(2048, nclass, **kwargs)
        if aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

    def forward(self, x):
        size = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


def get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='~/.torch/models',
            pretrained_base=True, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from ..data.dataloader import datasets
    model = FCN(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(kwargs['local_rank'])
        model.load_state_dict(torch.load(get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root),
                              map_location=device))
    return model


def get_fcn_resnet50_voc(**kwargs):
    return get_fcn('pascal_voc', 'resnet50', **kwargs)


def get_fcn_resnet101_voc(**kwargs):
    return get_fcn('pascal_voc', 'resnet101', **kwargs)


def get_fcn_resnet152_voc(**kwargs):
    return get_fcn('pascal_voc', 'resnet152', **kwargs)

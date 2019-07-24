"""Criss-Cross Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.nn import CrissCrossAttention
from .segbase import SegBaseModel
from .fcn import _FCNHead

__all__ = ['CCNet', 'get_ccnet', 'get_ccnet_resnet50_citys', 'get_ccnet_resnet101_citys',
           'get_ccnet_resnet152_citys', 'get_ccnet_resnet50_ade', 'get_ccnet_resnet101_ade',
           'get_ccnet_resnet152_ade']


class CCNet(SegBaseModel):
    r"""CCNet

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
        Zilong Huang, et al. "CCNet: Criss-Cross Attention for Semantic Segmentation."
        arXiv preprint arXiv:1811.11721 (2018).
    """

    def __init__(self, nclass, backbone='resnet50', aux=False, pretrained_base=True, **kwargs):
        super(CCNet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)
        self.head = _CCHead(nclass, **kwargs)
        if aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

    def forward(self, x):
        size = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)
        outputs = list()
        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _CCHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_CCHead, self).__init__()
        self.rcca = _RCCAModule(2048, 512, norm_layer, **kwargs)
        self.out = nn.Conv2d(512, nclass, 1)

    def forward(self, x):
        x = self.rcca(x)
        x = self.out(x)
        return x


class _RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, **kwargs):
        super(_RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.Dropout2d(0.1))

    def forward(self, x, recurrence=1):
        out = self.conva(x)
        for i in range(recurrence):
            out = self.cca(out)
        out = self.convb(out)
        out = torch.cat([x, out], dim=1)
        out = self.bottleneck(out)

        return out


def get_ccnet(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='~/.torch/models',
              pretrained_base=True, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from ..data.dataloader import datasets
    model = CCNet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(kwargs['local_rank'])
        model.load_state_dict(torch.load(get_model_file('ccnet_%s_%s' % (backbone, acronyms[dataset]), root=root),
                              map_location=device))
    return model


def get_ccnet_resnet50_citys(**kwargs):
    return get_ccnet('citys', 'resnet50', **kwargs)


def get_ccnet_resnet101_citys(**kwargs):
    return get_ccnet('citys', 'resnet101', **kwargs)


def get_ccnet_resnet152_citys(**kwargs):
    return get_ccnet('citys', 'resnet152', **kwargs)


def get_ccnet_resnet50_ade(**kwargs):
    return get_ccnet('ade20k', 'resnet50', **kwargs)


def get_ccnet_resnet101_ade(**kwargs):
    return get_ccnet('ade20k', 'resnet101', **kwargs)


def get_ccnet_resnet152_ade(**kwargs):
    return get_ccnet('ade20k', 'resnet152', **kwargs)


if __name__ == '__main__':
    model = get_ccnet_resnet50_citys()
    img = torch.randn(1, 3, 480, 480)
    outputs = model(img)

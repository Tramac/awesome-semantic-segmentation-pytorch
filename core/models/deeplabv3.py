"""Pyramid Scene Parsing Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from .fcn import _FCNHead

__all__ = ['DeepLabV3', 'get_deeplabv3', 'get_deeplabv3_resnet50_voc', 'get_deeplabv3_resnet101_voc',
           'get_deeplabv3_resnet152_voc', 'get_deeplabv3_resnet50_ade', 'get_deeplabv3_resnet101_ade',
           'get_deeplabv3_resnet152_ade']


class DeepLabV3(SegBaseModel):
    r"""DeepLabV3

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
        Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation."
        arXiv preprint arXiv:1706.05587 (2017).
    """

    def __init__(self, nclass, backbone='resnet50', aux=False, pretrained_base=True, **kwargs):
        super(DeepLabV3, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)
        self.head = _DeepLabHead(nclass, **kwargs)
        if self.aux:
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


class _DeepLabHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(2048, [12, 24, 36], norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_layer(256, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(256, nclass, 1)
        )

    def forward(self, x):
        x = self.aspp(x)
        return self.block(x)


class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs, **kwargs):
        super(_ASPP, self).__init__()
        out_channels = 256
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x


def get_deeplabv3(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='~/.torch/models',
                  pretrained_base=True, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from ..data.dataloader import datasets
    model = DeepLabV3(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(kwargs['local_rank'])
        model.load_state_dict(torch.load(get_model_file('deeplabv3_%s_%s' % (backbone, acronyms[dataset]), root=root),
                              map_location=device))
    return model


def get_deeplabv3_resnet50_voc(**kwargs):
    return get_deeplabv3('pascal_voc', 'resnet50', **kwargs)


def get_deeplabv3_resnet101_voc(**kwargs):
    return get_deeplabv3('pascal_voc', 'resnet101', **kwargs)


def get_deeplabv3_resnet152_voc(**kwargs):
    return get_deeplabv3('pascal_voc', 'resnet152', **kwargs)


def get_deeplabv3_resnet50_ade(**kwargs):
    return get_deeplabv3('ade20k', 'resnet50', **kwargs)


def get_deeplabv3_resnet101_ade(**kwargs):
    return get_deeplabv3('ade20k', 'resnet101', **kwargs)


def get_deeplabv3_resnet152_ade(**kwargs):
    return get_deeplabv3('ade20k', 'resnet152', **kwargs)


if __name__ == '__main__':
    model = get_deeplabv3_resnet50_voc()
    img = torch.randn(2, 3, 480, 480)
    output = model(img)

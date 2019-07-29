"""Point-wise Spatial Attention Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.nn import CollectAttention, DistributeAttention
from .segbase import SegBaseModel
from .fcn import _FCNHead

__all__ = ['PSANet', 'get_psanet', 'get_psanet_resnet50_voc', 'get_psanet_resnet101_voc',
           'get_psanet_resnet152_voc', 'get_psanet_resnet50_citys', 'get_psanet_resnet101_citys',
           'get_psanet_resnet152_citys']


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
        self.head = _PSAHead(nclass, **kwargs)
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


class _PSAHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_PSAHead, self).__init__()
        self.collect = _CollectModule(2048, 512, 60, 60, norm_layer, **kwargs)
        self.distribute = _DistributeModule(2048, 512, 60, 60, norm_layer, **kwargs)

        self.conv_post = nn.Sequential(
            nn.Conv2d(1024, 2048, 1, bias=False),
            norm_layer(2048),
            nn.ReLU(True))
        self.project = nn.Sequential(
            nn.Conv2d(4096, 512, 3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(True),
            nn.Conv2d(512, nclass, 1)
        )

    def forward(self, x):
        global_feature_collect = self.collect(x)
        global_feature_distribute = self.distribute(x)

        global_feature = torch.cat([global_feature_collect, global_feature_distribute], dim=1)
        out = self.conv_post(global_feature)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = torch.cat([x, out], dim=1)
        out = self.project(out)

        return out


class _CollectModule(nn.Module):
    def __init__(self, in_channels, reduced_channels, feat_w, feat_h, norm_layer, **kwargs):
        super(_CollectModule, self).__init__()
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            norm_layer(reduced_channels),
            nn.ReLU(True))
        self.conv_adaption = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, 1, bias=False),
            norm_layer(reduced_channels),
            nn.ReLU(True),
            nn.Conv2d(reduced_channels, (feat_w - 1) * (feat_h), 1, bias=False))
        self.collect_attention = CollectAttention()

        self.reduced_channels = reduced_channels
        self.feat_w = feat_w
        self.feat_h = feat_h

    def forward(self, x):
        x = self.conv_reduce(x)
        # shrink
        x_shrink = F.interpolate(x, scale_factor=1 / 2, mode='bilinear', align_corners=True)
        x_adaption = self.conv_adaption(x_shrink)
        ca = self.collect_attention(x_adaption)
        global_feature_collect_list = list()
        for i in range(x_shrink.shape[0]):
            x_shrink_i = x_shrink[i].view(self.reduced_channels, -1)
            ca_i = ca[i].view(ca.shape[1], -1)
            global_feature_collect_list.append(
                torch.mm(x_shrink_i, ca_i).view(1, self.reduced_channels, self.feat_h // 2, self.feat_w // 2))
        global_feature_collect = torch.cat(global_feature_collect_list)

        return global_feature_collect


class _DistributeModule(nn.Module):
    def __init__(self, in_channels, reduced_channels, feat_w, feat_h, norm_layer, **kwargs):
        super(_DistributeModule, self).__init__()
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            norm_layer(reduced_channels),
            nn.ReLU(True))
        self.conv_adaption = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, 1, bias=False),
            norm_layer(reduced_channels),
            nn.ReLU(True),
            nn.Conv2d(reduced_channels, (feat_w - 1) * (feat_h), 1, bias=False))
        self.distribute_attention = DistributeAttention()

        self.reduced_channels = reduced_channels
        self.feat_w = feat_w
        self.feat_h = feat_h

    def forward(self, x):
        x = self.conv_reduce(x)
        x_shrink = F.interpolate(x, scale_factor=1 / 2, mode='bilinear', align_corners=True)
        x_adaption = self.conv_adaption(x_shrink)
        da = self.distribute_attention(x_adaption)
        global_feature_distribute_list = list()
        for i in range(x_shrink.shape[0]):
            x_shrink_i = x_shrink[i].view(self.reduced_channels, -1)
            da_i = da[i].view(da.shape[1], -1)
            global_feature_distribute_list.append(
                torch.mm(x_shrink_i, da_i).view(1, self.reduced_channels, self.feat_h // 2, self.feat_w // 2))
        global_feature_distribute = torch.cat(global_feature_distribute_list)

        return global_feature_distribute


def get_psanet(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='~/.torch/models',
               pretrained_base=True, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from ..data.dataloader import datasets
    model = PSANet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(kwargs['local_rank'])
        model.load_state_dict(torch.load(get_model_file('deeplabv3_%s_%s' % (backbone, acronyms[dataset]), root=root),
                              map_location=device))
    return model


def get_psanet_resnet50_voc(**kwargs):
    return get_psanet('pascal_voc', 'resnet50', **kwargs)


def get_psanet_resnet101_voc(**kwargs):
    return get_psanet('pascal_voc', 'resnet101', **kwargs)


def get_psanet_resnet152_voc(**kwargs):
    return get_psanet('pascal_voc', 'resnet152', **kwargs)


def get_psanet_resnet50_citys(**kwargs):
    return get_psanet('citys', 'resnet50', **kwargs)


def get_psanet_resnet101_citys(**kwargs):
    return get_psanet('citys', 'resnet101', **kwargs)


def get_psanet_resnet152_citys(**kwargs):
    return get_psanet('citys', 'resnet152', **kwargs)


if __name__ == '__main__':
    model = get_psanet_resnet50_voc()
    img = torch.randn(1, 3, 480, 480)
    output = model(img)

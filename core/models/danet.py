"""Dual Attention Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel

__all__ = ['DANet', 'get_danet', 'get_danet_resnet50_citys',
           'get_danet_resnet101_citys', 'get_danet_resnet152_citys']


class DANet(SegBaseModel):
    r"""Pyramid Scene Parsing Network

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.
    Reference:
        Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang,and Hanqing Lu.
        "Dual Attention Network for Scene Segmentation." *CVPR*, 2019
    """

    def __init__(self, nclass, backbone='resnet50', aux=True, pretrained_base=True, **kwargs):
        super(DANet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)
        self.head = _DAHead(2048, nclass, aux, **kwargs)

        self.__setattr__('exclusive', ['head'])

    def forward(self, x):
        size = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c4)
        x0 = F.interpolate(x[0], size, mode='bilinear', align_corners=True)
        outputs.append(x0)

        if self.aux:
            x1 = F.interpolate(x[1], size, mode='bilinear', align_corners=True)
            x2 = F.interpolate(x[2], size, mode='bilinear', align_corners=True)
            outputs.append(x1)
            outputs.append(x2)
        return outputs


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class _DAHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)

        return tuple(outputs)


def get_danet(dataset='citys', backbone='resnet50', pretrained=False,
              root='~/.torch/models', pretrained_base=True, **kwargs):
    r"""Dual Attention Network

    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    pretrained_base : bool or str, default True
        This will load pretrained backbone network, that was trained on ImageNet.
    Examples
    --------
    >>> model = get_danet(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from ..data.dataloader import datasets
    model = DANet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(kwargs['local_rank'])
        model.load_state_dict(torch.load(get_model_file('danet_%s_%s' % (backbone, acronyms[dataset]), root=root),
                              map_location=device))
    return model


def get_danet_resnet50_citys(**kwargs):
    return get_danet('citys', 'resnet50', **kwargs)


def get_danet_resnet101_citys(**kwargs):
    return get_danet('citys', 'resnet101', **kwargs)


def get_danet_resnet152_citys(**kwargs):
    return get_danet('citys', 'resnet152', **kwargs)


if __name__ == '__main__':
    img = torch.randn(2, 3, 480, 480)
    model = get_danet_resnet50_citys()
    outputs = model(img)

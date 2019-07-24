"""LEDNet: A Lightweight Encoder-Decoder Network for Real-time Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.nn import _ConvBNReLU

__all__ = ['LEDNet', 'get_lednet', 'get_lednet_citys']

class LEDNet(nn.Module):
    r"""LEDNet

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
        Yu Wang, et al. "LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation."
        arXiv preprint arXiv:1905.02423 (2019).
    """

    def __init__(self, nclass, backbone='', aux=False, jpu=False, pretrained_base=True, **kwargs):
        super(LEDNet, self).__init__()
        self.encoder = nn.Sequential(
            Downsampling(3, 32),
            SSnbt(32, **kwargs), SSnbt(32, **kwargs), SSnbt(32, **kwargs),
            Downsampling(32, 64),
            SSnbt(64, **kwargs), SSnbt(64, **kwargs),
            Downsampling(64, 128),
            SSnbt(128, **kwargs),
            SSnbt(128, 2, **kwargs),
            SSnbt(128, 5, **kwargs),
            SSnbt(128, 9, **kwargs),
            SSnbt(128, 2, **kwargs),
            SSnbt(128, 5, **kwargs),
            SSnbt(128, 9, **kwargs),
            SSnbt(128, 17, **kwargs),
        )
        self.decoder = APNModule(128, nclass)

        self.__setattr__('exclusive', ['encoder', 'decoder'])

    def forward(self, x):
        size = x.size()[2:]
        x = self.encoder(x)
        x = self.decoder(x)
        outputs = list()
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)

        return tuple(outputs)


class Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Downsampling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 3, 2, 2, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 2, 3, 2, 2, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.pool(x1)

        x2 = self.conv2(x)
        x2 = self.pool(x2)

        return torch.cat([x1, x2], dim=1)


class SSnbt(nn.Module):
    def __init__(self, in_channels, dilation=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(SSnbt, self).__init__()
        inter_channels = in_channels // 2
        self.branch1 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, (3, 1), padding=(1, 0), bias=False),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, inter_channels, (1, 3), padding=(0, 1), bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, inter_channels, (3, 1), padding=(dilation, 0), dilation=(dilation, 1),
                      bias=False),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, inter_channels, (1, 3), padding=(0, dilation), dilation=(1, dilation),
                      bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True))

        self.branch2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, (1, 3), padding=(0, 1), bias=False),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, inter_channels, (3, 1), padding=(1, 0), bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, inter_channels, (1, 3), padding=(0, dilation), dilation=(1, dilation),
                      bias=False),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, inter_channels, (3, 1), padding=(dilation, 0), dilation=(dilation, 1),
                      bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True))

        self.relu = nn.ReLU(True)

    @staticmethod
    def channel_shuffle(x, groups):
        n, c, h, w = x.size()

        channels_per_group = c // groups
        x = x.view(n, groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(n, -1, h, w)

        return x

    def forward(self, x):
        # channels split
        x1, x2 = x.split(x.size(1) // 2, 1)

        x1 = self.branch1(x1)
        x2 = self.branch2(x2)

        out = torch.cat([x1, x2], dim=1)
        out = self.relu(out + x)
        out = self.channel_shuffle(out, groups=2)

        return out


class APNModule(nn.Module):
    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(APNModule, self).__init__()
        self.conv1 = _ConvBNReLU(in_channels, in_channels, 3, 2, 1, norm_layer=norm_layer)
        self.conv2 = _ConvBNReLU(in_channels, in_channels, 5, 2, 2, norm_layer=norm_layer)
        self.conv3 = _ConvBNReLU(in_channels, in_channels, 7, 2, 3, norm_layer=norm_layer)
        self.level1 = _ConvBNReLU(in_channels, nclass, 1, norm_layer=norm_layer)
        self.level2 = _ConvBNReLU(in_channels, nclass, 1, norm_layer=norm_layer)
        self.level3 = _ConvBNReLU(in_channels, nclass, 1, norm_layer=norm_layer)
        self.level4 = _ConvBNReLU(in_channels, nclass, 1, norm_layer=norm_layer)
        self.level5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _ConvBNReLU(in_channels, nclass, 1))

    def forward(self, x):
        w, h = x.size()[2:]
        branch3 = self.conv1(x)
        branch2 = self.conv2(branch3)
        branch1 = self.conv3(branch2)

        out = self.level1(branch1)
        out = F.interpolate(out, ((w + 3) // 4, (h + 3) // 4), mode='bilinear', align_corners=True)
        out = self.level2(branch2) + out
        out = F.interpolate(out, ((w + 1) // 2, (h + 1) // 2), mode='bilinear', align_corners=True)
        out = self.level3(branch3) + out
        out = F.interpolate(out, (w, h), mode='bilinear', align_corners=True)
        out = self.level4(x) * out
        out = self.level5(x) + out
        return out


def get_lednet(dataset='citys', backbone='', pretrained=False, root='~/.torch/models',
               pretrained_base=True, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from ..data.dataloader import datasets
    model = LEDNet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(kwargs['local_rank'])
        model.load_state_dict(torch.load(get_model_file('lednet_%s' % (acronyms[dataset]), root=root),
                              map_location=device))
    return model


def get_lednet_citys(**kwargs):
    return get_lednet('citys', **kwargs)


if __name__ == '__main__':
    model = get_lednet_citys()

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vgg import vgg16
from models.utils import weights_init

__all__ = ['fcn32s_vgg16', 'fcn16s_vgg16', 'fcn8s_vgg16']


class FCN32s(nn.Module):
    """There are some difference from original fcn"""

    def __init__(self, num_classes, init_weights=False):
        super(FCN32s, self).__init__()
        self.features = vgg16(pretrained=True).features
        self.head = _FCNHead(512, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        pool5 = self.features(x)
        out = self.head(pool5)
        out = F.interpolate(out, x.size()[2:], mode='bilinear', align_corners=True)

        return out

    def _initialize_weights(self):
        self.head.apply(weights_init)


class FCN16s(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super(FCN16s, self).__init__()
        self.features = vgg16(pretrained=True).features
        self.pool4 = nn.Sequential(*self.features[:24])
        self.pool5 = nn.Sequential(*self.features[24:])
        self.head = _FCNHead(512, num_classes)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        pool4 = self.pool4(x)
        pool5 = self.pool5(pool4)
        score_fr = self.head(pool5)

        score_pool4 = self.score_pool4(pool4)

        upscore2 = F.interpolate(score_fr, score_pool4.size()[2:], mode='bilinear', align_corners=True)
        fuse_pool4 = upscore2 + score_pool4

        out = F.interpolate(fuse_pool4, x.size()[2:], mode='bilinear', align_corners=True)

        return out

    def _initialize_weights(self):
        self.head.apply(weights_init)
        self.score_pool4.apply(weights_init)


class FCN8s(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super(FCN8s, self).__init__()
        self.features = vgg16(pretrained=True).features
        self.pool3 = nn.Sequential(*self.features[:17])
        self.pool4 = nn.Sequential(*self.features[17:24])
        self.pool5 = nn.Sequential(*self.features[24:])
        self.head = _FCNHead(512, num_classes)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)
        score_fr = self.head(pool5)

        score_pool4 = self.score_pool4(pool4)
        score_pool3 = self.score_pool3(pool3)

        upscore2 = F.interpolate(score_fr, score_pool4.size()[2:], mode='bilinear', align_corners=True)
        fuse_pool4 = upscore2 + score_pool4

        upscore_pool4 = F.interpolate(fuse_pool4, score_pool3.size()[2:], mode='bilinear', align_corners=True)
        fuse_pool3 = upscore_pool4 + score_pool3

        out = F.interpolate(fuse_pool3, x.size()[2:], mode='bilinear', align_corners=True)

        return out

    def _initialize_weights(self):
        self.head.apply(weights_init)
        self.score_pool4.apply(weights_init)
        self.score_pool3.apply(weights_init)


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


def fcn32s_vgg16(num_classes=21):
    model = FCN32s(num_classes)
    return model


def fcn16s_vgg16(num_classes=21):
    model = FCN16s(num_classes)
    return model


def fcn8s_vgg16(num_classes=21):
    model = FCN8s(num_classes)
    return model


if __name__ == '__main__':
    img = torch.randn((4, 3, 512, 512))
    model = fcn32s_vgg16(21)
    out = model(img)

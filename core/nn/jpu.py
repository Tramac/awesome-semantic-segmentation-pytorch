"""Joint Pyramid Upsampling"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['JPU']


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


# copy from: https://github.com/wuhuikai/FastFCN/blob/master/encoding/nn/customize.py
class JPU(nn.Module):
    def __init__(self, in_channels, width=512, norm_layer=nn.BatchNorm2d, **kwargs):
        super(JPU, self).__init__()

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(True))

        self.dilation1 = nn.Sequential(
            SeparableConv2d(3 * width, width, 3, padding=1, dilation=1, bias=False),
            norm_layer(width),
            nn.ReLU(True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(3 * width, width, 3, padding=2, dilation=2, bias=False),
            norm_layer(width),
            nn.ReLU(True))
        self.dilation3 = nn.Sequential(
            SeparableConv2d(3 * width, width, 3, padding=4, dilation=4, bias=False),
            norm_layer(width),
            nn.ReLU(True))
        self.dilation4 = nn.Sequential(
            SeparableConv2d(3 * width, width, 3, padding=8, dilation=8, bias=False),
            norm_layer(width),
            nn.ReLU(True))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        size = feats[-1].size()[2:]
        feats[-2] = F.interpolate(feats[-2], size, mode='bilinear', align_corners=True)
        feats[-3] = F.interpolate(feats[-3], size, mode='bilinear', align_corners=True)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)],
                         dim=1)

        return inputs[0], inputs[1], inputs[2], feat

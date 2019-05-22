"""MobileNet and MobileNetV2."""
import torch
import torch.nn as nn

from core.nn import _ConvBNReLU, _DepthwiseConv, InvertedResidual

__all__ = ['MobileNet', 'MobileNetV2', 'get_mobilenet', 'get_mobilenet_v2',
           'mobilenet1_0', 'mobilenet_v2_1_0', 'mobilenet0_75', 'mobilenet_v2_0_75',
           'mobilenet0_5', 'mobilenet_v2_0_5', 'mobilenet0_25', 'mobilenet_v2_0_25']


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, multiplier=1.0, norm_layer=nn.BatchNorm2d, **kwargs):
        super(MobileNet, self).__init__()
        conv_dw_setting = [
            [64, 1, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 2, 2]]
        input_channels = int(32 * multiplier) if multiplier > 1.0 else 32
        features = [_ConvBNReLU(3, input_channels, 3, 2, 1, norm_layer=norm_layer)]

        for c, n, s in conv_dw_setting:
            out_channels = int(c * multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(_DepthwiseConv(input_channels, out_channels, stride, norm_layer))
                input_channels = out_channels
        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Linear(int(1024 * multiplier), num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), x.size(1)))
        return x


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, multiplier=1.0, norm_layer=nn.BatchNorm2d, **kwargs):
        super(MobileNetV2, self).__init__()
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]]
        # building first layer
        input_channels = int(32 * multiplier) if multiplier > 1.0 else 32
        last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
        features = [_ConvBNReLU(3, input_channels, 3, 2, 1, relu6=True, norm_layer=norm_layer)]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            out_channels = int(c * multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channels, out_channels, stride, t, norm_layer))
                input_channels = out_channels

        # building last several layers
        features.append(_ConvBNReLU(input_channels, last_channels, 1, relu6=True, norm_layer=norm_layer))
        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Linear(last_channels, num_classes))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), x.size(1)))
        return x


# Constructor
def get_mobilenet(multiplier=1.0, pretrained=False, root='~/.torch/models', **kwargs):
    model = MobileNet(multiplier=multiplier, **kwargs)

    if pretrained:
        raise ValueError("Not support pretrained")
    return model


def get_mobilenet_v2(multiplier=1.0, pretrained=False, root='~/.torch/models', **kwargs):
    model = MobileNetV2(multiplier=multiplier, **kwargs)

    if pretrained:
        raise ValueError("Not support pretrained")
    return model


def mobilenet1_0(**kwargs):
    return get_mobilenet(1.0, **kwargs)


def mobilenet_v2_1_0(**kwargs):
    return get_mobilenet_v2(1.0, **kwargs)


def mobilenet0_75(**kwargs):
    return get_mobilenet(0.75, **kwargs)


def mobilenet_v2_0_75(**kwargs):
    return get_mobilenet_v2(0.75, **kwargs)


def mobilenet0_5(**kwargs):
    return get_mobilenet(0.5, **kwargs)


def mobilenet_v2_0_5(**kwargs):
    return get_mobilenet_v2(0.5, **kwargs)


def mobilenet0_25(**kwargs):
    return get_mobilenet(0.25, **kwargs)


def mobilenet_v2_0_25(**kwargs):
    return get_mobilenet_v2(0.25, **kwargs)


if __name__ == '__main__':
    model = mobilenet0_5()

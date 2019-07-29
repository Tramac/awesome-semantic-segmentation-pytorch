import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_models.xception import get_xception
from .deeplabv3 import _ASPP
from .fcn import _FCNHead
from ..nn import _ConvBNReLU

__all__ = ['DeepLabV3Plus', 'get_deeplabv3_plus', 'get_deeplabv3_plus_xception_voc']


class DeepLabV3Plus(nn.Module):
    r"""DeepLabV3Plus
    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'xception').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic
        Image Segmentation."
    """

    def __init__(self, nclass, backbone='xception', aux=True, pretrained_base=True, dilated=True, **kwargs):
        super(DeepLabV3Plus, self).__init__()
        self.aux = aux
        self.nclass = nclass
        output_stride = 8 if dilated else 32

        self.pretrained = get_xception(pretrained=pretrained_base, output_stride=output_stride, **kwargs)

        # deeplabv3 plus
        self.head = _DeepLabHead(nclass, **kwargs)
        if aux:
            self.auxlayer = _FCNHead(728, nclass, **kwargs)

    def base_forward(self, x):
        # Entry flow
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)

        x = self.pretrained.conv2(x)
        x = self.pretrained.bn2(x)
        x = self.pretrained.relu(x)

        x = self.pretrained.block1(x)
        # add relu here
        x = self.pretrained.relu(x)
        low_level_feat = x

        x = self.pretrained.block2(x)
        x = self.pretrained.block3(x)

        # Middle flow
        x = self.pretrained.midflow(x)
        mid_level_feat = x

        # Exit flow
        x = self.pretrained.block20(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.conv3(x)
        x = self.pretrained.bn3(x)
        x = self.pretrained.relu(x)

        x = self.pretrained.conv4(x)
        x = self.pretrained.bn4(x)
        x = self.pretrained.relu(x)

        x = self.pretrained.conv5(x)
        x = self.pretrained.bn5(x)
        x = self.pretrained.relu(x)
        return low_level_feat, mid_level_feat, x

    def forward(self, x):
        size = x.size()[2:]
        c1, c3, c4 = self.base_forward(x)
        outputs = list()
        x = self.head(c4, c1)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _DeepLabHead(nn.Module):
    def __init__(self, nclass, c1_channels=128, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(2048, [12, 24, 36], norm_layer=norm_layer, **kwargs)
        self.c1_block = _ConvBNReLU(c1_channels, 48, 3, padding=1, norm_layer=norm_layer)
        self.block = nn.Sequential(
            _ConvBNReLU(304, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.5),
            _ConvBNReLU(256, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.1),
            nn.Conv2d(256, nclass, 1))

    def forward(self, x, c1):
        size = c1.size()[2:]
        c1 = self.c1_block(c1)
        x = self.aspp(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return self.block(torch.cat([x, c1], dim=1))


def get_deeplabv3_plus(dataset='pascal_voc', backbone='xception', pretrained=False, root='~/.torch/models',
                       pretrained_base=True, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from ..data.dataloader import datasets
    model = DeepLabV3Plus(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(kwargs['local_rank'])
        model.load_state_dict(
            torch.load(get_model_file('deeplabv3_plus_%s_%s' % (backbone, acronyms[dataset]), root=root),
                map_location=device))
    return model


def get_deeplabv3_plus_xception_voc(**kwargs):
    return get_deeplabv3_plus('pascal_voc', 'xception', **kwargs)


if __name__ == '__main__':
    model = get_deeplabv3_plus_xception_voc()

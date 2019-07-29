"ESPNetv2: A Light-weight, Power Efficient, and General Purpose for Semantic Segmentation"
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.base_models import eespnet, EESP
from core.nn import _ConvBNPReLU, _BNPReLU


class ESPNetV2(nn.Module):
    r"""ESPNetV2

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
        Sachin Mehta, et al. "ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network."
        arXiv preprint arXiv:1811.11431 (2018).
    """

    def __init__(self, nclass, backbone='', aux=False, jpu=False, pretrained_base=False, **kwargs):
        super(ESPNetV2, self).__init__()
        self.pretrained = eespnet(pretrained=pretrained_base, **kwargs)
        self.proj_L4_C = _ConvBNPReLU(256, 128, 1, **kwargs)
        self.pspMod = nn.Sequential(
            EESP(256, 128, stride=1, k=4, r_lim=7, **kwargs),
            _PSPModule(128, 128, **kwargs))
        self.project_l3 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(128, nclass, 1, bias=False))
        self.act_l3 = _BNPReLU(nclass, **kwargs)
        self.project_l2 = _ConvBNPReLU(64 + nclass, nclass, 1, **kwargs)
        self.project_l1 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(32 + nclass, nclass, 1, bias=False))

        self.aux = aux

        self.__setattr__('exclusive', ['proj_L4_C', 'pspMod', 'project_l3', 'act_l3', 'project_l2', 'project_l1'])

    def forward(self, x):
        size = x.size()[2:]
        out_l1, out_l2, out_l3, out_l4 = self.pretrained(x, seg=True)
        out_l4_proj = self.proj_L4_C(out_l4)
        up_l4_to_l3 = F.interpolate(out_l4_proj, scale_factor=2, mode='bilinear', align_corners=True)
        merged_l3_upl4 = self.pspMod(torch.cat([out_l3, up_l4_to_l3], 1))
        proj_merge_l3_bef_act = self.project_l3(merged_l3_upl4)
        proj_merge_l3 = self.act_l3(proj_merge_l3_bef_act)
        out_up_l3 = F.interpolate(proj_merge_l3, scale_factor=2, mode='bilinear', align_corners=True)
        merge_l2 = self.project_l2(torch.cat([out_l2, out_up_l3], 1))
        out_up_l2 = F.interpolate(merge_l2, scale_factor=2, mode='bilinear', align_corners=True)
        merge_l1 = self.project_l1(torch.cat([out_l1, out_up_l2], 1))

        outputs = list()
        merge1_l1 = F.interpolate(merge_l1, scale_factor=2, mode='bilinear', align_corners=True)
        outputs.append(merge1_l1)
        if self.aux:
            # different from paper
            auxout = F.interpolate(proj_merge_l3_bef_act, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)

        return tuple(outputs)


# different from PSPNet
class _PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels=1024, sizes=(1, 2, 4, 8), **kwargs):
        super(_PSPModule, self).__init__()
        self.stages = nn.ModuleList(
            [nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False) for _ in sizes])
        self.project = _ConvBNPReLU(in_channels * (len(sizes) + 1), out_channels, 1, 1, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        feats = [x]
        for stage in self.stages:
            x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
            upsampled = F.interpolate(stage(x), size, mode='bilinear', align_corners=True)
            feats.append(upsampled)
        return self.project(torch.cat(feats, dim=1))


def get_espnet(dataset='pascal_voc', backbone='', pretrained=False, root='~/.torch/models',
               pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from core.data.dataloader import datasets
    model = ESPNetV2(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(kwargs['local_rank'])
        model.load_state_dict(torch.load(get_model_file('espnet_%s_%s' % (backbone, acronyms[dataset]), root=root),
                              map_location=device))
    return model


def get_espnet_citys(**kwargs):
    return get_espnet('citys', **kwargs)


if __name__ == '__main__':
    model = get_espnet_citys()

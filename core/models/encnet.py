"""Context Encoding for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from .fcn import _FCNHead

__all__ = ['EncNet', 'EncModule', 'get_encnet', 'get_encnet_resnet50_ade',
           'get_encnet_resnet101_ade', 'get_encnet_resnet152_ade']


class EncNet(SegBaseModel):
    def __init__(self, nclass, backbone='resnet50', aux=True, se_loss=True, lateral=False,
                 pretrained_base=True, **kwargs):
        super(EncNet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)
        self.head = _EncHead(2048, nclass, se_loss=se_loss, lateral=lateral, **kwargs)
        if aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

    def forward(self, x):
        size = x.size()[2:]
        features = self.base_forward(x)

        x = list(self.head(*features))
        x[0] = F.interpolate(x[0], size, mode='bilinear', align_corners=True)
        if self.aux:
            auxout = self.auxlayer(features[2])
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            x.append(auxout)
        return tuple(x)


class _EncHead(nn.Module):
    def __init__(self, in_channels, nclass, se_loss=True, lateral=True,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_EncHead, self).__init__()
        self.lateral = lateral
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
            norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        if lateral:
            self.connect = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(512, 512, 1, bias=False),
                    norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
                    nn.ReLU(True)),
                nn.Sequential(
                    nn.Conv2d(1024, 512, 1, bias=False),
                    norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
                    nn.ReLU(True)),
            ])
            self.fusion = nn.Sequential(
                nn.Conv2d(3 * 512, 512, 3, padding=1, bias=False),
                norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
                nn.ReLU(True)
            )
        self.encmodule = EncModule(512, nclass, ncodes=32, se_loss=se_loss,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
        self.conv6 = nn.Sequential(
            nn.Dropout(0.1, False),
            nn.Conv2d(512, nclass, 1)
        )

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))
        outs = list(self.encmodule(feat))
        outs[0] = self.conv6(outs[0])
        return tuple(outs)


class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            Encoding(D=in_channels, K=ncodes),
            nn.BatchNorm1d(ncodes),
            nn.ReLU(True),
            Mean(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)


class Encoding(nn.Module):
    def __init__(self, D, K):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1. / ((self.K * self.D) ** (1 / 2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        # input X is a 4D tensor
        assert (X.size(1) == self.D)
        B, D = X.size(0), self.D
        if X.dim() == 3:
            # BxDxN -> BxNxD
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW -> Bx(HW)xD
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # assignment weights BxNxK
        A = F.softmax(self.scale_l2(X, self.codewords, self.scale), dim=2)
        # aggregate
        E = self.aggregate(A, X, self.codewords)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'N x' + str(self.D) + '=>' + str(self.K) + 'x' \
               + str(self.D) + ')'

    @staticmethod
    def scale_l2(X, C, S):
        S = S.view(1, 1, C.size(0), 1)
        X = X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1))
        C = C.unsqueeze(0).unsqueeze(0)
        SL = S * (X - C)
        SL = SL.pow(2).sum(3)
        return SL

    @staticmethod
    def aggregate(A, X, C):
        A = A.unsqueeze(3)
        X = X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1))
        C = C.unsqueeze(0).unsqueeze(0)
        E = A * (X - C)
        E = E.sum(1)
        return E


class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


def get_encnet(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='~/.torch/models',
               pretrained_base=True, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from ..data.dataloader import datasets
    model = EncNet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(kwargs['local_rank'])
        model.load_state_dict(torch.load(get_model_file('encnet_%s_%s' % (backbone, acronyms[dataset]), root=root),
                              map_location=device))
    return model


def get_encnet_resnet50_ade(**kwargs):
    return get_encnet('ade20k', 'resnet50', **kwargs)


def get_encnet_resnet101_ade(**kwargs):
    return get_encnet('ade20k', 'resnet101', **kwargs)


def get_encnet_resnet152_ade(**kwargs):
    return get_encnet('ade20k', 'resnet152', **kwargs)


if __name__ == '__main__':
    img = torch.randn(2, 3, 224, 224)
    model = get_encnet_resnet50_ade()
    outputs = model(img)

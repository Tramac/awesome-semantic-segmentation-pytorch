"""Efficient Neural Network"""
import torch
import torch.nn as nn

__all__ = ['ENet', 'get_enet', 'get_enet_citys']


class ENet(nn.Module):
    """Efficient Neural Network"""

    def __init__(self, nclass, backbone='', aux=False, jpu=False, pretrained_base=None, **kwargs):
        super(ENet, self).__init__()
        self.initial = InitialBlock(13, **kwargs)
#block 1:
        self.bottleneck1_0 = Bottleneck(16, 16, 64, downsampling=True, **kwargs)
        self.bottleneck1_1 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck1_2 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck1_3 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck1_4 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck1_5 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck1_6 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck1_7 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck1_8 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck1_9 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck1_10 = Bottleneck(64, 16, 64, **kwargs)
#blcok 2:
        self.bottleneck2_0 = Bottleneck(64, 32, 128, downsampling=True, **kwargs)
        self.bottleneck2_1 = Bottleneck(128, 32, 128, **kwargs)
        self.bottleneck2_2 = Bottleneck(128, 32, 128, dilation=2, **kwargs)
        self.bottleneck2_3 = Bottleneck(128, 32, 128, asymmetric=True, **kwargs)
        self.bottleneck2_4 = Bottleneck(128, 32, 128, dilation=4, **kwargs)
        self.bottleneck2_5 = Bottleneck(128, 32, 128, **kwargs)
        self.bottleneck2_6 = Bottleneck(128, 32, 128, **kwargs)
        self.bottleneck2_7 = Bottleneck(128, 32, 128, **kwargs)
        self.bottleneck2_8 = Bottleneck(128, 32, 128, dilation=8, **kwargs)
        self.bottleneck2_9 = Bottleneck(128, 32, 128, asymmetric=True, **kwargs)
        self.bottleneck2_10 = Bottleneck(128, 32, 128, dilation=16, **kwargs)
#block 3:
        self.bottleneck3_0 = Bottleneck(128, 32, 128, **kwargs)
        self.bottleneck3_1 = Bottleneck(128, 32, 128, dilation=2, **kwargs)
        self.bottleneck3_2 = Bottleneck(128, 32, 128, asymmetric=True, **kwargs)
        self.bottleneck3_3 = Bottleneck(128, 32, 128, dilation=4, **kwargs)
        self.bottleneck3_4 = Bottleneck(128, 32, 128, **kwargs)
        self.bottleneck3_5 = Bottleneck(128, 32, 128, **kwargs)
        self.bottleneck3_6 = Bottleneck(128, 32, 128, **kwargs)
        self.bottleneck3_7 = Bottleneck(128, 32, 128, **kwargs)
        self.bottleneck3_8 = Bottleneck(128, 32, 128, dilation=8, **kwargs)
        self.bottleneck3_9 = Bottleneck(128, 32, 128, asymmetric=True, **kwargs)
        self.bottleneck3_10 = Bottleneck(128, 32, 128, dilation=16, **kwargs)
#block 4:
        self.bottleneck4_0 = UpsamplingBottleneck(128, 16, 64, **kwargs)
        self.bottleneck4_1 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck4_2 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck4_3 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck4_4 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck4_5 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck4_6 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck4_7 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck4_8 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck4_9 = Bottleneck(64, 16, 64, **kwargs)
        self.bottleneck4_10 = Bottleneck(64, 16, 64, **kwargs)
#block 5:
        self.bottleneck5_0 = UpsamplingBottleneck(64, 4, 16, **kwargs)
        self.bottleneck5_1 = Bottleneck(16, 4, 16, **kwargs)
        self.bottleneck5_2 = Bottleneck(16, 4, 16, **kwargs)
        self.bottleneck5_3 = Bottleneck(16, 4, 16, **kwargs)
        self.bottleneck5_4 = Bottleneck(16, 4, 16, **kwargs)
        self.bottleneck5_5 = Bottleneck(16, 4, 16, **kwargs)
        self.bottleneck5_6 = Bottleneck(16, 4, 16, **kwargs)
        self.bottleneck5_7 = Bottleneck(16, 4, 16, **kwargs)
        self.bottleneck5_8 = Bottleneck(16, 4, 16, **kwargs)
        self.bottleneck5_9 = Bottleneck(16, 4, 16, **kwargs)
        self.bottleneck5_10 = Bottleneck(16, 4, 16, **kwargs)
#block 6:
        self.fullconv = nn.ConvTranspose2d(16, nclass, 2, 2, bias=False)

        self.__setattr__('exclusive', ['bottleneck1_0', 'bottleneck1_1', 'bottleneck1_2', 'bottleneck1_3',
                                       'bottleneck1_4', 'bottleneck1_5', 'bottleneck1_6', 'bottleneck1_7',
                                       'bottleneck1_8', 'bottleneck1_9', 'bottleneck1_10','bottleneck2_0', 
                                       'bottleneck2_1', 'bottleneck2_2', 'bottleneck2_3', 'bottleneck2_4', 
                                       'bottleneck2_5', 'bottleneck2_6', 'bottleneck2_7', 'bottleneck2_8', 
                                       'bottleneck2_9', 'bottleneck2_10','bottleneck3_0', 'bottleneck3_1', 
                                       'bottleneck3_2', 'bottleneck3_3', 'bottleneck3_4', 'bottleneck3_5', 
                                       'bottleneck3_6', 'bottleneck3_7', 'bottleneck3_8', 'bottleneck3_9', 
                                       'bottleneck3_10','bottleneck4_0', 'bottleneck4_1', 'bottleneck4_2',
                                       'bottleneck4_3', 'bottleneck4_4', 'bottleneck4_5', 'bottleneck4_6',
                                       'bottleneck4_7', 'bottleneck4_8', 'bottleneck4_9', 'bottleneck4_10',
                                       'bottleneck5_0', 'bottleneck5_1', 'bottleneck5_2', 'bottleneck5_3',
                                       'bottleneck5_4', 'bottleneck5_5', 'bottleneck5_6', 'bottleneck5_7',
                                       'bottleneck5_8', 'bottleneck5_9', 'bottleneck5_10','fullconv'])

    def forward(self, x):
        # init
        x = self.initial(x)

        # stage 1
        x, max_indices1 = self.bottleneck1_0(x)
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)
        x = self.bottleneck1_5(x)
        x = self.bottleneck1_6(x)
        x = self.bottleneck1_7(x)
        x = self.bottleneck1_8(x)
        x = self.bottleneck1_9(x)
        x = self.bottleneck1_10(x)
        # stage 2
        x, max_indices2 = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x)
        x = self.bottleneck2_8(x)
        x = self.bottleneck2_9(x)
        x = self.bottleneck2_10(x)
        # stage 3
        x = self.bottleneck3_0(x)
        x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x)
        x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x)
        x = self.bottleneck3_5(x)
        x = self.bottleneck3_6(x)
        x = self.bottleneck3_7(x)
        x = self.bottleneck3_8(x)
        x = self.bottleneck3_9(x)
        x = self.bottleneck3_10(x)

        # stage 4
        x = self.bottleneck4_0(x, max_indices2)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)
        x = self.bottleneck4_3(x)
        x = self.bottleneck4_4(x)
        x = self.bottleneck4_5(x)
        x = self.bottleneck4_6(x)
        x = self.bottleneck4_7(x)
        x = self.bottleneck4_8(x)
        x = self.bottleneck4_9(x)
        x = self.bottleneck4_10(x)
        # stage 5
        x = self.bottleneck5_0(x, max_indices1)
        x = self.bottleneck5_1(x)
        x = self.bottleneck5_2(x)
        x = self.bottleneck5_3(x)
        x = self.bottleneck5_4(x)
        x = self.bottleneck5_5(x)
        x = self.bottleneck5_6(x)
        x = self.bottleneck5_7(x)
        x = self.bottleneck5_8(x)
        x = self.bottleneck5_9(x)
        x = self.bottleneck5_10(x)
        # out
        x = self.fullconv(x)
        return tuple([x])


class InitialBlock(nn.Module):
    """ENet initial block"""

    def __init__(self, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(3, out_channels, 3, 2, 1, bias=False)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.bn = norm_layer(out_channels + 3)
        self.act = nn.RReLU()

    def forward(self, x):
        x_conv = self.conv(x)
        x_pool = self.maxpool(x)
        x = torch.cat([x_conv, x_pool], dim=1)
        x = self.bn(x)
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    """Bottlenecks include regular, asymmetric, downsampling, dilated"""

    def __init__(self, in_channels, inter_channels, out_channels, dilation=1, asymmetric=False,
                 downsampling=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(Bottleneck, self).__init__()
        self.downsamping = downsampling
        if downsampling:
            self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)
            self.conv_down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                norm_layer(out_channels)
            )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels),
            nn.RReLU()
        )

        if downsampling:
            self.conv2 = nn.Sequential(
                nn.Conv2d(inter_channels, inter_channels, 2, stride=2, bias=False),
                norm_layer(inter_channels),
                nn.RReLU()
            )
        else:
            if asymmetric:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(inter_channels, inter_channels, (5, 1), padding=(2, 0), bias=False),
                    nn.Conv2d(inter_channels, inter_channels, (1, 5), padding=(0, 2), bias=False),
                    norm_layer(inter_channels),
                    nn.RReLU()
                )
            else:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(inter_channels, inter_channels, 3, dilation=dilation, padding=dilation, bias=False),
                    norm_layer(inter_channels),
                    nn.RReLU()
                )
        self.conv3 = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.Dropout2d(0.1)
        )
        self.act = nn.RReLU()

    def forward(self, x):
        identity = x
        if self.downsamping:
            identity, max_indices = self.maxpool(identity)
            identity = self.conv_down(identity)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.act(out + identity)

        if self.downsamping:
            return out, max_indices
        else:
            return out


class UpsamplingBottleneck(nn.Module):
    """upsampling Block"""

    def __init__(self, in_channels, inter_channels, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(UpsamplingBottleneck, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )
        self.upsampling = nn.MaxUnpool2d(2)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels),
            nn.RReLU(),
            nn.ConvTranspose2d(inter_channels, inter_channels, 2, 2, bias=False),
            norm_layer(inter_channels),
            nn.RReLU(),
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.Dropout2d(0.1)
        )
        self.act = nn.RReLU()

    def forward(self, x, max_indices):
        out_up = self.conv(x)
        out_up = self.upsampling(out_up, max_indices)

        out_ext = self.block(x)
        out = self.act(out_up + out_ext)
        return out


def get_enet(dataset='citys', backbone='', pretrained=False, root='~/.torch/models', pretrained_base=True, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from core.data.dataloader import datasets
    model = ENet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(kwargs['local_rank'])
        model.load_state_dict(torch.load(get_model_file('enet_%s' % (acronyms[dataset]), root=root),
                              map_location=device))
    return model


def get_enet_citys(**kwargs):
    return get_enet('citys', '', **kwargs)


if __name__ == '__main__':
    img = torch.randn(1, 3, 512, 512)
    model = get_enet_citys()
    output = model(img)

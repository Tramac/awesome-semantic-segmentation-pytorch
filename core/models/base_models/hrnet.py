import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels,
                 fuse_method, multi_scale_output=True, norm_layer=nn.BatchNorm2d):
        super(HighResolutionModule, self).__init__()
        assert num_branches == len(num_blocks)
        assert num_branches == len(num_channels)
        assert num_branches == len(num_inchannels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels, norm_layer=norm_layer)
        self.fuse_layers = self._make_fuse_layers(norm_layer)
        self.relu = nn.ReLU(True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion,
                          1, stride, bias=False),
                norm_layer(num_channels[branch_index] * block.expansion))

        layers = list()
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index],
                            stride, downsample, norm_layer=norm_layer))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels, norm_layer=nn.BatchNorm2d):
        branches = list()
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels, norm_layer=norm_layer))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self, norm_layer=nn.BatchNorm2d):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = list()
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, bias=False),
                        norm_layer(num_inchannels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = list()
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                norm_layer(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                norm_layer(num_outchannels_conv3x3),
                                nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = list()
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HighResolutionNet(nn.Module):
    def __init__(self, blocks, num_channels, num_modules, num_branches, num_blocks,
                 fuse_method, norm_layer=nn.BatchNorm2d, **kwargs):
        super(HighResolutionNet, self).__init__()
        self.num_branches = num_branches

        # deep stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            norm_layer(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            norm_layer(64),
            nn.ReLU(True))

        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4, norm_layer=norm_layer)

        # stage 2
        num_channel, block = num_channels[0], blocks[0]
        channels = [channel * block.expansion for channel in num_channel]
        self.transition1 = self._make_transition_layer([256], channels, norm_layer)
        self.stage2, pre_stage_channels = self._make_stage(num_modules[0], num_branches[0],
                                                           num_blocks[0], channels, block,
                                                           fuse_method[0], channels,
                                                           norm_layer=norm_layer)

        # stage 3
        num_channel, block = num_channels[1], blocks[1]
        channels = [channel * block.expansion for channel in num_channel]
        self.transition1 = self._make_transition_layer(pre_stage_channels, channels, norm_layer)
        self.stage3, pre_stage_channels = self._make_stage(num_modules[1], num_branches[1],
                                                           num_blocks[1], channels, block,
                                                           fuse_method[1], channels,
                                                           norm_layer=norm_layer)

        # stage 4
        num_channel, block = num_channels[2], blocks[2]
        channels = [channel * block.expansion for channel in num_channel]
        self.transition1 = self._make_transition_layer(pre_stage_channels, channels, norm_layer)
        self.stage4, pre_stage_channels = self._make_stage(num_modules[2], num_branches[2],
                                                           num_blocks[2], channels, block,
                                                           fuse_method[2], channels,
                                                           norm_layer=norm_layer)

        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(pre_stage_channels, norm_layer)

        self.classifier = nn.Linear(2048, 1000)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion))

        layers = list()
        layers.append(block(inplanes, planes, stride, downsample=downsample, norm_layer=norm_layer))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer, norm_layer=nn.BatchNorm2d):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = list()
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, padding=1, bias=False),
                        norm_layer(num_channels_cur_layer[i]),
                        nn.ReLU(True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = list()
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] if j == i - num_branches_pre else in_channels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
                        norm_layer(out_channels),
                        nn.ReLU(True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, num_modules, num_branches, num_blocks, num_channels, block,
                    fuse_method, num_inchannels, multi_scale_output=True, norm_layer=nn.BatchNorm2d):
        modules = list()
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels,
                                                fuse_method, reset_multi_scale_output, norm_layer=norm_layer))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def _make_head(self, pre_stage_channels, norm_layer=nn.BatchNorm2d):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = list()
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block, channels, head_channels[i], 1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                norm_layer(out_channels),
                nn.ReLU(True))

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(head_channels[3] * head_block.expansion, 2048, 1),
            norm_layer(2048),
            nn.ReLU(True))

        return incre_modules, downsamp_modules, final_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)

        x_list = list()
        for i in range(self.num_branches[0]):
            if self.transition1[i] is not None:
                tmp = self.transition1[i](x)
                print(tmp.size())
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.num_branches[1]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.num_branches[2]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # Classification Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](y)

        y = self.final_layer(y)

        y = F.avg_pool2d(y, kernel_size=y.size()
        [2:]).view(y.size(0), -1)

        y = self.classifier(y)

        return y


blocks = [BasicBlock, BasicBlock, BasicBlock]
num_modules = [1, 1, 1]
num_branches = [2, 3, 4]
num_blocks = [[4, 4], [4, 4, 4], [4, 4, 4, 4]]
num_channels = [[256, 256], [32, 64, 128], [32, 64, 128, 256]]
fuse_method = ['sum', 'sum', 'sum']

if __name__ == '__main__':
    img = torch.randn(1, 3, 256, 256)
    model = HighResolutionNet(blocks, num_channels, num_modules, num_branches, num_blocks, fuse_method)
    output = model(img)

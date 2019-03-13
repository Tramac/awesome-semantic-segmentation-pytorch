import torch
import torch.nn as nn
import torch.nn.functional as F

from base_models.vgg import vgg16

__all__ = ['fcn32s']


class FCN32s(nn.Module):
    def __init__(self, num_classes):
        super(FCN32s, self).__init__()
        self.features = vgg16(pretrained=True).features
        self.fc = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

    def forward(self, x):
        pool5 = self.features(x)
        fc7 = self.fc(pool5)
        score = self.score_fr(fc7)
        out = F.interpolate(score, x.size()[2:])
        return out


def fcn32s(num_classes=21):
    model = FCN32s(num_classes)
    return model


if __name__ == '__main__':
    img = torch.randn((4, 3, 224, 224))
    model = fcn32s(21)
    out = model(img)
    print(out.size())

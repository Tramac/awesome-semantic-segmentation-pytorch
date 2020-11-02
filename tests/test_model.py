"""Model overfitting test"""
import argparse
import time
import os
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from core.models.model_zoo import get_segmentation_model
from core.utils.loss import MixSoftmaxCrossEntropyLoss, EncNetLoss, ICNetLoss
from core.utils.lr_scheduler import LRScheduler
from core.utils.score import hist_info, compute_score
from core.utils.visualize import get_color_pallete
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Overfitting Test')
    # model
    parser.add_argument('--model', type=str, default='fcn32s',
                        choices=['fcn32s', 'fcn16s', 'fcn8s', 'fcn', 'psp', 
                        'deeplabv3', 'danet', 'denseaspp', 'bisenet', 'encnet', 
                        'dunet', 'icnet', 'enet', 'ocnet'],
                        help='model name (default: fcn32s)')
    parser.add_argument('--backbone', type=str, default='vgg16',
                        choices=['vgg16', 'resnet18', 'resnet50', 'resnet101', 
                        'resnet152', 'densenet121', '161', '169', '201'],
                        help='backbone name (default: vgg16)')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        choices=['pascal_voc', 'pascal_aug', 'ade20k', 'citys', 
                        'sbu'],
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    args.device = device
    print(args)
    return args


class VOCSegmentation(object):
    def __init__(self):
        super(VOCSegmentation, self).__init__()
        self.img = Image.open('test_img.jpg').convert('RGB')
        self.mask = Image.open('test_mask.png')

        self.img = self.img.resize((504, 368), Image.BILINEAR)
        self.mask = self.mask.resize((504, 368), Image.NEAREST)

    def get(self):
        img, mask = self._img_transform(self.img), self._mask_transform(self.mask)
        return img, mask

    def _img_transform(self, img):
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        img = input_transform(img)
        img = img.unsqueeze(0)

        # For adaptive pooling
        # img = torch.cat([img, img], dim=0)
        return img

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        target = torch.from_numpy(target).long()
        target = target.unsqueeze(0)

        # For adaptive pooling
        # target = torch.cat([target, target], dim=0)
        return target


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.img, self.target = VOCSegmentation().get()

        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            aux=False, norm_layer=nn.BatchNorm2d).to(args.device)

        self.criterion = MixSoftmaxCrossEntropyLoss(False, 0., ignore_label=-1).to(args.device)

        # for EncNet
        # self.criterion = EncNetLoss(nclass=21, ignore_label=-1).to(args.device)
        # for ICNet
        # self.criterion = ICNetLoss(nclass=21, ignore_index=-1).to(args.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=args.lr,
                                          weight_decay=args.weight_decay)
        self.lr_scheduler = LRScheduler(mode='poly', base_lr=args.lr, nepochs=args.epochs,
                                        iters_per_epoch=1, power=0.9)

    def train(self):
        self.model.train()
        start_time = time.time()
        for epoch in range(self.args.epochs):
            self.lr_scheduler(self.optimizer, epoch)
            cur_lr = self.lr_scheduler.learning_rate
            # self.lr_scheduler(self.optimizer, epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = cur_lr

            images = self.img.to(self.args.device)
            targets = self.target.to(self.args.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss['loss'].backward()
            self.optimizer.step()

            pred = torch.argmax(outputs[0], 1).cpu().data.numpy()
            mask = get_color_pallete(pred.squeeze(0), self.args.dataset)
            save_pred(self.args, epoch, mask)
            hist, labeled, correct = hist_info(pred, targets.cpu().numpy(), 21)
            _, mIoU, _, pixAcc = compute_score(hist, correct, labeled)

            print('Epoch: [%2d/%2d] || Time: %4.4f sec || lr: %.8f || Loss: %.4f || pixAcc: %.3f || mIoU: %.3f' % (
                epoch, self.args.epochs, time.time() - start_time, cur_lr, loss['loss'].item(), pixAcc, mIoU))


def save_pred(args, epoch, mask):
    directory = "runs/%s/" % (args.model)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + '{}_epoch_{}.png'.format(args.model, epoch + 1)
    mask.save(filename)


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    print('Test model: ', args.model)
    trainer.train()

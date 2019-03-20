import argparse
import time
import os

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from data_loader import get_segmentation_dataset
from models.utils import adjust_learning_rate
from models.model_zoo import get_model
from utils.score import SegmentationMetric

parser = argparse.ArgumentParser(
    description='Semantic Segmentation Training With Pytorch')
# model and dataset
parser.add_argument('--model', type=str, default='fcn32s_vgg16',
                    help='model name (default: fcn32)')
parser.add_argument('--dataset_root', default='./datasets',
                    help='Dataset root directory path')
parser.add_argument('--dataset', type=str, default='pascal_voc',
                    help='dataset name (default: pascal_voc)')
parser.add_argument('--base-size', type=int, default=520,
                    help='base image size')
parser.add_argument('--crop-size', type=int, default=480,
                    help='crop image size')
parser.add_argument('--train-split', type=str, default='train',
                    help='dataset train split (default: train)')
# training hyper params
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--start_epoch', type=int, default=0,
                    metavar='N', help='start epochs (default:0)')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                    help='w-decay (default: 5e-4)')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--decay-step', default=200, type=int,
                    help='Sets the lr to the init lr decayed by 10 every decay_step epochs')
parser.add_argument('--decay-mode', default='step', type=str,
                    help='lr decay mode, step/poly')
# checking point
parser.add_argument('--resume', type=str, default=None,
                    help='put the path to resuming file if needed')
parser.add_argument('--num_classes', default=21, type=int,
                    help='Number of classes.')
parser.add_argument('--save-folder', default='./weights',
                    help='Directory for saving checkpoint models')
# evaluation only
parser.add_argument('--eval', action='store_true', default=False,
                    help='evaluation only')
parser.add_argument('--no-val', action='store_true', default=True,
                    help='skip validation during training')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        train_dataset = get_segmentation_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=False)

        # create network
        self.model = get_model(args.model, num_classes=args.num_classes)
        # for multi-GPU
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2])
        self.model.to(device)

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        self.criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # evaluation metrics
        self.metric = SegmentationMetric(args.num_classes)

    def train(self):
        self.model.train()
        total_step = 0
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            for i, (images, targets) in enumerate(self.train_loader):
                adjust_learning_rate(args, self.optimizer, epoch)

                images = images.to(device)
                targets = targets.to(device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_step += 1
                if total_step % 10 == 0:
                    print('Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || Loss: %.4f' % (
                        epoch, args.epochs, i + 1, len(self.train_loader) // args.batch_size, time.time() - start_time,
                        loss.item()))

            if not args.no_eval:
                self.validation(epoch)

            # save every 10 epoch
            if epoch != 0 and epoch % 10 == 0:
                print('Saving state, epoch:', epoch)
                torch.save(self.model.state_dict(),
                           os.path.join(args.save_folder, args.model + '_' + args.dataset + str(total_step) + '.pth'))
        torch.save(self.model.state_dict(), os.path.join(args.save_folder, args.model + '_' + args.dataset + '.pth'))

    def validation(self, epoch):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        self.metric.reset()
        self.model.eval()
        for i, (image, target) in enumerate(self.val_loader):
            image = image.to(device)

            output = self.model(image)
            self.metric.update(output.numpy(), target.numpy())
            pixAcc, mIoU = self.metric.get()
            print('Epoch %d, validation pixAcc: %.3f, mIoU: %.3f' % (epoch, pixAcc, mIoU))


if __name__ == '__main__':
    trainer = Trainer(args)
    if args.eval:
        print('Evaluation model: ', args.resume)
        trainer.validation(args.start_epoch)
    else:
        print('Starting Epoch: %d, Total Epochs: %d' % (args.start_epoch, args.epochs))
        trainer.train()

from __future__ import print_function

import os
import argparse

import torch
import torch.utils.data as data

from torchvision import transforms
from data_loader import get_segmentation_dataset
from models.model_zoo import get_segmentation_model
from utils.score import SegmentationMetric
from utils.visualize import get_color_pallete

parser = argparse.ArgumentParser(
    description='Semantic Segmentation Evaluation')
parser.add_argument('--model', type=str, default='fcn32s',
                    help='model name (default: fcn32)')
parser.add_argument('--backbone', type=str, default='vgg16',
                    help='backbone name (default: resnet50)')
parser.add_argument('--dataset', type=str, default='pascal_voc',
                    help='dataset name (default: pascal_voc, pascal_aug. choice=[pascal_voc, ade20k, citys]')
parser.add_argument('--base-size', type=int, default=520,
                    help='base image size')
parser.add_argument('--crop-size', type=int, default=480,
                    help='crop image size')
parser.add_argument('--save-result', default=True,
                    help='save the predict')
parser.add_argument('--outdir', default='./eval', type=str,
                    help='path to save the predict result')
args = parser.parse_args()


def eval(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if config.save_result:
        if not os.path.exists(config.outdir):
            os.makedirs(config.outdir)
    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    # dataset and dataloader
    data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
    test_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)

    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=False)

    # create network
    model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                   pretrained=True, crop_size=args.crop_size).to(device)
    print('Finished loading model!')

    metric = SegmentationMetric(test_dataset.num_class)

    model.eval()
    for i, (image, label) in enumerate(test_loader):
        image = image.to(device)

        outputs = model(image)

        pred = torch.argmax(outputs[0], 1)
        pred = pred.cpu().data.numpy()
        label = label.numpy()

        metric.update(pred, label)
        pixAcc, mIoU = metric.get()
        print('Sample %d, validation pixAcc: %.3f%%, mIoU: %.3f%%' % (i + 1, pixAcc * 100, mIoU * 100))

        if config.save_result:
            predict = pred.squeeze(0)
            mask = get_color_pallete(predict, config.dataset)
            mask.save(os.path.join(config.outdir, 'seg_{}.png'.format(i)))


if __name__ == '__main__':
    print('Testing model: ', args.model)
    eval(args)

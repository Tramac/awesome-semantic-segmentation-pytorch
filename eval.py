from __future__ import print_function

import argparse
import os
import torch
import torch.utils.data as data
import numpy as np

from data_loader.transforms import TestTransform
from data_loader.voc import VOCSegmentation, VOC_PALETTE, MEAN, STD, VOC_CLASSES
from models.fcn import fcn32s
from utils.visualize import save_colorful_images, print_iou
from data_loader.functional import to_pil_image, denomalize
from utils.score import hist_info, compute_score

parser = argparse.ArgumentParser(
    description='Semantic Segmentation Evaluation')
parser.add_argument('--trained_model', default='./weights/VOC2012.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--dataset', default='VOC2012', choices=['VOC2007', 'VOC2012'],
                    type=str, help='VOC2007 or VOC2012')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--dataset_root', default='./datasets',
                    help='Dataset root directory path')
parser.add_argument('--num_classes', default=21, type=int,
                    help='Number of classes.')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = VOCSegmentation(root=args.dataset_root,
                                   year=args.dataset[3:],
                                   image_set='val',
                                   transform=TestTransform())

    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=False)

    model = fcn32s(args.num_classes).to(device)
    # model.load_state_dict(torch.load(args.trained_model, map_location='cpu'))
    model.load_state_dict({k.replace('module.',''):v for k, v in torch.load(args.trained_model, map_location='cpu').items()})
    print('Finished loading model!')

    model.eval()
    hist = np.zeros((args.num_classes, args.num_classes))
    correct = 0
    labeled = 0
    for i, (image, label) in enumerate(test_loader):
        image = image.to(device)

        outputs = model(image)
        _, pred = torch.max(outputs, 1)
        pred = pred.cpu().data.numpy()

        label = label.numpy()
        hist_tmp, labeled_tmp, correct_tmp = hist_info(pred.flatten(), label.flatten(), args.num_classes)
        hist += hist_tmp
        labeled += labeled_tmp
        correct += correct_tmp
        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct, labeled)
        print_iou(iu, mean_pixel_acc, VOC_CLASSES, False)
        img = to_pil_image(denomalize(image.squeeze(), MEAN, STD))
        img.save('./eval/img_%d.png' % i)
        save_colorful_images(pred, 'test_%d.png' % i, './eval', VOC_PALETTE)


if __name__ == '__main__':
    eval()

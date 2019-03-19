from __future__ import print_function

import os
import argparse
import numpy as np

import torch
import torch.utils.data as data

from data_loader.transforms import TestTransform
from data_loader.voc import VOCSegmentation
from models.model_zoo import get_model
from utils.score import SegmentationMetric

from utils.score import hist_info, compute_score
from utils.visualize import print_iou, save_colorful_images
from data_loader.voc import VOC_PALETTE, VOC_CLASSES, MEAN, STD
from data_loader.functional import denomalize, to_pil_image

parser = argparse.ArgumentParser(
    description='Semantic Segmentation Evaluation')
parser.add_argument('--model', type=str, default='fcn32s_vgg16',
                    help='model name (default: fcn32)')
parser.add_argument('--save-folder', default='./weights',
                    help='Directory for saving checkpoint models')
parser.add_argument('--dataset_root', default='./datasets',
                    help='Dataset root directory path')
parser.add_argument('--dataset', default='VOC2012', choices=['VOC2007', 'VOC2012'],
                    type=str, help='VOC2007 or VOC2012')
parser.add_argument('--num_classes', default=21, type=int,
                    help='Number of classes.')
parser.add_argument('--outdir', default='./eval', type=str,
                    help='path to save the predict result')
args = parser.parse_args()


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def eval(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(config.outdir):
        os.makedirs(config.outdir)
    # image transform
    input_transform = TestTransform([.485, .456, .406], [.229, .224, .225])

    # dataset and dataloader
    test_dataset = VOCSegmentation(root=args.dataset_root,
                                   year=args.dataset[3:],
                                   image_set='val',
                                   transform=input_transform)

    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=False)

    # create network
    model = get_model(config.model, num_classes=config.num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(config.save_folder, config.model + '_' + config.dataset + '.pth'),
                                     map_location='cpu'))
    print('Finished loading model!')

    metric = SegmentationMetric(config.num_classes)

    model.eval()
    hist = np.zeros((config.num_classes, config.num_classes))
    correct = 0
    labeled = 0
    for i, (image, label) in enumerate(test_loader):
        image = image.to(device)

        outputs = model(image)

        _, pred = torch.max(outputs, 1)
        pred = pred.data.numpy()

        label = label.numpy()
        metric.update(pred, label)
        pixAcc, mIoU = metric.get()
        print('mIoU: %.8f, pixAcc: %.8f' % (mIoU * 100, pixAcc * 100))
        hist_tmp, labeled_tmp, correct_tmp = hist_info(pred.flatten(), label.flatten(), config.num_classes)
        hist += hist_tmp
        labeled += labeled_tmp
        correct += correct_tmp
        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct, labeled)
        print_iou(iu, mean_pixel_acc, VOC_CLASSES, False)
        # img = to_pil_image(denomalize(image.squeeze(), MEAN, STD))
        # img.save('./eval/img_%d.png' % i)
        # save_colorful_images(pred, 'test_%d.png' % i, './eval', VOC_PALETTE)


if __name__ == '__main__':
    eval(args)

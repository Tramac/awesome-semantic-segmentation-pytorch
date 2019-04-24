from __future__ import print_function

import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.utils.data as data

from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.score import SegmentationMetric
from core.utils.visualize import get_color_pallete

from train import parse_args


def eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    outdir = 'test_result'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    # dataset and dataloader
    test_dataset = get_segmentation_dataset(args.dataset, split='val', mode='testval', transform=input_transform)

    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=False)

    # create network
    model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                   aux=args.aux, pretrained=True, pretrained_base=False).to(device)
    print('Finished loading model!')

    metric = SegmentationMetric(test_dataset.num_class)

    model.eval()
    for i, (image, label, filename) in enumerate(test_loader):
        image = image.to(device)

        with torch.no_grad():
            outputs = model(image)

            pred = torch.argmax(outputs[0], 1)
            pred = pred.cpu().data.numpy()
            label = label.numpy()

            metric.update(pred, label)
            pixAcc, mIoU = metric.get()
            print('Sample %d, validation pixAcc: %.3f%%, mIoU: %.3f%%' % (i + 1, pixAcc * 100, mIoU * 100))

            if args.save_result:
                predict = pred.squeeze(0)
                mask = get_color_pallete(predict, args.dataset)
                mask.save(os.path.join(outdir, os.path.splitext(filename)[0] + '.png'))


if __name__ == '__main__':
    args = parse_args()
    save_result = True
    args.save_result = save_result
    print('Testing model: ', args.model)
    eval(args)

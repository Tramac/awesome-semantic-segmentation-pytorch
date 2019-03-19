import os
import argparse
import torch

from torchvision import transforms
from PIL import Image
from utils.visualize import get_color_pallete
from models.model_zoo import get_model

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fcn32s_vgg16',
                    help='model name (default: fcn32_vgg16)')
parser.add_argument('--save-folder', default='./weights',
                    help='Directory for saving checkpoint models')
parser.add_argument('--dataset', default='VOC2012', choices=['VOC2007', 'VOC2012'],
                    type=str, help='VOC2007 or VOC2012')
parser.add_argument('--input-pic', type=str, default='./datasets/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg',
                    help='path to the input picture')
parser.add_argument('--num_classes', default=21, type=int,
                    help='Number of classes.')
parser.add_argument('--outdir', default='./eval', type=str,
                    help='path to save the predict result')
args = parser.parse_args()


def demo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(config.outdir):
        os.makedirs(config.outdir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(config.input_pic).convert('RGB')
    images = transform(image).unsqueeze(0).to(device)

    model = get_model(args.model, num_classes=config.num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(config.save_folder, args.model + '_' + args.dataset + '.pth'),
                                     map_location='cpu'))
    print('Finished loading model!')

    model.eval()
    with torch.no_grad():
        output = model(images)

    pred = torch.argmax(output, 1).squeeze(0).cpu().numpy()
    mask = get_color_pallete(pred, args.dataset)
    outname = os.path.splitext(os.path.split(args.input_pic)[-1])[0] + '.png'
    print(outname)
    mask.save(os.path.join(args.outdir, outname))


if __name__ == '__main__':
    demo(args)

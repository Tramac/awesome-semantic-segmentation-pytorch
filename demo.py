import argparse
import sys
import os
import torch

from models.model_zoo import get_model
from PIL import Image
from torchvision import transforms
from utils.visualize import save_colorful_images
from data_loader.voc import VOC_PALETTE

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '../..'))

parser = argparse.ArgumentParser(description='Predict segmentation result from a single image')
parser.add_argument('--model', type=str, default='fcn32s_vgg16',
                    help='model name (default: fcn32)')
parser.add_argument('--model_path', type=str, default=os.path.join(os.path.expanduser('~'), '.torch/models'),
                    help='path to save segmentation result')
parser.add_argument('--cuda', type=bool, default=False, help='use GPU')
parser.add_argument('--img_path', type=str, default='./datasets/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg',
                    help='path to the test image')
args = parser.parse_args()


def demo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open('./datasets/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg').convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    images = transform(image).unsqueeze(0).to(device)

    model = get_model(args.model, num_classes=21).to(device)
    model.load_state_dict(torch.load('./weights/fcn32s_vgg16VOC2012.pth', map_location='cpu'))
    print('Finished loading model!')

    model.eval()
    with torch.no_grad():
        output = model(images)

    pred = torch.argmax(output, 1).squeeze(0).cpu().numpy()
    save_colorful_images(pred, 'test.png', './eval', VOC_PALETTE)


if __name__ == '__main__':
    demo(args)

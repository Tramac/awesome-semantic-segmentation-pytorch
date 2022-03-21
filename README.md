# slightweight Segmentation

This project aims at providing a concise, easy-to-use, modifiable reference implementation for semantic segmentation models using PyTorch.

stage 1: # Installation

```
stage 2: # dependencies
pip install ninja tqdm

stage 3: # follow PyTorch installation in https://pytorch.org/get-started/locally/
conda install pytorch torchvision -c pytorch

stage 4: # for example, train swnet_resnet_citys:
python train.py --model swnet --backbone resnet --dataset citys --lr 0.0001 --epochs 50


stage 5: # for example, evaluate swnet_resnet_citys
python eval.py --model swnet --backbone resnet --dataset citys

### Demo
```
cd ./scripts
#for new users:
python demo.py --model swnet_resnet_citys --input-pic ../tests/test_img.jpg
#you should add 'test.jpg' by yourself
python demo.py --model swnet_resnet_citys --input-pic ../datasets/test.jpg
```

```
.{SEG_ROOT}
├── scripts
│   ├── demo.py
│   ├── eval.py
│   └── train.py
```


#### Dataset

You can run script to download dataset, such as:

```
cd ./core/data/downloader
python ade20k.py --download-dir ../datasets/ade
```
Acknowledgement: we thank the code support from "awesome-semantic-segmentation-pytorch (https://github.com/Tramac/Awesome-semantic-segmentation-pytorch)". The swnet is a improvement from enet.
 

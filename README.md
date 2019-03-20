# Awesome Semantic Segmentation on PyTorch

<p align="center"><img width="100%" src="datasets/cityscapes_demo_img.png" /></p>

--------------------------------------------------------------------------------
This project aims at providing a concise, easy-to-use, modular reference implementation for semantic segmentation models using PyTorch.

## Requisites
- [PyTorch 1.0](https://pytorch.org/get-started/locally/)
- Python 3.x

## Usage
#### Train
```
python train.py --model fcn32s_vgg16 --dataset pascal_voc
```
#### Evaluation
```
python eval.py --model fcn32s_vgg16 --dataset pascal_voc
```
#### Run Demo
```
python demo.py --model fcn32s_vgg16 --input-pic ./datasets/test.jpg
```
## Model Zoo & Datasets
#### Supported Model
- [FCN](https://arxiv.org/abs/1411.4038)

#### Supported Dataset
You can run script to download dataset, such as:
```
cd ./datasets
python ade20k.py --download-dir ./datasets/ade
```
- VOC2012, [download](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
- VOCAug, [download](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)
- ADK20K, [download](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
- Cityscapes, [download](https://www.cityscapes-dataset.com/downloads/)
- COCO

## Result
#### PASCAL VOC 2012
|Methods|Backbone|TrainSet|EvalSet|Mean IoU|pixAcc|
|:-:|:-:|:-:|:-:|:-:|:-:|
|FCN32s|VGG16|train|val|||

## To Do
- [x] Update VOC dataloader
- [ ] Add more semantic segmentation models
- [x] Add Cityscapes, ADE, COCO datasets

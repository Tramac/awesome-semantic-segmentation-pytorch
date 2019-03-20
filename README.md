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
python train.py
```
#### Evaluation
```
python eval.py
```
#### Run Demo
```
python demo.py
```
## Model Zoo & Datasets
#### Supported Model
- [FCN](https://arxiv.org/abs/1411.4038)

#### Supported Dataset
- VOC2012, [download](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
- ADK20K, [download](http://groups.csail.mit.edu/vision/datasets/ADE20K/)

## Result
#### PASCAL VOC 2012
|Methods|Backbone|TrainSet|EvalSet|Mean IoU|pixAcc|
|:-:|:-:|:-:|:-:|:-:|:-:|
|FCN32s|VGG16|train|val|||

## To Do
* [ ] Add more semantic segmentation models
* [ ] Add Cityscapes, ADE, COCO datasets

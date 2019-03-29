# Awesome Semantic Segmentation on PyTorch
This project aims at providing a concise, easy-to-use, modular reference implementation for semantic segmentation models using PyTorch.

<p align="center"><img width="100%" src="datasets/weimar_000091_000019_gtFine_color.png" /></p>

--------------------------------------------------------------------------------

## Update
- Add DANet
- Update MultiLoss

## Requisites
- [PyTorch 1.0](https://pytorch.org/get-started/locally/)
- Python 3.x

## Usage
#### Train
```
python train.py --model fcn32s --backbone vgg16 --dataset pascal_voc
```
#### Evaluation
```
python eval.py --model fcn32s --backbone vgg16 --dataset pascal_voc
```
#### Run Demo
```
python demo.py --model fcn32s_vgg16_voc --input-pic ./datasets/test.jpg
```
## Model Zoo & Datasets
#### Supported Model
- [FCN](https://arxiv.org/abs/1411.4038)
- [PSPNet](https://arxiv.org/pdf/1612.01105)
- [DeepLabv3](https://arxiv.org/abs/1706.05587)
- [DANet](https://arxiv.org/pdf/1809.02983)
- [DenseASPP](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf)

#### Supported Dataset
You can run script to download dataset, such as:
```
cd ./datasets
python ade20k.py --download-dir ./datasets/ade
```
- [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
- [VOCAug](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)
- [ADK20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
- [Cityscapes](https://www.cityscapes-dataset.com/downloads/)
- [COCO](http://cocodataset.org/#download)

## Result
#### PASCAL VOC 2012
|Methods|Backbone|TrainSet|EvalSet|crops_size|epochs|Mean IoU|pixAcc|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|FCN32s|vgg16|train|val|480|60|47.50%|85.39%|
|FCN16s|vgg16|train|val|480|60|49.16%|85.98%|
|FCN8s|vgg16|train|val|480|60|48.87%|85.02%|
|PSPNet|resnet50|train|val|480|60|63.44%|89.78%|
|DeepLabv3|resnet50|train|val|480|60|60.15%|88.36%|

## To Do
- [ ] Add more semantic segmentation models (in process)
- [ ] Train and evaluate
- [ ] Find difference between ```cuda``` and ```only cpu```
- [ ] Add DataParallelModel and DataParallelCriterion
- [ ] Add Synchronized BN

# Semantic Segmentation on PyTorch
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]

This project aims at providing a concise, easy-to-use, modifiable reference implementation for semantic segmentation models using PyTorch.

<p align="center"><img width="100%" src="docs/weimar_000091_000019_gtFine_color.png" /></p>

## Update
- Add OCNet
- Add ENet

## Requisites
- [PyTorch 1.0](https://pytorch.org/get-started/locally/)
- Python 3.x

## Usage
- **Train**
```
cd ./scripts
python train.py --model fcn32s --backbone vgg16 --dataset pascal_voc
```
- **Evaluation**
```
cd ./scripts
python eval.py --model fcn32s --backbone vgg16 --dataset pascal_voc
```
- **Run Demo**
```
cd ./scripts
python demo.py --model fcn32s_vgg16_voc --input-pic ./datasets/test.jpg
```

```
.{SEG_ROOT}
├── scripts
│   ├── demo.py
│   ├── eval.py
│   └── train.py
```

## Support

#### Model

- [FCN](https://arxiv.org/abs/1411.4038)
- [ENet](https://arxiv.org/pdf/1606.02147)
- [PSPNet](https://arxiv.org/pdf/1612.01105)
- [ICNet](https://arxiv.org/pdf/1704.08545)
- [DeepLabv3](https://arxiv.org/abs/1706.05587)
- [DenseASPP](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf)
- [EncNet](https://arxiv.org/abs/1803.08904v1)
- [BiSeNet](https://arxiv.org/abs/1808.00897)
- [DANet](https://arxiv.org/pdf/1809.02983)
- [OCNet](https://arxiv.org/pdf/1809.00916)
- [DUNet(DUpsampling)](https://arxiv.org/abs/1903.02120)

[DETAILS](https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/docs/DETAILS.md) for model & backbone.
```
.{SEG_ROOT}
├── core
│   ├── models
│   │   ├── bisenet.py
│   │   ├── danet.py
│   │   ├── deeplabv3.py
│   │   ├── denseaspp.py
│   │   ├── dunet.py
│   │   ├── encnet.py
│   │   ├── fcn.py
│   │   ├── pspnet.py
│   │   ├── icnet.py
│   │   ├── enet.py
│   │   ├── ocnet.py
│   │   ├── ......
```

#### Dataset

You can run script to download dataset, such as:

```
cd ./core/data/downloader
python ade20k.py --download-dir ../datasets/ade
```

- [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
- [VOCAug](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)
- [ADK20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
- [Cityscapes](https://www.cityscapes-dataset.com/downloads/)
- [COCO](http://cocodataset.org/#download)
- [SBU-shadow](http://www3.cs.stonybrook.edu/~cvl/content/datasets/shadow_db/SBU-shadow.zip)

```
.{SEG_ROOT}
├── core
│   ├── data
│   │   ├── dataloader
│   │   │   ├── ade.py
│   │   │   ├── cityscapes.py
│   │   │   ├── mscoco.py
│   │   │   ├── pascal_aug.py
│   │   │   ├── pascal_voc.py
│   │   │   ├── sbu_shadow.py
│   │   └── downloader
│   │       ├── ade20k.py
│   │       ├── cityscapes.py
│   │       ├── mscoco.py
│   │       ├── pascal_voc.py
│   │       └── sbu_shadow.py
```

## Result
- **PASCAL VOC 2012**

|Methods|Backbone|TrainSet|EvalSet|crops_size|epochs|Mean IoU|pixAcc|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|FCN32s|vgg16|train|val|480|60|47.50%|85.39%|
|FCN16s|vgg16|train|val|480|60|49.16%|85.98%|
|FCN8s|vgg16|train|val|480|60|48.87%|85.02%|
|PSPNet|resnet50|train|val|480|60|63.44%|89.78%|
|DeepLabv3|resnet50|train|val|480|60|60.15%|88.36%|

Note: The parameter settings of each method are different, including crop_size, learning rata, epochs, etc. For specific parameters, please see paper.

## Overfitting Test
See [TEST](https://github.com/Tramac/Awesome-semantic-segmentation-pytorch/tree/master/tests) for details.

```
.{SEG_ROOT}
├── tests
│   └── test_model.py
```

## To Do
- [ ] Add distributed training
- [ ] Add LIP dataset
- [x] Save the best model
- [x] Test DataParallel
- [ ] Add more semantic segmentation models (in process)
- [ ] Train and evaluate
- [x] Add Synchronized BN ([Why SyncBN?](https://tramac.github.io/2019/04/08/SyncBN/))

## References
- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)

<!--
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]
-->

[python-image]: https://img.shields.io/badge/Python-2.x|3.x-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.0-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
[lic-image]: http://dmlc.github.io/img/apache2.svg
[lic-url]: https://github.com/Tramac/Awesome-semantic-segmentation-pytorch/blob/master/LICENSE

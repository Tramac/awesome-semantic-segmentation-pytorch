# Semantic Segmentation on PyTorch

English | [简体中文](/README_zh-CN.md)

[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]

This project aims at providing a concise, easy-to-use, modifiable reference implementation for semantic segmentation models using PyTorch.

<p align="center"><img width="100%" src="docs/weimar_000091_000019_gtFine_color.png" /></p>

## Installation

```
# semantic-segmentation-pytorch dependencies
pip install ninja tqdm

# follow PyTorch installation in https://pytorch.org/get-started/locally/
conda install pytorch torchvision -c pytorch

# install PyTorch Segmentation
git clone https://github.com/Tramac/awesome-semantic-segmentation-pytorch.git
```

## Usage
### Train
-----------------
- **Single GPU training**
```
# for example, train fcn32_vgg16_pascal_voc:
python train.py --model fcn32s --backbone vgg16 --dataset pascal_voc --lr 0.0001 --epochs 50
```
- **Multi-GPU training**

```
# for example, train fcn32_vgg16_pascal_voc with 4 GPUs:
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --model fcn32s --backbone vgg16 --dataset pascal_voc --lr 0.0001 --epochs 50
```

### Evaluation
-----------------
- **Single GPU evaluating**
```
# for example, evaluate fcn32_vgg16_pascal_voc
python eval.py --model fcn32s --backbone vgg16 --dataset pascal_voc
```
- **Multi-GPU evaluating**
```
# for example, evaluate fcn32_vgg16_pascal_voc with 4 GPUs:
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS eval.py --model fcn32s --backbone vgg16 --dataset pascal_voc
```
### Demo
```
cd ./scripts
#for new users:
python demo.py --model fcn32s_vgg16_voc --input-pic ../tests/test_img.jpg
#you should add 'test.jpg' by yourself
python demo.py --model fcn32s_vgg16_voc --input-pic ../datasets/test.jpg
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
- [DeepLabv3+](https://arxiv.org/pdf/1802.02611)
- [DenseASPP](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf)
- [EncNet](https://arxiv.org/abs/1803.08904v1)
- [BiSeNet](https://arxiv.org/abs/1808.00897)
- [PSANet](https://hszhao.github.io/papers/eccv18_psanet.pdf)
- [DANet](https://arxiv.org/pdf/1809.02983)
- [OCNet](https://arxiv.org/pdf/1809.00916)
- [CGNet](https://arxiv.org/pdf/1811.08201)
- [ESPNetv2](https://arxiv.org/abs/1811.11431)
- [DUNet(DUpsampling)](https://arxiv.org/abs/1903.02120)
- [FastFCN(JPU)](https://arxiv.org/abs/1903.11816)
- [LEDNet](https://arxiv.org/abs/1905.02423)
- [Fast-SCNN](https://github.com/Tramac/Fast-SCNN-pytorch)
- [LightSeg](https://github.com/Tramac/Lightweight-Segmentation)
- [DFANet](https://arxiv.org/abs/1904.02216)

[DETAILS](https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/docs/DETAILS.md) for model & backbone.
```
.{SEG_ROOT}
├── core
│   ├── models
│   │   ├── bisenet.py
│   │   ├── danet.py
│   │   ├── deeplabv3.py
│   │   ├── deeplabv3+.py
│   │   ├── denseaspp.py
│   │   ├── dunet.py
│   │   ├── encnet.py
│   │   ├── fcn.py
│   │   ├── pspnet.py
│   │   ├── icnet.py
│   │   ├── enet.py
│   │   ├── ocnet.py
│   │   ├── psanet.py
│   │   ├── cgnet.py
│   │   ├── espnet.py
│   │   ├── lednet.py
│   │   ├── dfanet.py
│   │   ├── ......
```

#### Dataset

You can run script to download dataset, such as:

```
cd ./core/data/downloader
python ade20k.py --download-dir ../datasets/ade
```

|                           Dataset                            | training set | validation set | testing set |
| :----------------------------------------------------------: | :----------: | :------------: | :---------: |
| [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) |     1464     |      1449      |      ✘      |
| [VOCAug](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz) |    11355     |      2857      |      ✘      |
| [ADK20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/) |    20210     |      2000      |      ✘      |
| [Cityscapes](https://www.cityscapes-dataset.com/downloads/)  |     2975     |      500       |      ✘      |
| [COCO](http://cocodataset.org/#download)           |              |                |             |
| [SBU-shadow](http://www3.cs.stonybrook.edu/~cvl/content/datasets/shadow_db/SBU-shadow.zip) |     4085     |      638       |      ✘      |
| [LIP(Look into Person)](http://sysu-hcp.net/lip/)       |    30462     |     10000      |    10000    |

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

|Methods|Backbone|TrainSet|EvalSet|crops_size|epochs|JPU|Mean IoU|pixAcc|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|FCN32s|vgg16|train|val|480|60|✘|47.50|85.39|
|FCN16s|vgg16|train|val|480|60|✘|49.16|85.98|
|FCN8s|vgg16|train|val|480|60|✘|48.87|85.02|
|FCN32s|resnet50|train|val|480|50|✘|54.60|88.57|
|PSPNet|resnet50|train|val|480|60|✘|63.44|89.78|
|DeepLabv3|resnet50|train|val|480|60|✘|60.15|88.36|

Note: `lr=1e-4, batch_size=4, epochs=80`.

## Overfitting Test
See [TEST](https://github.com/Tramac/Awesome-semantic-segmentation-pytorch/tree/master/tests) for details.

```
.{SEG_ROOT}
├── tests
│   └── test_model.py
```

## To Do
- [x] add train script
- [ ] remove syncbn
- [ ] train & evaluate
- [x] test distributed training
- [x] fix syncbn ([Why SyncBN?](https://tramac.github.io/2019/02/25/%E8%B7%A8%E5%8D%A1%E5%90%8C%E6%AD%A5%20Batch%20Normalization[%E8%BD%AC]/))
- [x] add distributed ([How DIST?]("https://tramac.github.io/2019/03/06/%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83-PyTorch/"))
<!--
- [x] fix syncbn ([Why SyncBN?](https://tramac.github.io/2019/04/08/SyncBN/))
- [x] add distributed ([How DIST?](https://tramac.github.io/2019/04/22/%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83-PyTorch/))
-->
## References
- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
- [gloun-cv](https://github.com/dmlc/gluon-cv)
- [imagenet](https://github.com/pytorch/examples/tree/master/imagenet)



<!--
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]
-->

[python-image]: https://img.shields.io/badge/Python-2.x|3.x-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.1-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
[lic-image]: https://img.shields.io/badge/Apache-2.0-blue.svg
[lic-url]: https://github.com/Tramac/Awesome-semantic-segmentation-pytorch/blob/master/LICENSE

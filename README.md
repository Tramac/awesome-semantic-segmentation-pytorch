# Semantic Segmentation on PyTorch
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]

This project aims at providing a concise, easy-to-use, modifiable reference implementation for semantic segmentation models using PyTorch.

<p align="center"><img width="100%" src="docs/weimar_000091_000019_gtFine_color.png" /></p>

## Update
- add distributed training (Note: I have no enough device to test distributed, If you are interested in it, welcome to complete testing and fix bugs.)
- add OCNet

## Requisites
- Python 3.x
- [PyTorch 1.0](https://pytorch.org/get-started/locally/)
```
conda install pytorch torchvision -c pytorch
```
- Ninja
```
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 
```

## Usage
### Train
-----------------
- Single GPU training
```
# for example, train fcn32_vgg16_pascal_voc:
python train.py --model fcn32s --backbone vgg16 --dataset pascal_voc --lr 0.0001 --epochs 50
```
- Multi-GPU training

```
# for example, train fcn32_vgg16_pascal_voc with 4 GPUs:
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --model fcn32s --backbone vgg16 --dataset pascal_voc --lr 0.0001 --epochs 50
```
Note: The loss functions of EncNet and ICNet are special, `MixSoftmaxCrossEntropyLoss` need to be replaced by `EncNetLoss` and `ICNetLoss` in `train.py`, respectively.

### Evaluation
-----------------
- Single GPU training
```
# for example, evaluate fcn32_vgg16_pascal_voc
python eval.py --model fcn32s --backbone vgg16 --dataset pascal_voc
```
- Multi-GPU training
```
# for example, evaluate fcn32_vgg16_pascal_voc with 4 GPUs:
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS --model fcn32s --backbone vgg16 --dataset pascal_voc
```
### Demo
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

|                           Dataset                            | training set | validation set | testing set |
| :----------------------------------------------------------: | :----------: | :------------: | :---------: |
| [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) |     1464     |      1449      |      ✘      |
| [VOCAug](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz) |    11355     |      2857      |      ✘      |
| [ADK20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/) |    20210     |      2000      |      ✘      |
| [Cityscapes](https://www.cityscapes-dataset.com/downloads/)  |     2975     |      500       |      ✘      |
|           [COCO](http://cocodataset.org/#download)           |              |                |             |
| [SBU-shadow](http://www3.cs.stonybrook.edu/~cvl/content/datasets/shadow_db/SBU-shadow.zip) |     4085     |      638       |      ✘      |
|      [LIP(Look into Person)](http://sysu-hcp.net/lip/)       |    30462     |     10000      |    10000    |

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
- [ ] fix downloader
- [ ] optim loss
- [ ] test distributed training
- [x] add distributed training ([How DIST?](https://tramac.github.io/2019/04/22/%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83-PyTorch/))
- [x] add LIP dataset
- [ ] add more models (in process)
- [ ] train and evaluate
- [x] fix SyncBN ([Why SyncBN?](https://tramac.github.io/2019/04/08/SyncBN/))

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
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.0-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
[lic-image]: http://dmlc.github.io/img/apache2.svg
[lic-url]: https://github.com/Tramac/Awesome-semantic-segmentation-pytorch/blob/master/LICENSE

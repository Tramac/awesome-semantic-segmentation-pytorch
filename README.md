# slightweight Segmentation

This project aims at providing a concise, easy-to-use, modifiable reference implementation for semantic segmentation models using PyTorch.

stage 1: # Installation

stage 2: # dependencies
pip install ninja tqdm

stage 3: # follow PyTorch installation in https://pytorch.org/get-started/locally/
conda install pytorch torchvision -c pytorch

stage 4: # for example, train swnet_resnet_citys:
python train.py --model swnet --backbone resnet --dataset citys --lr 0.0001 --epochs 50


stage 5: # for example, evaluate swnet_resnet_citys
python eval.py --model swnet --backbone resnet --dataset citys

### Demo

cd ./scripts
#for new users:
python demo.py --model swnet_resnet_citys --input-pic ../tests/test_img.jpg
#you should add 'test.jpg' by yourself
python demo.py --model swnet_resnet_citys --input-pic ../datasets/test.jpg

### performance evaluation
```
![image](https://user-images.githubusercontent.com/43395674/159203398-86f4874e-7b0f-48a3-8414-cdf662d56f99.png)
![image](https://user-images.githubusercontent.com/43395674/159203405-7b656176-2e93-4d67-98e6-6d650204b0d6.png)
```
![image](https://user-images.githubusercontent.com/43395674/159203470-99a509cc-68cc-4fa4-be65-43e0c9204cb1.png)
![image](https://user-images.githubusercontent.com/43395674/159203480-10ff8f81-965f-419c-ab98-83fade7b3b65.png)
```
### a experiment on a simulative scene based on Jetson
![image](https://user-images.githubusercontent.com/43395674/159203486-19980424-c6c4-4644-a44b-9f52085b2067.png)
```

#### Dataset

You can run script to download dataset, such as:


cd ./core/data/downloader
python ade20k.py --download-dir ../datasets/ade

Acknowledgement: we thank the code support from "awesome-semantic-segmentation-pytorch (https://github.com/Tramac/Awesome-semantic-segmentation-pytorch)". The swnet is a improvement from enet.
 

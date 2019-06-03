#!/usr/bin/env bash

# train
export NGPUS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --model fcn32s \
    --backbone vgg16 --dataset pascal_voc \
    --lr 0.01 --epochs 80 --batch_size 16
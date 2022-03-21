#!/usr/bin/env bash

# train
CUDA_VISIBLE_DEVICES=0 python train.py --model enet \
    --backbone resnet50 --dataset citys \
    --lr 0.0001 --epochs 50

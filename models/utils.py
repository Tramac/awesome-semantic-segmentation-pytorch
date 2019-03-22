from __future__ import division

import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


def adjust_learning_rate(args, optimizer, epoch):
    """Reference by https://github.com/pytorch/examples/blob/master/imagenet/main.py"""
    if args.decay_mode == 'step':
        lr = args.lr * (args.gamma ** (epoch // args.decay_step))
    elif args.decay_mode == 'poly':
        lr = args.lr * (1. - epoch / args.epochs) ** 0.9
    else:
        raise TypeError('Unknown lr decay mode {}'.format(args.decay_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
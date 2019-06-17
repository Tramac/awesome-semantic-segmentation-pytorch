"""Popular Learning Rate Schedulers"""
from __future__ import division
import math
import torch

from bisect import bisect_right

__all__ = ['LRScheduler', 'WarmupMultiStepLR', 'WarmupPolyLR']


class LRScheduler(object):
    r"""Learning Rate Scheduler

    Parameters
    ----------
    mode : str
        Modes for learning rate scheduler.
        Currently it supports 'constant', 'step', 'linear', 'poly' and 'cosine'.
    base_lr : float
        Base learning rate, i.e. the starting learning rate.
    target_lr : float
        Target learning rate, i.e. the ending learning rate.
        With constant mode target_lr is ignored.
    niters : int
        Number of iterations to be scheduled.
    nepochs : int
        Number of epochs to be scheduled.
    iters_per_epoch : int
        Number of iterations in each epoch.
    offset : int
        Number of iterations before this scheduler.
    power : float
        Power parameter of poly scheduler.
    step_iter : list
        A list of iterations to decay the learning rate.
    step_epoch : list
        A list of epochs to decay the learning rate.
    step_factor : float
        Learning rate decay factor.
    """

    def __init__(self, mode, base_lr=0.01, target_lr=0, niters=0, nepochs=0, iters_per_epoch=0,
                 offset=0, power=0.9, step_iter=None, step_epoch=None, step_factor=0.1, warmup_epochs=0):
        super(LRScheduler, self).__init__()
        assert (mode in ['constant', 'step', 'linear', 'poly', 'cosine'])

        if mode == 'step':
            assert (step_iter is not None or step_epoch is not None)
        self.niters = niters
        self.step = step_iter
        epoch_iters = nepochs * iters_per_epoch
        if epoch_iters > 0:
            self.niters = epoch_iters
            if step_epoch is not None:
                self.step = [s * iters_per_epoch for s in step_epoch]

        self.step_factor = step_factor
        self.base_lr = base_lr
        self.target_lr = base_lr if mode == 'constant' else target_lr
        self.offset = offset
        self.power = power
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.mode = mode

    def __call__(self, optimizer, num_update):
        self.update(num_update)
        assert self.learning_rate >= 0
        self._adjust_learning_rate(optimizer, self.learning_rate)

    def update(self, num_update):
        N = self.niters - 1
        T = num_update - self.offset
        T = min(max(0, T), N)

        if self.mode == 'constant':
            factor = 0
        elif self.mode == 'linear':
            factor = 1 - T / N
        elif self.mode == 'poly':
            factor = pow(1 - T / N, self.power)
        elif self.mode == 'cosine':
            factor = (1 + math.cos(math.pi * T / N)) / 2
        elif self.mode == 'step':
            if self.step is not None:
                count = sum([1 for s in self.step if s <= T])
                factor = pow(self.step_factor, count)
            else:
                factor = 1
        else:
            raise NotImplementedError

        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            factor = factor * 1.0 * T / self.warmup_iters

        if self.mode == 'step':
            self.learning_rate = self.base_lr * factor
        else:
            self.learning_rate = self.target_lr + (self.base_lr - self.target_lr) * factor

    def _adjust_learning_rate(self, optimizer, lr):
        optimizer.param_groups[0]['lr'] = lr
        # enlarge the lr at the head
        for i in range(1, len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr * 10


# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
# reference: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/solver/lr_scheduler.py
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3,
                 warmup_iters=500, warmup_method="linear", last_epoch=-1):
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones)
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted got {}".format(warmup_method))

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_factor == 'linear':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, target_lr=0, max_iters=0, power=0.9, warmup_factor=1.0 / 3,
                 warmup_iters=500, warmup_method='linear', last_epoch=-1):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted "
                "got {}".format(warmup_method))

        self.target_lr = target_lr
        self.max_iters = max_iters
        self.power = power
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        N = self.max_iters - self.warmup_iters
        T = self.last_epoch - self.warmup_iters
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError("Unknown warmup type.")
            return [self.target_lr + (base_lr - self.target_lr) * warmup_factor for base_lr in self.base_lrs]
        factor = pow(1 - T / N, self.power)
        return [self.target_lr + (base_lr - self.target_lr) * factor for base_lr in self.base_lrs]


if __name__ == '__main__':
    import torch
    import torch.nn as nn

    model = nn.Conv2d(16, 16, 3, 1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = WarmupPolyLR(optimizer, niters=1000)

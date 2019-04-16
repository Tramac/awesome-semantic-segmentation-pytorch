##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Synchronized Cross-GPU Batch Normalization Module"""
import warnings
import torch

from torch.nn.modules.batchnorm import _BatchNorm
from queue import Queue
from .functions import *

__all__ = ['SyncBatchNorm', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d']


# Adopt from https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/nn/syncbn.py
class SyncBatchNorm(_BatchNorm):
    """Cross-GPU Synchronized Batch normalization (SyncBN)

    Parameters:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        sync: a boolean value that when set to ``True``, synchronize across
            different gpus. Default: ``True``
        activation : str
            Name of the activation functions, one of: `leaky_relu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Reference:
        .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." *ICML 2015*
        .. [2] Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, and Amit Agrawal. "Context Encoding for Semantic Segmentation." *CVPR 2018*
    Examples:
        >>> m = SyncBatchNorm(100)
        >>> net = torch.nn.DataParallel(m)
        >>> output = net(input)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, sync=True, activation='none', slope=0.01, inplace=True):
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=True)
        self.activation = activation
        self.inplace = False if activation == 'none' else inplace
        self.slope = slope
        self.devices = list(range(torch.cuda.device_count()))
        self.sync = sync if len(self.devices) > 1 else False
        # Initialize queues
        self.worker_ids = self.devices[1:]
        self.master_queue = Queue(len(self.worker_ids))
        self.worker_queues = [Queue(1) for _ in self.worker_ids]

    def forward(self, x):
        # resize the input to (B, C, -1)
        input_shape = x.size()
        x = x.view(input_shape[0], self.num_features, -1)
        if x.get_device() == self.devices[0]:
            # Master mode
            extra = {
                "is_master": True,
                "master_queue": self.master_queue,
                "worker_queues": self.worker_queues,
                "worker_ids": self.worker_ids
            }
        else:
            # Worker mode
            extra = {
                "is_master": False,
                "master_queue": self.master_queue,
                "worker_queue": self.worker_queues[self.worker_ids.index(x.get_device())]
            }
        if self.inplace:
            return inp_syncbatchnorm(x, self.weight, self.bias, self.running_mean, self.running_var,
                                     extra, self.sync, self.training, self.momentum, self.eps,
                                     self.activation, self.slope).view(input_shape)
        else:
            return syncbatchnorm(x, self.weight, self.bias, self.running_mean, self.running_var,
                                 extra, self.sync, self.training, self.momentum, self.eps,
                                 self.activation, self.slope).view(input_shape)

    def extra_repr(self):
        if self.activation == 'none':
            return 'sync={}'.format(self.sync)
        else:
            return 'sync={}, act={}, slope={}, inplace={}'.format(
                self.sync, self.activation, self.slope, self.inplace)


class BatchNorm1d(SyncBatchNorm):
    """BatchNorm1d is deprecated in favor of :class:`core.nn.sync_bn.SyncBatchNorm`."""

    def __init__(self, *args, **kwargs):
        warnings.warn("core.nn.sync_bn.{} is now deprecated in favor of core.nn.sync_bn.{}."
                      .format('BatchNorm1d', SyncBatchNorm.__name__), DeprecationWarning)
        super(BatchNorm1d, self).__init__(*args, **kwargs)


class BatchNorm2d(SyncBatchNorm):
    """BatchNorm1d is deprecated in favor of :class:`core.nn.sync_bn.SyncBatchNorm`."""

    def __init__(self, *args, **kwargs):
        warnings.warn("core.nn.sync_bn.{} is now deprecated in favor of core.nn.sync_bn.{}."
                      .format('BatchNorm2d', SyncBatchNorm.__name__), DeprecationWarning)
        super(BatchNorm2d, self).__init__(*args, **kwargs)


class BatchNorm3d(SyncBatchNorm):
    """BatchNorm1d is deprecated in favor of :class:`core.nn.sync_bn.SyncBatchNorm`."""

    def __init__(self, *args, **kwargs):
        warnings.warn("core.nn.sync_bn.{} is now deprecated in favor of core.nn.sync_bn.{}."
                      .format('BatchNorm3d', SyncBatchNorm.__name__), DeprecationWarning)
        super(BatchNorm3d, self).__init__(*args, **kwargs)

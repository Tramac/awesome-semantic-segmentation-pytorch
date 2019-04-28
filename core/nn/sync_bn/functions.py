##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Synchronized Cross-GPU Batch Normalization functions"""
import torch.cuda.comm as comm

from torch.autograd import Function
from torch.autograd.function import once_differentiable
from core.nn.sync_bn import lib

__all__ = ['syncbatchnorm', 'inp_syncbatchnorm']


class syncbatchnorm_(Function):
    @classmethod
    def forward(cls, ctx, x, gamma, beta, running_mean, running_var,
                extra, sync=True, training=True, momentum=0.1, eps=1e-05,
                activation="none", slope=0.01):
        # save context
        cls._parse_extra(ctx, extra)
        ctx.sync = sync
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        assert activation == 'none'

        # continous inputs
        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()

        if ctx.training:
            if x.is_cuda:
                _ex, _exs = lib.gpu.expectation_forward(x)
            else:
                raise NotImplemented

            if ctx.sync:
                if ctx.is_master:
                    _ex, _exs = [_ex.unsqueeze(0)], [_exs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _ex_w, _exs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _ex.append(_ex_w.unsqueeze(0))
                        _exs.append(_exs_w.unsqueeze(0))

                    _ex = comm.gather(_ex).mean(0)
                    _exs = comm.gather(_exs).mean(0)

                    tensors = comm.broadcast_coalesced((_ex, _exs), [_ex.get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_ex, _exs))
                    _ex, _exs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()

            # Update running stats
            _var = _exs - _ex ** 2
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * _ex)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * _var)

            # Mark in-place modified tensors
            ctx.mark_dirty(running_mean, running_var)
        else:
            _ex, _var = running_mean.contiguous(), running_var.contiguous()
            _exs = _var + _ex ** 2

        # BN forward
        if x.is_cuda:
            y = lib.gpu.batchnorm_forward(x, _ex, _exs, gamma, beta, ctx.eps)
        else:
            y = lib.cpu.batchnorm_forward(x, _ex, _exs, gamma, beta, ctx.eps)

        # Output
        ctx.save_for_backward(x, _ex, _exs, gamma, beta)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        x, _ex, _exs, gamma, beta = ctx.saved_tensors
        dz = dz.contiguous()

        # BN backward
        if dz.is_cuda:
            dx, _dex, _dexs, dgamma, dbeta = lib.gpu.batchnorm_backward(dz, x, _ex, _exs, gamma, beta, ctx.eps)
        else:
            raise NotImplemented

        if ctx.training:
            if ctx.sync:
                if ctx.is_master:
                    _dex, _dexs = [_dex.unsqueeze(0)], [_dexs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _dex_w, _dexs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _dex.append(_dex_w.unsqueeze(0))
                        _dexs.append(_dexs_w.unsqueeze(0))

                    _dex = comm.gather(_dex).mean(0)
                    _dexs = comm.gather(_dexs).mean(0)

                    tensors = comm.broadcast_coalesced((_dex, _dexs), [_dex.get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_dex, _dexs))
                    _dex, _dexs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()

            if x.is_cuda:
                dx_ = lib.gpu.expectation_backward(x, _dex, _dexs)
            else:
                raise NotImplemented
            dx = dx + dx_

        return dx, dgamma, dbeta, None, None, None, None, None, None, None, None, None

    @staticmethod
    def _parse_extra(ctx, extra):
        ctx.is_master = extra["is_master"]
        if ctx.is_master:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queues = extra["worker_queues"]
            ctx.worker_ids = extra["worker_ids"]
        else:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queue = extra["worker_queue"]


def _act_forward(ctx, x):
    if ctx.activation.lower() == "leaky_relu":
        if x.is_cuda:
            lib.gpu.leaky_relu_forward(x, ctx.slope)
        else:
            raise NotImplemented
    else:
        assert ctx.activation == 'none'


def _act_backward(ctx, x, dx):
    if ctx.activation.lower() == "leaky_relu":
        if x.is_cuda:
            lib.gpu.leaky_relu_backward(x, dx, ctx.slope)
        else:
            raise NotImplemented
    else:
        assert ctx.activation == 'none'


class inp_syncbatchnorm_(Function):
    @classmethod
    def forward(cls, ctx, x, gamma, beta, running_mean, running_var,
                extra, sync=True, training=True, momentum=0.1, eps=1e-5,
                activation='none', slope=0.01):
        # save context
        cls._parse_extra(ctx, extra)
        ctx.sync = sync
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope

        # continous inputs
        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()

        if ctx.training:
            if x.is_cuda:
                _ex, _exs = lib.gpu.expectation_forward(x)
            else:
                raise NotImplemented

            if ctx.sync:
                if ctx.is_master:
                    _ex, _exs = [_ex.unsqueeze(0)], [_exs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _ex_w, _exs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _ex.append(_ex_w.unsqueeze(0))
                        _exs.append(_exs_w.unsuqeeze(0))

                    _ex = comm.gather(_ex).mean(0)
                    _exs = comm.gather(_exs).mean(0)

                    tensors = comm.broadcast_coalesced((_ex, _exs), [_ex.get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_ex, _exs))
                    _ex, _exs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()

            # Update running stats
            _var = _exs - _ex ** 2
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * _ex)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * _var)

            # Mark in-place modified tensors
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            _ex, _var = running_mean.contiguous(), running_var.contiguous()
            _exs = _var + _ex ** 2
            ctx.mark_dirty(x)

        # BN forward + activation
        if x.is_cuda:
            lib.gpu.batchnorm_inp_forward(x, _ex, _exs, gamma, beta, ctx.eps)
        else:
            raise NotImplemented

        _act_forward(ctx, x)

        # Output
        ctx.save_for_backward(x, _ex, _exs, gamma, beta)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, _ex, _exs, gamma, beta = ctx.saved_tensors
        dz = dz.contiguous()

        # Undo activation
        _act_backward(ctx, z, dz)

        # BN backward
        if dz.is_cuda:
            dx, _dex, _dexs, dgamma, dbeta = lib.gpu.batchnorm_inp_backward(dz, z, _ex, _exs, gamma, beta, ctx.eps)
        else:
            raise NotImplemented

        if ctx.training:
            if ctx.sync:
                if ctx.is_master:
                    _dex, _dexs = [_dex.unsqueeze(0)], [_dexs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _dex_w, _dexs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _dex.append(_dex_w.unsqueeze(0))
                        _dexs.append(_dexs_w.unsqueeze(0))

                    _dex = comm.gather(_dex).mean(0)
                    _dexs = comm.gather(_dexs).mean(0)

                    tensors = comm.broadcast_coalesced((_dex, _dexs), [_dex.get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_dex, _dexs))
                    _dex, _dexs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()

            if z.is_cuda:
                lib.gpu.expectation_inp_backward(dx, z, _dex, _dexs, _ex, _exs, gamma, beta, ctx.eps)
            else:
                raise NotImplemented

        return dx, dgamma, dbeta, None, None, None, None, None, None, None, None, None

    @staticmethod
    def _parse_extra(ctx, extra):
        ctx.is_master = extra["is_master"]
        if ctx.is_master:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queues = extra["worker_queues"]
            ctx.worker_ids = extra["worker_ids"]
        else:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queue = extra["worker_queue"]


syncbatchnorm = syncbatchnorm_.apply
inp_syncbatchnorm = inp_syncbatchnorm_.apply

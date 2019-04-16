import core
import torch
import numpy as np

from torch.autograd import Variable

EPS = 1e-3
ATOL = 1e-3


def _assert_tensor_close(a, b, atol=ATOL, rtol=EPS):
    npa, npb = a.cpu().numpy(), b.cpu().numpy()
    assert np.allclose(npa, npb, rtol=rtol, atol=atol), \
        'Tensor close check failed\n{}\n{}\nadiff={}, rdiff={}'.format(
            a, b, np.abs(npa - npb).max(), np.abs((npa - npb) / np.fmax(npa, 1e-5)).max())


def testSyncBN():
    def _check_batchnorm_result(bn1, bn2, input, is_train, cuda=False):
        def _find_bn(module):
            for m in module.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                  core.nn.SyncBatchNorm)):
                    return m

        def _syncParameters(bn1, bn2):
            bn1.reset_parameters()
            bn2.reset_parameters()
            if bn1.affine and bn2.affine:
                bn2.weight.data.copy_(bn1.weight.data)
                bn2.bias.data.copy_(bn1.bias.data)
                bn2.running_mean.copy_(bn1.running_mean)
                bn2.running_var.copy_(bn1.running_var)

        bn1.train(mode=is_train)
        bn2.train(mode=is_train)

        if cuda:
            input = input.cuda()
        # using the same values for gamma and beta
        _syncParameters(_find_bn(bn1), _find_bn(bn2))

        input1 = Variable(input.clone().detach(), requires_grad=True)
        input2 = Variable(input.clone().detach(), requires_grad=True)
        if is_train:
            bn1.train()
            bn2.train()
            output1 = bn1(input1)
            output2 = bn2(input2)
        else:
            bn1.eval()
            bn2.eval()
            with torch.no_grad():
                output1 = bn1(input1)
                output2 = bn2(input2)
        # assert forwarding
        # _assert_tensor_close(input1.data, input2.data)
        _assert_tensor_close(output1.data, output2.data)
        if not is_train:
            return
        (output1 ** 2).sum().backward()
        (output2 ** 2).sum().backward()
        _assert_tensor_close(_find_bn(bn1).bias.grad.data, _find_bn(bn2).bias.grad.data)
        _assert_tensor_close(_find_bn(bn1).weight.grad.data, _find_bn(bn2).weight.grad.data)
        _assert_tensor_close(input1.grad.data, input2.grad.data)
        _assert_tensor_close(_find_bn(bn1).running_mean, _find_bn(bn2).running_mean)
        # _assert_tensor_close(_find_bn(bn1).running_var, _find_bn(bn2).running_var)

    bn = torch.nn.BatchNorm2d(10).cuda().double()
    sync_bn = core.nn.SyncBatchNorm(10, inplace=True, sync=True).cuda().double()
    sync_bn = torch.nn.DataParallel(sync_bn).cuda()
    # check with unsync version
    # _check_batchnorm_result(bn, sync_bn, torch.rand(2, 1, 2, 2).double(), True, cuda=True)
    for i in range(10):
        print(i)
        _check_batchnorm_result(bn, sync_bn, torch.rand(16, 10, 16, 16).double(), True, cuda=True)
        # _check_batchnorm_result(bn, sync_bn, torch.rand(16, 10, 16, 16).double(), False, cuda=True)


if __name__ == '__main__':
    import nose

    nose.runmodule()

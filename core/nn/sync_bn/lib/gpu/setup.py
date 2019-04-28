from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='syncbn_gpu',
    ext_modules=[
        CUDAExtension('sync_gpu', [
            'operator.cpp',
            'activation_kernel.cu',
            'syncbn_kernel.cu',
            ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
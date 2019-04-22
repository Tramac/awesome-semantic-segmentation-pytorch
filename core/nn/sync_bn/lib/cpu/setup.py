from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='syncbn_cpu',
    ext_modules=[
        CppExtension('syncbn_cpu', [
            'operator.cpp',
            'syncbn_cpu.cpp',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# !/usr/bin/env python
# reference: https://github.com/facebookresearch/maskrcnn-benchmark/blob/90c226cf10e098263d1df28bda054a5f22513b4f/setup.py

import os
import glob
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

requirements = ["torch"]


def get_extension():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
        )
    ]

    return ext_modules


setup(
    name="semantic_segmentation",
    version="0.1",
    author="tramac",
    description="semantic segmentation in pytorch",
    ext_modules=get_extension(),
    cmdclass={"build_ext": BuildExtension}
)
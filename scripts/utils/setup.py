#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize('split_to_patches.pyx'),
    include_dirs=[numpy.get_include()]
)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

from chainer import cuda
from chainer import function
from chainer.functions import Softmax
from chainer.utils import type_check


class CIS(function.Function):

    def __init__(self, c=0, use_cudnn=True):
        self.c = c
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            t_type.dtype == numpy.int32,
            x_type.ndim == 4,
            t_type.ndim == 3,
            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2:] == t_type.shape[1:]
        )

    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        x, t = inputs
        x[:, self.c, :, :] = 0
        self.y, = Softmax(self.use_cudnn).forward((x,))
        self.count = t.shape[0]
        y = cupy.rollaxis(self.y, 1, self.y.ndim)
        ret = cuda.reduce(
            'S t, raw T y, int32 n_channel, T inv_count', 'T out',
            'log(y[_j * n_channel + t])',
            'a + b', 'out = a * inv_count', '0', 'cis_fwd'
        )(t, y.reduced_view(), y.shape[-1], -1.0 / t.shape[0])
        return ret,

    def backward_gpu(self, inputs, grad_outputs):
        cupy = cuda.cupy
        x, t = inputs
        gloss = grad_outputs[0]
        n_unit = t.size // t.shape[0]
        coeff = cuda.cupy.divide(gloss, self.count, dtype=gloss.dtype)
        gx = cuda.elementwise(
            'T y, S t, raw T coeff, S n_channel, S n_unit, S inhibit',
            'T gx',
            '''
            const int c = (i / n_unit % n_channel);
            if (c == inhibit) {
                gx = 0;
            } else {
                gx = coeff[0] * (y - (c == t));
            }
            ''',
            'cis_bwd')(self.y, cupy.expand_dims(t, 1), coeff, x.shape[1],
                       n_unit, self.c)
        return gx, None


def cis(x, t, c=0, use_cudnn=True):
    """Channel-wise Inhibited Softmax loss.

    It calculates Channel-wise Inhibited Softmax for regularizing inputs to
    softmax function.

    :math:


    """
    return CIS(c, use_cudnn)(x, t)

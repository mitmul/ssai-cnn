#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from cis import cis


class TestCIS(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (10, 3, 5, 5)).astype(numpy.float32)
        self.t = numpy.random.randint(
            0, 3, (10, 1, 5, 5,)).astype(numpy.int32)
        self.c = 0

    def check_forward(self, x_data, t_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        loss = cis(x, t, self.c, use_cudnn)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = float(cuda.to_cpu(loss.data))

        # Compute expected value
        self.x[:, self.c, :, :] = 0
        y = numpy.exp(self.x)
        print('y:', y.shape)
        loss_expect = 0.0
        for i in six.moves.range(y.shape[0]):
            y[i] = y[i] / y[i].sum(axis=0)
            for yy in six.moves.range(y.shape[2]):
                for xx in six.moves.range(y.shape[3]):
                    tt = t_data[i, 0, yy, xx]
                    loss = y[i, tt, yy, xx]
                    loss_expect -= math.log(loss)
        loss_expect /= y.shape[0]

        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    @attr.cudnn
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), False)

    def check_backward(self, x_data, t_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        loss = cis(x, t, self.c, use_cudnn)
        loss.backward()
        self.assertEqual(None, t.grad)

        func = loss.creator
        f = lambda: func.forward((x.data, t.data))
        gx, = gradient_check.numerical_grad(f, (x.data,), (1,), eps=0.02)

        gradient_check.assert_allclose(gx, x.grad, atol=1e-4)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), False)


testing.run_module(__name__, __file__)

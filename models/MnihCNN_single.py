#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Chain
import chainer.links as L
import chainer.functions as F


class MnihCNN_single(Chain):

    def __init__(self):
        super(MnihCNN_single, self).__init__(
            conv1=L.Convolution2D(3, 64, 15, stride=4, pad=7),
            conv2=L.Convolution2D(64, 112, 5, stride=1, pad=2),
            conv3=L.Convolution2D(112, 80, 3, stride=1, pad=1),
            conv4=L.Convolution2D(80, 1, 3, stride=1, pad=1),
        )
        self.train = True

    def __call__(self, x, t):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        self.pred = self.conv4(h)

        if t is not None:
            self.loss = F.sigmoid_cross_entropy(self.pred, t)
            return self.loss
        else:
            self.pred = F.sigmoid(self.pred)
            return self.pred

model = MnihCNN_single()

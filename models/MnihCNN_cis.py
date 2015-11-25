#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, 'functions')
import chainer.links as L
import chainer.functions as F
from chainer import Chain
from cis import cis


class MnihCNN_cis(Chain):

    def __init__(self):
        super(MnihCNN_cis, self).__init__(
            conv1=L.Convolution2D(3, 64, 16, stride=4, pad=0),
            conv2=L.Convolution2D(64, 112, 4, stride=1, pad=0),
            conv3=L.Convolution2D(112, 80, 3, stride=1, pad=0),
            fc4=L.Linear(5120, 4096),
            fc5=L.Linear(4096, 768),
        )
        self.train = True
        self.c = 0  # inhibit channel 0

    def __call__(self, x, t):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))
        self.pred = F.reshape(h, (x.data.shape[0], 3, 16, 16))

        if t is not None:
            self.loss = cis(self.pred, t, self.c)
            self.loss /= 16 * 16
            return self.loss
        else:
            self.pred = F.softmax(self.pred)
            return self.pred

model = MnihCNN_cis()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer.links as L
import chainer.functions as F
from chainer import Chain
from chainer import cuda
from chainer import Variable


class MnihCNN_cis(Chain):

    def __init__(self):
        super(MnihCNN_cis, self).__init__(
            conv1=L.Convolution2D(3, 64, 16, stride=4, pad=0),
            conv2=L.Convolution2D(64, 112, 4, stride=1, pad=0),
            conv3=L.Convolution2D(112, 80, 3, stride=1, pad=0),
            fc4=L.Linear(3920, 4096),
            fc5=L.Linear(4096, 768),
        )
        self.train = True
        self.c = 0  # inhibit channel 0

    def channelwise_inhibited(self, h, train):
        xp = cuda.get_array_module(h.data)
        num = h.data.shape[0]

        h = F.split_axis(h, 3, 1)
        c = F.reshape(h[self.c], (num, 16, 16))
        z = Variable(xp.zeros_like(c.data), volatile=not train)
        c = F.batch_matmul(c, z)
        c = F.reshape(c, (num, 1, 16, 16))
        hs = []
        for i, s in enumerate(h):
            if i == self.c:
                hs.append(c)
            else:
                hs.append(s)
        return F.concat(hs, 1)

    def __call__(self, x, t):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2, 1)
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.fc4(h))
        h = self.fc5(h)
        h = F.reshape(h, (x.data.shape[0], 3, 16, 16))
        h = self.channelwise_inhibited(h, self.train)

        if self.train:
            self.loss = F.softmax_cross_entropy(h, t, normalize=False)
            return self.loss
        else:
            self.pred = F.softmax(h)
            return self.pred

    def middle_layers(self, x):
        middles = []
        middles.append((self.conv1.name, F.relu(self.conv1(x))))
        h = F.max_pooling_2d(middles[-1][-1], 2, 1)
        middles.append((self.conv2.name, F.relu(self.conv2(h))))
        middles.append((self.conv3.name, F.relu(self.conv3(middles[-1][-1]))))

        h = F.relu(self.fc4(middles[-1][-1]))
        h = self.fc5(h)
        h = F.reshape(h, (x.data.shape[0], 3, 16, 16))
        middles.append(('reshape', h))
        h = self.channelwise_inhibited(h, False)
        self.pred = F.softmax(h)
        middles.append(('pred', self.pred))

        return middles

model = MnihCNN_cis()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Chain
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from chainer import Variable


class VGG_cis(Chain):

    def __init__(self):
        super(VGG_cis, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            fc6=L.Linear(2048, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 768),
        )
        self.train = True
        self.c = 0  # inhibit channel 0

    def __call__(self, x, t):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)

        # Channelwise Inhibited
        h = F.split_axis(h, 3, 1)
        c = F.reshape(h[self.c], (x.data.shape[0], 16, 16))
        xp = cuda.get_array_module(x.data)
        z = Variable(xp.zeros_like(c.data))
        c = F.batch_matmul(c, z)
        c = F.reshape(c, (x.data.shape[0], 1, 16, 16))
        hs = []
        for i, s in enumerate(h):
            if i == self.c:
                hs.append(c)
            else:
                hs.append(s)
        self.pred = F.concat(hs, 1)

        if t is not None:
            self.loss = F.softmax_cross_entropy(self.pred, t)
            self.loss /= 16 * 16
            return self.loss
        else:
            self.pred = F.softmax(self.pred)
            return self.pred

model = VGG_cis()

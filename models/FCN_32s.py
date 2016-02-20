#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L


class FCN_32s(chainer.Chain):

    def __init__(self):
        super(FCN_32s, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, pad=100),
            conv1_2=L.Convolution2D(64, 64, 3),
            conv2_1=L.Convolution2D(64, 128, 3),
            conv2_2=L.Convolution2D(128, 128, 3),
            conv3_1=L.Convolution2D(128, 256, 3),
            conv3_2=L.Convolution2D(256, 256, 3),
            conv4_1=L.Convolution2D(256, 512, 3),
            conv4_2=L.Convolution2D(512, 512, 3),
            conv4_3=L.Convolution2D(512, 512, 3),
            conv5_1=L.Convolution2D(512, 512, 3),
            conv5_2=L.Convolution2D(512, 512, 3),
            conv5_3=L.Convolution2D(512, 512, 3),
            fc6=L.Convolution2D(512, 4096, 7),
            fc7=L.Convolution2D(4096, 4096, 1),
            score_fr=L.Convolution2D(4096, 21, 1),
            upsample=L.Deconvolution2D(21, 21, 64, 32),
        )
        self.train = True

    def __call__(self, x, t):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(F.relu(self.fc6(h)), ratio=0.5, train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), ratio=0.5, train=self.train)
        h = self.score_fr(h)
        h = self.upsample(h)

        return h

model = FCN_32s()

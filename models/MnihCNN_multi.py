#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L


class MnihCNN_multi(chainer.Chain):

    def __init__(self):
        super(MnihCNN_multi, self).__init__(
            conv1=L.Convolution2D(3, 64, 16, stride=4, pad=0),
            conv2=L.Convolution2D(64, 112, 4, stride=1, pad=0),
            conv3=L.Convolution2D(112, 80, 3, stride=1, pad=0),
            fc4=L.Linear(3920, 4096),
            fc5=L.Linear(4096, 768),
        )
        self.train = True

    def __call__(self, x, t):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2, 1)
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.dropout(F.relu(self.fc4(h)), train=self.train)
        h = self.fc5(h)
        h = F.reshape(h, (x.data.shape[0], 3, 16, 16))

        if t is not None:
            self.loss = F.softmax_cross_entropy(h, t, normalize=False)
            return self.loss
        else:
            self.pred = F.softmax(h)
            return self.pred

    def middle_layers(self, x):
        middles = []

        h = self.conv1(x)
        middles.append((self.conv1.name, h))

        h = F.relu(h)
        middles.append(('relu1', h))

        h = F.max_pooling_2d(h, 2, 1)
        middles.append(('mpool1', h))

        h = self.conv2(h)
        middles.append((self.conv2.name, h))

        h = F.relu(h)
        middles.append(('relu2', h))

        h = self.conv3(h)
        middles.append((self.conv3.name, h))

        h = F.relu(h)
        middles.append(('relu3', h))

        h = self.fc4(h)
        middles.append(('fc4', h))

        h = F.relu(h)
        middles.append(('relu4', h))

        h = self.fc5(h)
        middles.append(('fc5', h))

        h = F.reshape(h, (x.data.shape[0], 3, 16, 16))
        middles.append(('reshape', h))

        self.pred = F.softmax(h)
        middles.append(('pred', self.pred))

        return middles

model = MnihCNN_multi()

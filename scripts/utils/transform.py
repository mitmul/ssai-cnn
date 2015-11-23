#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from skimage.transform import rotate


class Transform(object):

    def __init__(self, args):
        self.args = args

    def transform(self, ortho, label):
        self.ortho = ortho
        self.label = label

        if self.args.fliplr == 1:
            self.random_fliplr()

        if self.args.rotate == 1:
            self.random_rotate()

        if self.args.crop == 1:
            self.crop_center()

        if self.ortho.dtype != np.float32:
            self.ortho = self.ortho.astype(np.float32)

        if self.label.dtype != np.int32:
            self.label = self.label.astype(np.int32)

        if self.args.norm == 1:
            # global contrast normalization
            self.ortho -= self.ortho.reshape(-1, 3).mean(axis=0)
            self.ortho /= self.ortho.reshape(-1, 3).std(axis=0) + 1e-5

        return self.ortho, self.label

    def random_fliplr(self):
        if np.random.randint(2) == 1:
            self.ortho = np.fliplr(self.ortho)
            self.label = np.fliplr(self.label)

    def random_rotate(self):
        angle = np.random.randint(self.args.angle)
        self.ortho = rotate(self.ortho, angle, order=0)
        self.label = rotate(self.label.astype(np.float), angle, order=0)

    def crop_center(self):
        oh, ow, _ = self.ortho.shape
        self.ortho = self.ortho[oh / 2 - self.args.ortho_side / 2:
                                oh / 2 + self.args.ortho_side / 2,
                                ow / 2 - self.args.ortho_side / 2:
                                ow / 2 + self.args.ortho_side / 2, :]
        lh, lw = self.label.shape
        self.label = self.label[lh / 2 - self.args.label_side / 2:
                                lh / 2 + self.args.label_side / 2,
                                lw / 2 - self.args.label_side / 2:
                                lw / 2 + self.args.label_side / 2]
